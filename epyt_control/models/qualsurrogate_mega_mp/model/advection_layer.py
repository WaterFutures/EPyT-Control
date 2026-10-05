import torch
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from collections.abc import Iterable
# AdvectionMPNN
#    - Msg-Fn     --> (msg_raw, delay_steps, flows)
#    - Agg-Fn     --> (msg_raw, delay_steps, flows)
#   (- AdvectionDynamicEdgeMask; optional)


class AdvectionLayer(MessagePassing):
    
    def __init__(self, advection_op, mixing_at_nodes=True, **kwargs):
        super().__init__(aggr='sum', node_dim=0, **kwargs)
        self.advection_op = advection_op
        self.mixing_at_nodes = mixing_at_nodes

    def message(self, x_i, x_j, warp_map, weight, sl_mask, xs_map=None, **kwargs):
        weight = weight.unsqueeze(-1)

        adv_x_i = self.advection_op(x_i, warp_map)
        adv_x_j = self.advection_op(x_j, warp_map)

        msg_advected = (adv_x_j * (1.0 - sl_mask) + adv_x_i * sl_mask) * weight

        if xs_map is not None:
            x_interp = torch.stack((x_i, x_j), -1).transpose(2,1) # shape: [N, 1, T, 2]
            xs_map = xs_map.view(x_i.shape[0], x_i.shape[1], 1) * 2. - 1
            grid = torch.stack((xs_map, torch.full_like(xs_map, -1.0)), dim=-1)
            msg_advected_interp = torch.nn.functional.grid_sample(
                x_interp, grid, mode=self.advection_op.interpolation_mode,
                align_corners=True, 
            ).squeeze(3).transpose(2,1) * weight
            msg_advected *= torch.isnan(xs_map).float()
            msg_advected += torch.nan_to_num(msg_advected_interp)
            
        return msg_advected

    def forward(self, X, edge_index, flows, warp_map, sl_mask, edge_mask, eps=1e-10, xs_map=None, **kwargs):
        '''
        Apply one step of advection message passing. Sends messages along edges 
        according to flow direction.  I.e.:
        (1) Computes mixing coefficients based on inflow. This is done if
        self.mixing_at_nodes is True. This should be True if concentrations
        are advected and False for masses.
        (2) Generates messages from neighboring nodes by (i) Shifing by delay
        steps and (ii) scaling by inflow/total_inflow
        (3) Aggregates messages by summing the scaled messages
        
        For efficiency, 
        we track the time passed per edge (s. self.aggregated_time) if an edge 
        has sent Tau many time steps of data, we can disable the edge to
        make subsequent comutations faster.

        Parameters
        ----------
        X : Tensor, float
            The concentrations of all nodes, shape [|N|, T] where |N| is the 
            number of nodes, and T is the number of historic tiumestpes (cT) + 
            the number of future timesteps (Tau).
        edge_index : Tensor, int
            Edge index of the network, shape [2, |E|], where |E| is the number 
            of edges.
        edge_mask : Tensor, bool
            The edges that are active for the current message-passing step.
            Determined by self.aggregated_time and Tau
        dt : float
            The length of each time step.
        cT : int
            Number of historic time steps. (T - cT = Tau)
        agg_time : Tuple(Tensor, Tensor)
            This info is None for the first step and populated by 
            aggregate_time. It is fed back at each subsequence call to propagate
        eps : float
            A small value that is added to divisions for numeric stability.

        Returns
        -------
        Tuple(Tensor[|N|, T], Tuple(Tensor, Tensor), Tensor[|E|, T])
            Updated node features; new agg_time as computed by aggregate_time;
            Messages sent in this message-passing step (for debugging purposes)
        '''
        snd, rec = edge_index
        sl_mask = sl_mask[edge_mask]
        
        # compute mixing coefficients
        if self.mixing_at_nodes:
            total_inflows = scatter_add(flows.relu(), rec, dim=0, dim_size=len(X))
            total_inflows = scatter_add((-flows).relu(), snd, dim=0, out=total_inflows)

            # inflow_scale1 = flows[edge_mask].relu() / (total_inflows[rec[edge_mask]] + eps)
            # inflow_scale2 = (-flows[edge_mask]).relu() / (total_inflows[snd[edge_mask]] + eps)

            rec_total = total_inflows[rec[edge_mask]]
            snd_total = total_inflows[snd[edge_mask]]
            mr = (rec_total == 0).float() * sl_mask * eps
            ms = (snd_total == 0).float() * sl_mask * eps

            inflow_scale1 = (flows[edge_mask].relu() + mr) / (rec_total + mr)
            inflow_scale2 = ((-flows[edge_mask]).relu() + ms) / (snd_total + ms)
        else:
            total_inflows = scatter_add((flows >= 0).float(), rec, dim=0, dim_size=len(X))
            total_inflows = scatter_add((flows < 0).float(), snd, dim=0, out=total_inflows)

            inflow_scale1 = (flows[edge_mask] >= 0).float() / (total_inflows[rec[edge_mask]] + eps)
            inflow_scale2 = (flows[edge_mask] < 0).float() / (total_inflows[snd[edge_mask]] + eps)
        
        edge_index = edge_index[:, edge_mask]
        snd, rec = edge_index
        
        # Fetch the delay steps for the active edges
        warp_map = warp_map[edge_mask]
        if xs_map is not None:
            xs_map = xs_map[edge_mask]

        # Compute how much time passes for each edge (for efficiency reasons)
        # Some edges may require selfloops. This happens when there is an inflow into a pipe,
        # followed by a reversal of flows. In this case the node has to acess its own history.
        sl_mask = sl_mask.unsqueeze(-1)

        warp_map = -warp_map.abs()

        bi_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        bi_weights = torch.cat([inflow_scale1, inflow_scale2], dim=0)
        bi_warp_map = torch.cat([warp_map, warp_map], dim=0)
        bi_sl_mask = torch.cat([sl_mask, sl_mask], dim=0)
        if xs_map is not None:
            xs_map = torch.cat([xs_map, xs_map], dim=0)

        # For additional inputs to the message function, apply edge mask and 
        # convert to bidirectional edge set for performance reasons
        _kwargs = {}
        for k, v in kwargs.items():
            if not isinstance(v, Iterable) or v.size(0) != edge_mask.size(0): 
                _kwargs[k] = v
                continue
            v = v[edge_mask]
            _kwargs[k] = torch.cat([v, v], dim=0)


        agg = self.propagate(bi_edge_index, x=X, warp_map=bi_warp_map, xs_map=xs_map, weight=bi_weights, sl_mask=bi_sl_mask, **_kwargs)

        return agg
    

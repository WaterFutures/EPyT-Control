import torch
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from collections.abc import Iterable
from torch_geometric.utils import degree
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

        msg = torch.where(sl_mask, adv_x_i, adv_x_j)

        if xs_map is not None:
            x_interp = torch.stack((x_i, x_j), -1).transpose(2,1) # shape: [N, 1, T, 2]
            xs_map = xs_map.abs().view(x_i.shape[0], x_i.shape[1], 1) * 2. - 1
            grid = torch.stack((xs_map, torch.full_like(xs_map, -1.0)), dim=-1)
            msg_interp = torch.nn.functional.grid_sample(
                x_interp, grid, mode=self.advection_op.interpolation_mode,
                align_corners=True, 
            ).squeeze(3).transpose(2,1)
            msg = torch.where(torch.isnan(xs_map), msg, torch.nan_to_num(msg_interp))
            
        return msg * weight

    def _compute_weights(self, n_nodes, edge_index, flows, sl_mask, degrees, eps, edge_ids=None):
        total_inflows = flows.new_zeros((n_nodes,) + flows.shape[1:])
        snd, rec = edge_index
        sl_mask = sl_mask.bool()

        if edge_ids is not None:
            snd_a, rec_a = edge_index[:, edge_ids]
            flows_a = flows[edge_ids]
            sl_a = sl_mask[edge_ids]
        else:
            snd_a, rec_a = snd, rec
            flows_a = flows
            sl_a = sl_mask

        if self.mixing_at_nodes:
            total_inflows.index_add_(0, rec, flows.clamp_min(0))
            total_inflows.index_add_(0, snd, (-flows).clamp_min(0))

            rec_total = total_inflows[rec_a]
            snd_total = total_inflows[snd_a]

            w_rec = flows_a.clamp_min(0) / (rec_total + eps)
            w_snd = (-flows_a).clamp_min(0) / (snd_total + eps)

            d_rec = degrees[rec_a].unsqueeze(-1)
            d_snd = degrees[snd_a].unsqueeze(-1)

            w_rec = torch.where((rec_total.abs() <= eps) & sl_a, d_rec.reciprocal(), w_rec)
            w_snd = torch.where((snd_total.abs() <= eps) & sl_a, d_snd.reciprocal(), w_snd)

        else:
            total_inflows.index_add_(0, rec, (flows >= 0).to(flows.dtype))
            total_inflows.index_add_(0, snd, (flows < 0).to(flows.dtype))

            rec_total = total_inflows.index_select(0, rec_a)
            snd_total = total_inflows.index_select(0, snd_a)

            w_rec = (flows_a >= 0).to(flows.dtype) / (rec_total + eps)
            w_snd = (flows_a < 0).to(flows.dtype) / (snd_total + eps)

        return w_rec, w_snd

    def forward(self, X, edge_index, warp_map, sl_mask, edge_mask, weights, eps=1e-10, xs_map=None,
                time_valid=None, **kwargs):
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
        edge_ids = edge_mask.nonzero(as_tuple=True)[0]
        sl_mask = sl_mask[edge_ids]

        warp_map = warp_map[edge_ids]
        
        # compute mixing coefficients
        weights = weights[edge_ids]

        edge_index = edge_index[:, edge_ids]
        
        # Fetch the delay steps for the active edges
        if xs_map is not None:
            xs_map = xs_map[edge_ids,:warp_map.shape[1]]

        # Compute how much time passes for each edge (for efficiency reasons)
        # Some edges may require selfloops. This happens when there is an inflow into a pipe,
        # followed by a reversal of flows. In this case the node has to acess its own history.
        sl_mask = sl_mask.unsqueeze(-1)
        warp_map = -warp_map.abs()

        # For additional inputs to the message function, apply edge mask and 
        # convert to bidirectional edge set for performance reasons
        #_kwargs = self._prepare_kwargs(kwargs, edge_ids, flows.shape[0])
        _kwargs = {k: v[edge_ids].clone() for k, v in kwargs.items() if isinstance(v, torch.Tensor)
                   and v.shape[0] == edge_mask.shape[0]}
        
        agg = self.propagate(edge_index, x=X, warp_map=warp_map, xs_map=xs_map, weight=weights,
                             sl_mask=sl_mask, **_kwargs)


        return agg

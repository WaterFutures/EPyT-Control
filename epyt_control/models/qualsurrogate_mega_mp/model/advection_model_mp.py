import numpy as np
import torch
from torch_geometric.nn import MessagePassing
from tqdm.auto import tqdm

from ..modules.torch_advection import AdvectionModuleGridSampleDynamic


class AdvectionLayer(MessagePassing):
    
    def __init__(self, advection_op, **kwargs):
        super().__init__(aggr='sum', node_dim=0, **kwargs)
        self.advection_op = advection_op

    def message(self, x_i, x_j, warp_map, weight, sl_mask):
        adv_x_i = self.advection_op(x_i, warp_map).squeeze(-1)
        adv_x_j = self.advection_op(x_j, warp_map).squeeze(-1)

        self.msg = torch.where(sl_mask, adv_x_i, adv_x_j) * weight
            
        return self.msg
    
    def forward(self, X, edge_index, warp_map, weights, sl_mask, edge_mask):
        edge_ids = edge_mask.nonzero(as_tuple=True)[0]
        sl_mask = sl_mask[edge_ids].bool()

        warp_map = -warp_map[edge_ids].abs()
        edge_index = edge_index[:, edge_ids]
        weights = weights[edge_ids]
        
        agg = self.propagate(edge_index, x=X.unsqueeze(-1), warp_map=warp_map, weight=weights, sl_mask=sl_mask)

        return agg


class AdvectionModelMP(torch.nn.Module):

    def __init__(
            self, mp_layer, max_msg_passing_rounds=25, constant_history=False, 
            progress=True, adaptive_steps=False, track_active_edges=True, **kwargs
        ):
        super().__init__(**kwargs)
        # The maximum number of message passing rounds. More message-passing
        # rounds are required in areas of low flow to achieve high accuracy.
        self.time_warp = AdvectionModuleGridSampleDynamic(interpolation_mode='nearest')
        self.solved_prop = AdvectionLayer(AdvectionModuleGridSampleDynamic(interpolation_mode='bilinear'))
        self.max_msg_passing_rounds = max_msg_passing_rounds
        self.advection_mpnn = mp_layer
        self.constant_history = constant_history
        self.progress = progress
        self.adaptive_steps = adaptive_steps
        self.track_active_edges = track_active_edges
        self.debug_info = {}

    def track_solved_edges(self, warp_map, cT, edge_mask, edge_index, time_unsolved, sl_mask,
                           weights):
        time_unsolved = self.solved_prop(time_unsolved, edge_index, -warp_map.abs(), weights,
                                         sl_mask, edge_mask)
        time_unsolved[:,:cT] = 0.
        
        edge_unresolved = self.solved_prop.msg
        
        edge_ids = edge_mask.nonzero(as_tuple=True)[0]
        edge_mask.index_copy_(0, edge_ids, edge_mask[edge_ids] & (edge_unresolved[:, cT:].amax(dim=1) > 0.001))
        
        return time_unsolved, edge_unresolved, edge_mask

    def create_initial_edge_mask(self, edge_index):
        return torch.ones(edge_index.shape[1], dtype=bool, device=edge_index.device)
        
    def set_flow_field(self, flows, delay_steps):
        self.flows = flows
        self.delay_steps = -delay_steps
    
    def _prepare_kwargs(self, kwargs, n_edges):
        if not kwargs:
            return kwargs

        result = {}

        for name, value in kwargs.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == n_edges:
                value = torch.cat((value, value), dim=0)

            result[name] = value

        return result
    
    def forward(
            self, x, edge_index, flows, delay_steps, sl_mask, Tau, n_steps=1, boundary_values=None,
            boundary_index=None, **kwargs):
        n_nodes, cT = x.shape[:2]
        edge_passes = 0
        edges_active = []
        agg_times = []
        aggs_all = []
        sl_mask = sl_mask[:,:cT+Tau].bool()

        steps_to_hist = torch.nn.functional.pad(torch.arange(1, Tau + 1, device=x.device), (cT, 0)).unsqueeze(0)

        bc_mask = x.new_zeros(n_nodes)
        if boundary_index is not None:
            bc_mask[boundary_index] = 1.
        
        flows = flows[:,:cT+Tau]
        time_valid = torch.zeros_like(flows)
        degrees = torch.bincount(edge_index.reshape(-1), minlength=n_nodes).to(flows.dtype)

        agg1 = torch.nn.functional.pad(x.clone(), (0,0,0,Tau,0,0))
        aggs = agg1.clone()
        
        agg_time = None
        time_unsolved = torch.zeros_like(agg1[:,:,0])
        time_unsolved[:,:cT] = 0.
        time_unsolved[:,cT:] = 1.
        warp_map = -delay_steps[:,:cT+Tau]

        if boundary_values is not None:
            assert boundary_index is not None
            agg1[boundary_index] = boundary_values[:,:cT+Tau]
            time_unsolved[boundary_index] = 0.
        
        aggs_all = [agg1.detach()]
        
        if self.progress:
            progress = tqdm(total=100)

        self.weights = self.advection_mpnn._compute_weights(
            n_nodes, edge_index, flows, sl_mask, degrees, eps=1e-10
        )
        bi_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        bi_weights = torch.cat(self.weights, dim=0)
        bi_warp_map = warp_map.repeat(2, 1)
        bi_sl_mask = sl_mask.repeat(2, 1)
        kwargs = self._prepare_kwargs(kwargs, flows.shape[0])
        time_active = self.create_initial_edge_mask(bi_edge_index)
        time_active_single = self.create_initial_edge_mask(edge_index)
        
        
        for i in range(self.max_msg_passing_rounds):
            agg1 = self.advection_mpnn(agg1, bi_edge_index, bi_warp_map, bi_sl_mask, time_active,
                                       bi_weights, time_valid=time_valid, degrees=degrees, **kwargs)
            
            time_unsolved, edge_unresolved, time_active = self.track_solved_edges(bi_warp_map, cT,
                                                                                  time_active,
                                                                                  bi_edge_index,
                                                                                  time_unsolved,
                                                                                  bi_sl_mask,
                                                                                  bi_weights)
                
            # Set zeros at injection node, this is a dirichlet boundary condition and we now the function value
            # Note: This is not universally true, if we inject mass then the masses should mix at 
            # the boundary condition too (TODO: Make a parameter for this)
            if boundary_index is not None:
                agg1[boundary_index] *= 0
                
            agg1[:, :cT].zero_()
            torch.nan_to_num_(agg1) # Prevent explosions in early training

            if self.progress:
                progress.n = np.round((1-time_active.float().mean().item()) * 100., decimals=2)
                progress.refresh()

            aggs += agg1

            if not time_active.any():
                if self.progress:
                    progress.set_description(f'Total iterations: {i}')
                break
        
        if self.progress:
            progress.close()
        
        if boundary_values is not None:
            aggs[boundary_index] = boundary_values[:,:aggs.shape[1]]
        print(i)
        return aggs, edge_passes, edges_active, agg_times, aggs_all

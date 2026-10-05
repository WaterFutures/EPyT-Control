import numpy as np
import torch
from torch_scatter import scatter_min
from tqdm.auto import tqdm

from ..modules.torch_advection import AdvectionModuleGridSampleDynamic


class AdvectionModelMP(torch.nn.Module):

    def __init__(
            self, mp_layer, max_msg_passing_rounds=25, constant_history=False, 
            progress=True, adaptive_steps=False, track_active_edges=True, **kwargs
        ):
        super().__init__(**kwargs)
        # The maximum number of message passing rounds. More message-passing
        # rounds are required in areas of low flow to achieve high accuracy.
        self.max_msg_passing_rounds = max_msg_passing_rounds
        self.advection_mpnn = mp_layer
        self.constant_history = constant_history
        self.progress = progress
        self.adaptive_steps = adaptive_steps
        self.track_active_edges = track_active_edges
        self.debug_info = {}    

    @torch.no_grad()
    def aggregated_time(self, n_nodes, Tau, edge_mask, edge_index, delay_steps, agg_time=None, sl_mask=None):
        snd, rec = edge_index[:, edge_mask]
        delay_steps = delay_steps[edge_mask]
        n_edges = len(edge_mask)
        n_time = delay_steps.shape[1]
        time_warp = AdvectionModuleGridSampleDynamic(interpolation_mode='bilinear')
        
        if agg_time is None:
            edge_age = torch.zeros(n_edges, n_time, device=delay_steps.device)
            edge_age[edge_mask] = delay_steps.abs()
            return (edge_age, edge_age)
            
        cum_edge_age, edge_age = agg_time
        time = -delay_steps

        node_age_new = torch.zeros(n_nodes, n_time, device=delay_steps.device) + 1e10
        edge_age_masked = edge_age[edge_mask]
        node_age_new, _ = scatter_min(edge_age_masked + 1e10 * (time < 0).float(), rec, 0, out=node_age_new)
        node_age_new, _ = scatter_min(edge_age_masked + 1e10 * (time >= 0).float(), snd, 0, out=node_age_new)

        snd_node_age = torch.where(time >= 0, node_age_new[snd], node_age_new[rec])
        sl = sl_mask[edge_mask].bool()
        snd_node_age[sl] = torch.where(time >= 0, node_age_new[rec], node_age_new[snd])[sl]
        edge_age[edge_mask] = time_warp(snd_node_age[...,None], -time.abs())[...,0]

        return (cum_edge_age + edge_age, edge_age)
    
    def create_initial_edge_mask(self, edge_index):
        return torch.ones(edge_index.shape[1], dtype=bool, device=edge_index.device)
        
    def set_flow_field(self, flows, delay_steps):
        self.flows = flows
        self.delay_steps = -delay_steps

    def set_step_flow_params(self, step):
        self._step_flows = self.flows[:,:step]
        self._step_delay_steps = self.delay_steps[:,:step]
        self._step_sl_mask = self.sl_mask[:,:step]
        self._step_xs_map = None
        if self.xs_map is not None:
            self._step_xs_map = self.xs_map[:,:step]
    
    def set_sl_mask(self, sl_mask):
        self.sl_mask = sl_mask
    
    def forward(
            self, x, edge_index, flows, delay_steps, sl_mask, Tau, n_steps=1, boundary_values=None, boundary_index=None, **kwargs
        ):
        xT = x.clone() # save all results
        average_steps = 0
        n_nodes, cT = x.shape[:2]
        edge_passes = 0
        edges_active = []
        agg_times = []
        aggs_all = []
        self.node_time_prev = None

        self.flows = flows
        self.delay_steps = -delay_steps
        self.sl_mask = sl_mask
        self.xs_map = kwargs.pop('xs_map')

        step = 0
        output_steps = cT + Tau * n_steps
        
        while x.shape[1] < output_steps:
        # for step in range(n_steps):
            if self.adaptive_steps:
                Tau = self.delay_steps.abs().min().int()
            self.set_step_flow_params(cT + Tau)
            time_active = self.create_initial_edge_mask(edge_index)
            
            agg_time, masking = None, None
            
            agg1 = torch.nn.functional.pad(x.clone(), (0,0,0,Tau,0,0))
            aggs = agg1.clone()

            if boundary_values is not None:
                assert boundary_index is not None
                start_idx = 0 if not self.constant_history else step*Tau
                agg1[boundary_index] = boundary_values[:,start_idx:agg1.shape[1]]
                
            total_ts = cT + Tau
            ct_mask = torch.ones(total_ts, 1, device=x.device)
            ct_mask[:cT].zero_()
            
            aggs_all = [agg1.detach()]
            
            if self.progress:
                progress = tqdm(total=100)

            for i in range(self.max_msg_passing_rounds):

                if not time_active.any():
                    if self.progress:
                        progress.set_description(f'Total iterations: {i}')
                    break

                flows = self._step_flows
                warp_map = self._step_delay_steps
                sl_mask = self._step_sl_mask
                xs_map = self._step_xs_map
                
                agg1 = self.advection_mpnn(agg1, edge_index, flows, warp_map, sl_mask, time_active, xs_map=xs_map, **kwargs)
                # Compute how much time passes for each edge (for efficiency reasons)
                if self.track_active_edges:
                    agg_time = self.aggregated_time(n_nodes, Tau, time_active, edge_index, warp_map, agg_time, sl_mask)
                    time_active = torch.logical_and(time_active, (agg_time[0].abs() <= (Tau+1)).any(1))

                # Set zeros at injection node, this is a dirichlet boundary condition and we now the function value
                # Note: This is not universally true, if we inject mass then the masses should mix at 
                # the boundary condition too (TODO: Make a parameter for this)
                if boundary_index is not None:
                    agg1[boundary_index] *= 0
                    
                agg1 = agg1 * ct_mask

                if self.progress:
                    progress.n = np.round((agg_time[0].abs() >= Tau).float().mean().item() * 100., decimals=2)
                    progress.refresh()

                agg1 = torch.nan_to_num(agg1) # Prevent explosions in early training
                aggs += agg1
            
            if self.progress:
                progress.close()   
            average_steps += i
            x_new = aggs
            xT = x_new
            
            if boundary_values is not None:
               xT[boundary_index] = boundary_values[:,start_idx:xT.shape[1]]
            if self.constant_history:
                x = xT[:,(step+1)*Tau:]
            else:
                x = xT
                cT = cT + Tau
            step += 1
            
        return xT, edge_passes, edges_active, agg_times, aggs_all

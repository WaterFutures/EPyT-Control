import torch
from .advection_layer import AdvectionLayer


class AdvectionReactionLayer(AdvectionLayer):
    
    def __init__(self, advection_op, mixing_at_nodes=True, **kwargs):
        super().__init__(advection_op, mixing_at_nodes, **kwargs)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 8, bias=True),
            torch.nn.SELU(),
            torch.nn.Linear(8, 1, bias=True)
        )
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=0.001)

    def set_hydraulis_parameters(self, diameters, lengths):
        diameter = torch.as_tensor(diameters, dtype=torch.get_default_dtype()).unsqueeze(1)
        lengths = torch.as_tensor(lengths, dtype=torch.get_default_dtype()).unsqueeze(1)
        if not hasattr(self, 'diameter'):
            self.register_buffer('diameter', diameter) # register for .to(device)
        if not hasattr(self, 'lengths'):
            self.register_buffer('lengths', lengths)   # register for .to(device)
        self.diameter = diameter
        self.lengths = lengths

    def message(self, x_i, x_j, warp_map, weight, sl_mask, edge_diameter_scaled):
        N, T = x_j.shape[:2]
        # Message passing: sender -> receiver (j -> i)
        msg_advected = self.advection_op(x_j, warp_map) * (1.-sl_mask)
        # Generate messages for selfloops (i -> i)
        msg_advected += self.advection_op(x_i, warp_map) * sl_mask
        
        react_inputs = 1 / edge_diameter_scaled.view(-1, 1, 1).expand(N, T, 1)
        react_residual = self.mlp(react_inputs) # edge_diameter_scaled = 1 / self.diameter[edge_mask].expand(N, T)
        msg_advected = msg_advected * (react_residual * warp_map.unsqueeze(-1).abs()).exp()
        
        return msg_advected * weight.unsqueeze(-1)


class AdvectionReactionLayerFullyLearned(AdvectionLayer):
    
    def __init__(self, hidden_dim, flow_attribute, mixing_at_nodes=True, **kwargs):
        super().__init__(None, mixing_at_nodes, **kwargs)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 3, hidden_dim, bias=True),
            torch.nn.SELU(),
            torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        )

        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight, gain=0.001)
                
        self.flow_attribute = flow_attribute

    def set_hydraulis_parameters(self, diameters, lengths):
        diameter = torch.as_tensor(diameters, dtype=torch.get_default_dtype()).unsqueeze(1)
        lengths = torch.as_tensor(lengths, dtype=torch.get_default_dtype()).unsqueeze(1)
        if not hasattr(self, 'diameter'):
            self.register_buffer('diameter', diameter) # register for .to(device)
        if not hasattr(self, 'lengths'):
            self.register_buffer('lengths', lengths)   # register for .to(device)
        self.diameter = diameter
        self.lengths = lengths

    def message(self, x_i, x_j, warp_map, weight, sl_mask, delay_steps_scaled, flows_scaled):
        N, T = x_j.shape[:2]
        if self.flow_attribute == 'delays':
            msg_input = torch.cat([x_i[...,0], x_j[...,0], delay_steps_scaled[:,:x_j.size(1)]], dim=-1)
        elif self.flow_attribute == 'flows':
            msg_input = torch.cat([x_i[...,0], x_j[...,0], flows_scaled[:,:x_j.size(1)]], dim=-1)
        # Message passing: sender -> receiver (j -> i)
        msg = self.mlp(msg_input) * weight
        
        return msg.unsqueeze(-1)

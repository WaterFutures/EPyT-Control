import torch

class AdvectionModule(torch.nn.Module):
    
    def __init__(self, antialiasing=1, interpolation_mode='nearest', **kwargs):
        super().__init__(**kwargs)
        self.interpolation_mode = interpolation_mode
        self.antialiasing = antialiasing
    
    def lift(self, x, smoothing=1.):
        return x
    def unlift(self, x, smoothing=1.):
        return x
    
    def forward(self, x, flow):
        raise NotImplementedError('Subclass this.')
    
class AdvectionModuleGridSampleDynamic(AdvectionModule):

    def forward(self, x, flow):
        B, L = x.shape[:2]
        
        # Gridsample only works for rank 4 or rank 5 tensors,
        # expand to shape: [B, 1, 1, L]
        x = x.transpose(2, 1).unsqueeze(2)  

        # Create the grid: [1, 1, L, 1]
        grid = torch.linspace(-1, 1, L, device=x.device).view(1, 1, L, 1)
        grid = grid.expand(B, 1, L, 1)
        
        # Normalize the shift
        flow_normalized = (flow).view(B, 1, L, 1) / (L / 2)
        shifted_grid = grid + flow_normalized
        
        # Add the second dimension for compatibility
        shifted_grid = torch.cat([shifted_grid, torch.zeros_like(shifted_grid)], dim=-1)
        #shifted_grid = shifted_grid.clamp(-1, 1)
        # shape: [B, 1, 1, L]
        shifted_tensor = torch.nn.functional.grid_sample(
            x, shifted_grid, mode=self.interpolation_mode, padding_mode="border", 
            align_corners=True
        )

        return shifted_tensor.squeeze(2).transpose(2, 1)


class AdvectionModuleGridSample(AdvectionModule):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x, flow):
        B, L = x.shape[:2]

        # Gridsample only works for rank 4 or rank 5 tensors,
        # expand to shape: [B, 1, 1, L]
        x = x.unsqueeze(1).unsqueeze(2)  

        # Normalize the shift
        grid = torch.linspace(-1, 1, L, device=x.device).view(1, 1, L, 1)
        grid = grid.expand(B, 1, L, 1)  # Shape: (B, 1, L, 1)

        # Add the shift (normalized to [-1, 1])
        flow_normalized = (flow).view(B, 1, 1, 1) / (L / 2)
        shifted_grid = grid + flow_normalized 

        # Add the second dimension for compatibility
        shifted_grid = torch.cat([shifted_grid, torch.zeros_like(shifted_grid)], dim=-1)

        # shape: [B, 1, 1, L]
        shifted_tensor = torch.nn.functional.grid_sample(
            x, shifted_grid, mode="nearest", padding_mode="zeros", align_corners=True
        )

        return shifted_tensor.squeeze(1).squeeze(1)

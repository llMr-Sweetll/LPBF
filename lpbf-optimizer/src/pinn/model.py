import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for predicting LPBF process outcomes
    (residual stress, porosity, geometric accuracy) from process parameters.
    
    Args:
        in_dim (int): Input dimension (number of process parameters)
        out_dim (int): Output dimension (number of predicted outcomes)
        width (int): Width of hidden layers
        depth (int): Number of hidden layers
    """
    def __init__(self, in_dim=6, out_dim=3, width=512, depth=5):
        super().__init__()
        # Process parameters typically include:
        # - Laser power (P)
        # - Scan speed (v)
        # - Hatch spacing (h)
        # - Scan angle (θ)
        # - Island length (l_island)
        # - Layer thickness
        
        # Build the network with SiLU activation (smooth, differentiable)
        layers = []
        layers.append(nn.Linear(in_dim, width))
        for _ in range(depth-1):
            layers += [nn.SiLU(), nn.Linear(width, width)]
        
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(width, out_dim)
    
    def forward(self, S):
        """
        Forward pass through the network
        
        Args:
            S (torch.Tensor): Process parameter tensor [batch_size, in_dim]
        
        Returns:
            torch.Tensor: Predicted outcomes [batch_size, out_dim]
                - Residual stress (σ_res)
                - Porosity (φ_pore)
                - Geometric accuracy ratio (GAR)
        """
        return self.out(self.hidden(S))
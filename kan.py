import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KANLinear(nn.Module):
    """
    Vectorized implementation of a Kolmogorov-Arnold Network (KAN) layer.
    Maps in_features to out_features using a combination of a base activation 
    (SiLU) and a learnable B-spline function for each input-output pair.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: tuple = (-1.0, 1.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Scaling coefficients
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        
        # Base weight (like standard linear layer weight)
        self.weight_base = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Spline coefficients weight. Size: (out_features, in_features, grid_size + spline_order)
        self.weight_spline = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        
        # Grid construction (static uniform grid, can be refined dynamically if needed)
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        )
        self.register_buffer("grid", grid)
        
        self.scale_noise = scale_noise
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize base weight using Xavier uniform
        nn.init.kaiming_uniform_(self.weight_base, a=math.sqrt(5))
        
        # Initialize spline coefficients weight with a small amount of noise
        nn.init.normal_(
            self.weight_spline,
            mean=0.0,
            std=self.scale_noise / math.sqrt(self.in_features + self.grid_size),
        )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate B-splines of given order for the input tensor x.
        Input x shape: (batch_size, in_features)
        Output shape: (batch_size, in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        
        # Add dimensions for broadcasting: (batch_size, in_features, 1)
        x = x.unsqueeze(-1)
        grid = self.grid  # Shape: (grid_size + 2 * spline_order + 1)
        
        # Order 0 basis functions
        # B_i,0(x) = 1 if grid[i] <= x < grid[i+1], else 0
        bases = ((x >= grid[:-1]) & (x < grid[1:])).to(x.dtype)
        
        # Cox-de Boor recursion formula for higher orders
        # B_i,k(x) = (x - t_i)/(t_{i+k} - t_i) * B_i,k-1(x) + (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[: -(k + 1)])
                / (grid[k:-1] - grid[: -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[k + 1 :] - x)
                / (grid[k + 1 :] - grid[1:-k])
                * bases[:, :, 1:]
            )
            
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the KAN Linear layer.
        x: (batch_size, in_features)
        returns: (batch_size, out_features)
        """
        # 1. Base activation part: SiLU(x) * weight_base
        base_act = F.silu(x)
        y_base = F.linear(base_act, self.weight_base) * self.scale_base
        
        # 2. Spline part
        # splines shape: (batch_size, in_features, grid_size + spline_order)
        splines = self.b_splines(x)
        
        # weight_spline shape: (out_features, in_features, grid_size + spline_order)
        # We perform Einstein summation to multiply splines and coefficients and sum over input features and spline bases
        y_spline = torch.einsum("bif,oif->bo", splines, self.weight_spline) * self.scale_spline
        
        return y_base + y_spline


class KAN(nn.Module):
    """
    Multi-layer Kolmogorov-Arnold Network (KAN).
    """
    def __init__(
        self,
        layers_hidden: list,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: tuple = (-1.0, 1.0),
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden[:-1], layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features=in_features,
                    out_features=out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

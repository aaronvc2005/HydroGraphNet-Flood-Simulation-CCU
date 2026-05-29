import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    """
    Formulates physics-informed constraints based on Shallow Water Equations (SWE).
    Uses a highly vectorized finite-volume mass conservation formulation over
    unstructured 2D cell meshes.
    """
    def __init__(self, lambda_mass: float = 0.1, lambda_bound: float = 0.05):
        super().__init__()
        self.lambda_mass = lambda_mass
        self.lambda_bound = lambda_bound
        self.mse = nn.MSELoss()

    def compute_mass_residuals(
        self,
        h_t: torch.Tensor,       # Depth at time t: [num_cells, 1]
        h_next: torch.Tensor,    # Depth at time t+1: [num_cells, 1]
        u_t: torch.Tensor,       # Velocity X at time t: [num_cells, 1]
        v_t: torch.Tensor,       # Velocity Y at time t: [num_cells, 1]
        face_cells: torch.Tensor,   # Cell pairs for each face: [num_faces, 2]
        face_widths: torch.Tensor,  # Face widths: [num_faces]
        face_normals: torch.Tensor, # Normals: [num_faces, 2]
        cell_areas: torch.Tensor,   # Cell areas: [num_cells]
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Computes the mass conservation residuals for each cell on the unstructured mesh.
        """
        num_cells = h_t.size(0)
        
        c1 = face_cells[:, 0]
        c2 = face_cells[:, 1]
        
        # Flatten tensors for calculation
        h_t_flat = h_t.squeeze(-1)
        u_t_flat = u_t.squeeze(-1)
        v_t_flat = v_t.squeeze(-1)
        
        # Reconstruct state at senders (c1)
        h1 = h_t_flat[c1]
        u1 = u_t_flat[c1]
        v1 = v_t_flat[c1]
        
        # Reconstruct state at receivers (c2). Handle boundary (-1) reflective conditions.
        c2_clamped = torch.clamp(c2, min=0)
        c2_mask = (c2 != -1).float()
        
        h2 = c2_mask * h_t_flat[c2_clamped] + (1.0 - c2_mask) * h1
        u2 = c2_mask * u_t_flat[c2_clamped] + (1.0 - c2_mask) * (-u1) # reflecting velocity
        v2 = c2_mask * v_t_flat[c2_clamped] + (1.0 - c2_mask) * (-v1)
        
        # Interpolate states at cell face boundaries
        h_face = 0.5 * (h1 + h2)
        u_face = 0.5 * (u1 + u2)
        v_face = 0.5 * (v1 + v2)
        
        # Normal velocity across the face
        v_n = u_face * face_normals[:, 0] + v_face * face_normals[:, 1]
        
        # Calculate volumetric discharge (m^2/s)
        Q = h_face * v_n * face_widths
        
        # Accumulate net flux for all cells
        net_flux = torch.zeros(num_cells, device=h_t.device)
        net_flux.index_add_(0, c1, -Q)
        
        # Add positive flux to receiver cell only if it's an interior cell (c2 != -1)
        interior_indices = c2[c2 != -1]
        net_flux.index_add_(0, interior_indices, Q[c2 != -1])
        
        # Rate of change of depth: dh/dt
        dh_dt = (h_next.squeeze(-1) - h_t_flat) / dt
        
        # Residual of discrete mass conservation equation
        # dh/dt + NetFlux / Area = 0
        residual = dh_dt - (net_flux / cell_areas)
        
        return residual

    def forward(
        self,
        pred_state: torch.Tensor,    # Predicted [h, u, v] at t+1: [num_cells, 3]
        true_state: torch.Tensor,    # True [h, u, v] at t+1: [num_cells, 3]
        prev_state: torch.Tensor,    # True [h, u, v] at t: [num_cells, 3]
        face_cells: torch.Tensor,
        face_widths: torch.Tensor,
        face_normals: torch.Tensor,
        cell_areas: torch.Tensor,
        dt: float = 1.0,
    ) -> tuple:
        """
        Calculates composite loss: data supervised loss + SWE physical mass loss + depth boundary loss.
        """
        # 1. Supervised Data Loss (HEC-RAS matching)
        loss_data = self.mse(pred_state, true_state)
        
        # Extract individual states
        h_prev = prev_state[:, 0:1]
        u_prev = prev_state[:, 1:2]
        v_prev = prev_state[:, 2:3]
        
        h_pred = pred_state[:, 0:1]
        
        # 2. Physics-Informed Mass Conservation Residual
        mass_residuals = self.compute_mass_residuals(
            h_t=h_prev,
            h_next=h_pred,
            u_t=u_prev,
            v_t=v_prev,
            face_cells=face_cells,
            face_widths=face_widths,
            face_normals=face_normals,
            cell_areas=cell_areas,
            dt=dt
        )
        loss_mass = torch.mean(mass_residuals ** 2)
        
        # 3. Boundary Hinge Penalty (Depth cannot be negative)
        loss_bound = torch.mean(torch.clamp(-h_pred, min=0.0) ** 2)
        
        # Composite Loss
        total_loss = loss_data + self.lambda_mass * loss_mass + self.lambda_bound * loss_bound
        
        return total_loss, loss_data, loss_mass, loss_bound

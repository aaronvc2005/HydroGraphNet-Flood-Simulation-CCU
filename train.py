import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

def train_hydrographnet(
    model,
    tensors,
    pinn_loss_fn,
    epochs: int = 150,
    lr: float = 0.005,
    pushforward_steps: int = 3, # K-steps rollout for stability
    dt: float = 1.0,
    device: str = "cpu"
) -> dict:
    """
    Main training function for HydroGraphNet using Physics-Informed losses
    and the Pushforward Trick for autoregressive stability.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    # Extract mesh static variables
    cell_centers = tensors["cell_centers"]
    cell_elevations = tensors["cell_elevations"]
    cell_areas = tensors["cell_areas"]
    face_cells = tensors["face_cells"]
    face_widths = tensors["face_widths"]
    face_normals = tensors["face_normals"]
    
    # Extract dynamic simulation data
    depths = tensors["depths"]           # Shape: [time_steps, num_cells]
    velocities = tensors["velocities"]   # Shape: [time_steps, num_cells, 2]
    
    time_steps = depths.size(0)
    num_cells = depths.size(1)
    
    # Build static node features: [elevation, cell_centers_x, cell_centers_y]
    static_features = torch.cat([
        cell_elevations.unsqueeze(-1),
        cell_centers
    ], dim=-1) # Shape: [num_cells, 3]
    
    # Build edge structures
    edge_index, edge_attr = model.prepare_edges(face_cells, face_widths, face_normals)
    
    history_losses = {"total": [], "data": [], "mass": [], "bound": []}
    
    print(f"--- Starting HydroGraphNet PINN Training ---")
    print(f"Number of nodes (cells): {num_cells}")
    print(f"Number of edges (face connections): {edge_index.size(1)}")
    print(f"Training for {epochs} epochs using Pushforward Rollout={pushforward_steps} steps...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        epoch_total_loss = 0.0
        epoch_data_loss = 0.0
        epoch_mass_loss = 0.0
        epoch_bound_loss = 0.0
        
        # We sample random start times for autoregressive rollouts to cover the timeseries
        # Ensure we have enough remaining steps to perform rollout
        max_start_time = time_steps - pushforward_steps - 1
        
        # Train over the entire sequence by chunking into rollout segments
        for start_t in range(0, max_start_time, pushforward_steps):
            # Get initial state: [h, u, v] at start_t
            state = torch.cat([
                depths[start_t].unsqueeze(-1),
                velocities[start_t]
            ], dim=-1) # Shape: [num_cells, 3]
            
            # Perform rollout
            accumulated_loss = 0.0
            
            for step in range(pushforward_steps):
                t_curr = start_t + step
                t_next = t_curr + 1
                
                # Combine static node features with current dynamic state
                node_features = torch.cat([static_features, state], dim=-1)
                
                # Forward pass: predict rate of change
                delta = model(node_features, edge_index, edge_attr)
                
                # Compute predicted state at next step
                h_pred = torch.clamp(state[:, 0:1] + delta[:, 0:1], min=0.0)
                u_pred = state[:, 1:2] + delta[:, 1:2]
                v_pred = state[:, 2:3] + delta[:, 2:3]
                pred_state = torch.cat([h_pred, u_pred, v_pred], dim=-1)
                
                # Target ground-truth state at next step
                true_state = torch.cat([
                    depths[t_next].unsqueeze(-1),
                    velocities[t_next]
                ], dim=-1)
                
                # Compute composite loss (data + physics + boundary)
                total, data, mass, bound = pinn_loss_fn(
                    pred_state=pred_state,
                    true_state=true_state,
                    prev_state=state,
                    face_cells=face_cells,
                    face_widths=face_widths,
                    face_normals=face_normals,
                    cell_areas=cell_areas,
                    dt=dt
                )
                
                accumulated_loss += total
                
                epoch_data_loss += data.item()
                epoch_mass_loss += mass.item()
                epoch_bound_loss += bound.item()
                
                # Update current state for the next step of the pushforward trick (autoregressive feedback)
                # To prevent gradient explosion we detach or keep gradients.
                # In standard pushforward, we keep gradients to backpropagate through time (BPTT).
                state = pred_state
                
            # Average loss over the rollout length
            accumulated_loss = accumulated_loss / pushforward_steps
            accumulated_loss.backward()
            
            epoch_total_loss += accumulated_loss.item()

        # Step optimizer
        optimizer.step()
        
        # Scale losses for logging
        num_batches = len(range(0, max_start_time, pushforward_steps))
        avg_total = epoch_total_loss / num_batches
        avg_data = epoch_data_loss / (num_batches * pushforward_steps)
        avg_mass = epoch_mass_loss / (num_batches * pushforward_steps)
        avg_bound = epoch_bound_loss / (num_batches * pushforward_steps)
        
        scheduler.step(avg_total)
        
        history_losses["total"].append(avg_total)
        history_losses["data"].append(avg_data)
        history_losses["mass"].append(avg_mass)
        history_losses["bound"].append(avg_bound)
        
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d}/{epochs:03d} | "
                f"Total Loss: {avg_total:.5f} | "
                f"Data (MSE): {avg_data:.5f} | "
                f"SWE Mass Res: {avg_mass:.5f} | "
                f"Bound Hinge: {avg_bound:.5f}"
            )
            
    print(f"--- Training Finished successfully! ---\n")
    return history_losses

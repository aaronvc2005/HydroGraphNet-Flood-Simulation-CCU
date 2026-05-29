import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from minxiong_mesh_builder import MinxiongMeshBuilder
from model import HydroGraphNet
from pinn_loss import PhysicsInformedLoss
from train import train_hydrographnet

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def run_minxiong_flood_demo():
    print("==================================================================")
    print("   HYDROGRAPHNET: MINXIONG TOWNSHIP FLOOD ROUTING SURROGATE      ")
    print("==================================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running execution on: {device.upper()}")
    
    # 1. Load the Real Minxiong DEM and Build the Unstructured Mesh
    print("\n[Step 1] Loading real Minxiong DEM and constructing spatial GNN mesh...")
    builder = MinxiongMeshBuilder(dem_path="Terrain.dem_minxiong.tif")
    
    # Sample every 6th pixel to build a computationally efficient mesh of ~100-200 nodes
    mesh_data = builder.build_unstructured_mesh(sample_step=6)
    
    cell_centers = mesh_data["cell_centers"]
    elevations = mesh_data["cell_elevations"]
    cell_areas = mesh_data["cell_areas"]
    face_cells = mesh_data["face_cells"]
    face_widths = mesh_data["face_widths"]
    face_normals = mesh_data["face_normals"]
    num_cells = mesh_data["num_cells"]
    
    # 2. Simulate Unsteady Typhoon Flood Runoff over the Real Minxiong Topography
    print("\n[Step 2] Simulating heavy rainfall/runoff flow over Minxiong elevations...")
    time_steps = 40
    dt = 1.5
    
    # Initial state: dry land (thin sheet layer for numerical stability)
    h = np.ones(num_cells) * 0.02
    u = np.zeros(num_cells)
    v = np.zeros(num_cells)
    
    h_history = []
    v_history = []
    
    g = 9.81
    manning_n = 0.04  # Higher friction for rural/agricultural terrain
    
    # Simple explicit numerical flood routing solver flowing from High to Low elevations
    sim_steps_per_output = 30
    dt_sim = dt / sim_steps_per_output
    
    # Identify high-elevation regions where rainfall enters, and low-elevation regions (drainage)
    high_el_idx = np.where(elevations > np.percentile(elevations, 75))[0]
    
    for t_step in range(time_steps):
        for sub_step in range(sim_steps_per_output):
            # Model high rainfall inflow wave at the start of the typhoon storm (t = 10 to t = 20)
            current_time = t_step * dt + sub_step * dt_sim
            rain_inflow = 0.8 * np.exp(-((current_time - 12.0) / 6.0)**2) + 0.02
            
            # Apply rainfall runoff to the high-elevation cells
            h[high_el_idx] += rain_inflow * dt_sim
            
            dh_dt = np.zeros(num_cells)
            du_dt = np.zeros(num_cells)
            dv_dt = np.zeros(num_cells)
            
            # Compute gravity-driven downslope flow between cells
            for f_idx, (c1, c2) in enumerate(face_cells):
                width_val = face_widths[f_idx]
                normal = face_normals[f_idx]
                
                h1, u1, v1, z1 = h[c1], u[c1], v[c1], elevations[c1]
                
                if c2 != -1:
                    h2, u2, v2, z2 = h[c2], u[c2], v[c2], elevations[c2]
                else:
                    # Lowland outflow boundary
                    h2, u2, v2, z2 = h1, u1 * 0.4, v1 * 0.4, z1 - 0.5
                
                h_face = 0.5 * (h1 + h2)
                if h_face < 1e-4:
                    continue
                
                # Gravity force pulls water from high z to low z
                water_level_diff = (h2 + z2) - (h1 + z1)
                u_normal = -0.15 * g * water_level_diff * normal[0] - 0.15 * g * water_level_diff * normal[1]
                
                mass_flux = h_face * u_normal
                
                dh_dt[c1] -= mass_flux * width_val / cell_areas[c1]
                if c2 != -1:
                    dh_dt[c2] += mass_flux * width_val / cell_areas[c2]
            
            # Apply Manning friction drag
            for i in range(num_cells):
                hi = h[i]
                if hi > 1e-3:
                    vel_mag = np.sqrt(u[i]**2 + v[i]**2) + 1e-6
                    sf_x = (manning_n**2 * u[i] * vel_mag) / (hi**(4/3))
                    sf_y = (manning_n**2 * v[i] * vel_mag) / (hi**(4/3))
                    
                    # Compute down-slope gravitational acceleration
                    # Water runs down the terrain gradient
                    du_dt[i] = -sf_x * 0.5
                    dv_dt[i] = -sf_y * 0.5
            
            # Explicit Euler update
            h = np.maximum(h + dh_dt * dt_sim, 0.0)
            u = u + du_dt * dt_sim
            v = v + dv_dt * dt_sim
            
            # Cap velocities for numerical stability
            vel_norm = np.sqrt(u**2 + v**2)
            cap = 3.0
            mask = vel_norm > cap
            u[mask] = (u[mask] / vel_norm[mask]) * cap
            v[mask] = (v[mask] / vel_norm[mask]) * cap
            
        h_history.append(h.copy())
        v_history.append(np.stack([u.copy(), v.copy()], axis=-1))
        
    # Packaging simulation data
    mesh_data["depths"] = np.array(h_history)
    mesh_data["velocities"] = np.array(v_history)
    mesh_data["time_steps"] = time_steps
    
    tensors = builder.to_torch_tensors(mesh_data, device=device)
    
    # 3. Instantiate HydroGraphNet
    print("\n[Step 3] Initializing HydroGraphNet GNN and KAN network layers...")
    model = HydroGraphNet(
        in_node_features=6,  # elevation, coordinates(2), depth, velocity(2)
        edge_features_dim=3,  # face width, face normal(2)
        hidden_dim=16,
        num_message_layers=2,
        grid_size=5,
        spline_order=3
    ).to(device)
    
    pinn_loss_fn = PhysicsInformedLoss(lambda_mass=0.15, lambda_bound=0.08)
    
    # 4. Train the Model using local Minxiong terrain physics
    print("\n[Step 4] Training GNN surrogate on Minxiong topography...")
    history = train_hydrographnet(
        model=model,
        tensors=tensors,
        pinn_loss_fn=pinn_loss_fn,
        epochs=80,
        lr=0.008,
        pushforward_steps=2,
        dt=dt,
        device=device
    )
    
    # 5. Evaluate rollout predictions
    print("\n[Step 5] Evaluating GNN rollout forecasting performance...")
    model.eval()
    
    static_features = torch.cat([
        tensors["cell_elevations"].unsqueeze(-1),
        tensors["cell_centers"]
    ], dim=-1)
    
    initial_state = torch.cat([
        tensors["depths"][0].unsqueeze(-1),
        tensors["velocities"][0]
    ], dim=-1)
    
    edge_index, edge_attr = model.prepare_edges(
        tensors["face_cells"], tensors["face_widths"], tensors["face_normals"]
    )
    
    with torch.no_grad():
        rollout_trajectory = model.autoregressive_rollout(
            static_features=static_features,
            initial_state=initial_state,
            edge_index=edge_index,
            edge_attr=edge_attr,
            steps=time_steps - 1
        )
        
    # Quantitative Verification
    forecast_depths = rollout_trajectory[:, :, 0]
    target_depths = tensors["depths"]
    depth_rmse = torch.sqrt(torch.mean((forecast_depths - target_depths)**2)).item()
    
    print(f"  * Minxiong Forecast Depth RMSE: {depth_rmse:.4f} meters")
    
    # 6. Save premium plots
    print("\n[Step 6] Rendering top-tier visual results...")
    
    # Loss curves
    plt.figure(figsize=(9, 4.5))
    epochs_range = range(1, len(history["total"]) + 1)
    plt.plot(epochs_range, history["total"], label="Total Loss", color="#1a73e8", linewidth=2)
    plt.plot(epochs_range, history["data"], label="Data Loss", color="#e8710a", linestyle="--")
    plt.plot(epochs_range, history["mass"], label="Physics Loss (SWE)", color="#137333", linestyle=":")
    plt.title("HydroGraphNet Training Loss (Minxiong DEM Run)", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Magnitude (Log)")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    loss_path = "minxiong_training_loss.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"  * Saved training curve: {os.path.abspath(loss_path)}")
    
    # Inundation scatter plot at peak (t=20)
    plt.figure(figsize=(10, 8))
    centers_x = cell_centers[:, 0]
    centers_y = cell_centers[:, 1]
    peak_pred = forecast_depths[20].cpu().numpy()
    
    sc = plt.scatter(centers_x, centers_y, c=peak_pred, cmap="Blues", s=110, edgecolors="none")
    plt.colorbar(sc, label="Water Depth (m)")
    plt.title("HydroGraphNet Forecasted Flood Inundation - Minxiong Township (t=20)", fontsize=12, fontweight="bold")
    plt.xlabel("TWD97 Easting X (meters)")
    plt.ylabel("TWD97 Northing Y (meters)")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    inundation_path = "minxiong_flood_inundation.png"
    plt.savefig(inundation_path, dpi=300)
    plt.close()
    print(f"  * Saved flood map: {os.path.abspath(inundation_path)}")
    
    print("\n==================================================================")
    print("            MINXIONG RUN SUCCESSFULLY COMPLETED & PLOTTED          ")
    print("==================================================================")

if __name__ == "__main__":
    run_minxiong_flood_demo()

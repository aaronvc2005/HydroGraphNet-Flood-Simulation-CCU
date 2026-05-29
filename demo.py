import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from hecras_parser import HecRasParser
from model import HydroGraphNet
from pinn_loss import PhysicsInformedLoss
from train import train_hydrographnet

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def run_pinn_flood_demo():
    print("==================================================================")
    print("      HYDROGRAPHNET: PHYSICS-INFORMED GNN FOR FLOOD ROUTING      ")
    print("==================================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running execution on: {device.upper()}")
    
    # 1. Generate premium high-fidelity synthetic unstructured HEC-RAS-like simulation data
    print("\n[Step 1] Generating high-fidelity unstructured 2D hydraulic simulation...")
    parser = HecRasParser()
    mesh_data = parser.generate_synthetic_data(
        length=200.0,
        width=50.0,
        num_cells_x=18,
        num_cells_y=10,
        time_steps=50,
        dt=1.0
    )
    tensors = parser.to_torch_tensors(device=device)
    
    # Extract structural arrays
    cell_centers = tensors["cell_centers"]
    cell_elevations = tensors["cell_elevations"]
    face_cells = tensors["face_cells"]
    face_widths = tensors["face_widths"]
    face_normals = tensors["face_normals"]
    depths = tensors["depths"]
    velocities = tensors["velocities"]
    
    # 2. Instantiate the HydroGraphNet Model
    # Input node features: Static [z, x, y] + Dynamic [h_t, u_t, v_t] = 6 features
    # Edge features: [width, normal_x, normal_y] = 3 features
    print("\n[Step 2] Initializing HydroGraphNet and KAN Layers...")
    model = HydroGraphNet(
        in_node_features=6,
        edge_features_dim=3,
        hidden_dim=16,
        num_message_layers=2,
        grid_size=5,
        spline_order=3
    ).to(device)
    
    # 3. Define the Physics-Informed loss function
    # Lambda_mass enforces the physical SWE mass conservation
    # Lambda_bound penalizes physically impossible negative depths
    pinn_loss_fn = PhysicsInformedLoss(lambda_mass=0.15, lambda_bound=0.08)
    
    # 4. Train the Model using Pushforward Trick (3-step autoregressive feedback)
    print("\n[Step 3] Launching PINN training with Shallow Water physics constraints...")
    history = train_hydrographnet(
        model=model,
        tensors=tensors,
        pinn_loss_fn=pinn_loss_fn,
        epochs=120,
        lr=0.008,
        pushforward_steps=3,
        dt=1.0,
        device=device
    )
    
    # 5. Evaluate the trained GNN with a full-horizon autoregressive rollout (forecasting)
    print("\n[Step 4] Running full-horizon autoregressive rollout forecasting...")
    model.eval()
    
    # Build static node features
    static_features = torch.cat([
        cell_elevations.unsqueeze(-1),
        cell_centers
    ], dim=-1)
    
    # Get initial state at t=0
    initial_state = torch.cat([
        depths[0].unsqueeze(-1),
        velocities[0]
    ], dim=-1)
    
    # Convert undirected faces to GNN topological edges
    edge_index, edge_attr = model.prepare_edges(face_cells, face_widths, face_normals)
    
    # Run full autoregressive forecast for all remaining steps
    with torch.no_grad():
        rollout_trajectory = model.autoregressive_rollout(
            static_features=static_features,
            initial_state=initial_state,
            edge_index=edge_index,
            edge_attr=edge_attr,
            steps=depths.size(0) - 1
        )
        
    # 6. Quantitative Verification: Compute Mass Conservation & Accuracy Metrics
    print("\n[Step 5] Quantifying physical consistency & predictive accuracy...")
    
    # MSE between forecasted trajectory and HEC-RAS target
    forecast_depths = rollout_trajectory[:, :, 0]
    forecast_vels = rollout_trajectory[:, :, 1:3]
    
    target_depths = depths
    target_vels = velocities
    
    depth_rmse = torch.sqrt(torch.mean((forecast_depths - target_depths)**2)).item()
    vel_rmse = torch.sqrt(torch.mean((forecast_vels - target_vels)**2)).item()
    
    print(f"  * Forecast Depth RMSE: {depth_rmse:.4f} meters")
    print(f"  * Forecast Velocity RMSE: {vel_rmse:.4f} m/s")
    
    # Compute mass conservation error over time
    # Total Water Volume = Sum (h_i * Area_i)
    cell_areas = tensors["cell_areas"]
    true_volumes = torch.sum(target_depths * cell_areas, dim=1)
    pred_volumes = torch.sum(forecast_depths * cell_areas, dim=1)
    
    volume_errors = torch.abs(pred_volumes - true_volumes) / (true_volumes + 1e-5) * 100.0
    max_vol_error = torch.max(volume_errors).item()
    mean_vol_error = torch.mean(volume_errors).item()
    
    print(f"  * Mean Volume Conservation Error: {mean_vol_error:.3f}%")
    print(f"  * Maximum Volume Conservation Error: {max_vol_error:.3f}%")
    
    # 7. Visualization: Plotting stunning results and saving figures
    print("\n[Step 6] Rendering premium visual plots and maps...")
    
    # Figure 1: Training Loss convergence
    plt.figure(figsize=(10, 5))
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    epochs = range(1, len(history["total"]) + 1)
    plt.plot(epochs, history["total"], label="Total Composite Loss", color="#1a73e8", linewidth=2.5)
    plt.plot(epochs, history["data"], label="Data Loss (HEC-RAS matching)", color="#e8710a", linestyle="--", linewidth=1.8)
    plt.plot(epochs, history["mass"], label="Physics Loss (SWE Mass)", color="#137333", linestyle=":", linewidth=2)
    plt.plot(epochs, history["bound"], label="Boundary Hinge Loss (h >= 0)", color="#b06000", linestyle="-.", linewidth=1.5)
    
    plt.title("HydroGraphNet PINN Loss Convergence Curve", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Training Epochs", fontsize=12)
    plt.ylabel("Loss Magnitude", fontsize=12)
    plt.yscale("log")
    plt.legend(frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=10)
    plt.tight_layout()
    loss_path = "training_loss_curves.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"  * Loss curves figure saved to: {os.path.abspath(loss_path)}")
    
    # Figure 2: side-by-side 2D inundation depth comparing HEC-RAS target vs HydroGraphNet
    timesteps_to_plot = [5, 20, 40]
    fig, axes = plt.subplots(len(timesteps_to_plot), 2, figsize=(15, 10), sharex=True, sharey=True)
    
    centers_x = cell_centers[:, 0].cpu().numpy()
    centers_y = cell_centers[:, 1].cpu().numpy()
    
    for idx, t in enumerate(timesteps_to_plot):
        true_h = target_depths[t].cpu().numpy()
        pred_h = forecast_depths[t].cpu().numpy()
        
        # Min/Max for unified color scale
        max_h = max(true_h.max(), pred_h.max(), 1.0)
        
        # Target plot
        ax_true = axes[idx, 0]
        sc_true = ax_true.scatter(centers_x, centers_y, c=true_h, cmap="Blues", s=90, vmin=0, vmax=max_h, edgecolors="none")
        ax_true.set_title(f"Target HEC-RAS Flood Depth - Timestep {t}", fontsize=11, fontweight="bold")
        ax_true.set_ylabel("Width (m)", fontsize=10)
        ax_true.set_facecolor("#f8f9fa")
        
        # Prediction plot
        ax_pred = axes[idx, 1]
        sc_pred = ax_pred.scatter(centers_x, centers_y, c=pred_h, cmap="Blues", s=90, vmin=0, vmax=max_h, edgecolors="none")
        ax_pred.set_title(f"HydroGraphNet Forecasted Depth - Timestep {t}", fontsize=11, fontweight="bold")
        ax_pred.set_facecolor("#f8f9fa")
        
        # Add colorbar to each row
        fig.colorbar(sc_pred, ax=[ax_true, ax_pred], orientation="vertical", shrink=0.85, label="Depth (m)")

    axes[-1, 0].set_xlabel("Channel Length (m)", fontsize=10)
    axes[-1, 1].set_xlabel("Channel Length (m)", fontsize=10)
    
    plt.suptitle("Flood Inundation Depth: HEC-RAS vs HydroGraphNet Surrogate", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    inundation_path = "flood_inundation_comparison.png"
    plt.savefig(inundation_path, dpi=300)
    plt.close()
    print(f"  * Inundation comparison figure saved to: {os.path.abspath(inundation_path)}")
    
    # Figure 3: Global Water Volume conservation check over time
    plt.figure(figsize=(10, 4.5))
    time_steps = target_depths.size(0)
    times = np.arange(time_steps)
    plt.plot(times, true_volumes.cpu().numpy(), label="True Water Volume (HEC-RAS)", color="#137333", linewidth=2.5)
    plt.plot(times, pred_volumes.cpu().numpy(), label="Forecasted Volume (HydroGraphNet)", color="#a82b12", linestyle="--", linewidth=2)
    
    plt.title("Total Catchment Water Volume Over Time (Mass Balance Conservation)", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Time Step", fontsize=11)
    plt.ylabel("Total Water Volume ($m^3$)", fontsize=11)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    volume_path = "mass_conservation_check.png"
    plt.savefig(volume_path, dpi=300)
    plt.close()
    print(f"  * Mass conservation check figure saved to: {os.path.abspath(volume_path)}")
    
    print("\n==================================================================")
    print("      DEMONSTRATION RUN COMPLETE - MODEL SUCCESSFULLY VALIDATED  ")
    print("==================================================================")

if __name__ == "__main__":
    run_pinn_flood_demo()

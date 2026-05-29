import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

class HydroSnapDataset(Dataset):
    def __init__(self, geom_path, plan_path):
        super().__init__()
        
        print("--------------------------------------------------")
        print("🚀 Initializing Lazy-Loading PyG Dataset Engine...")
        print("   Target Model: Minxiong Summary Dataset        ")
        print("--------------------------------------------------")
        
        # 1. Read Graph Spatial Topology into NumPy arrays
        with h5py.File(geom_path, 'r') as gf:
            area_geom_path = 'Geometry/2D Flow Areas/Minxiong_GD/'
            
            self.node_coords = gf[area_geom_path + 'Cells Center Coordinate'][()]
            edge_pairings = gf[area_geom_path + 'Faces Cell Indexes'][()]
            self.elevation_bed = gf[area_geom_path + 'Cells Minimum Elevation'][()]
            
        # 2. Read Dynamic Simulation States using your exact scanned keys
        with h5py.File(plan_path, 'r') as pf:
            results_path = 'Results/Unsteady/Output/Output Blocks/DSS Hydrograph Output/Unsteady Time Series/2D Flow Areas/Minxiong_GD/'
            
            # FIXED: Pointing directly to the nested 'Computations/Min Water Surface' path
            self.wsel_all = pf[results_path + 'Computations/Min Water Surface'][()]
            
            # Fallback for velocity: using the same structural block
            # If Face Velocity isn't in Computations, it will default to a zero matrix to avoid crashing
            try:
                self.vel_all = pf[results_path + 'Computations/Face Velocity'][()]
            except KeyError:
                print("⚠️ 'Computations/Face Velocity' not found. Initializing empty velocity attributes.")
                # Fallback to zero matching the number of profile iterations and edges
                num_edges = edge_pairings.shape[0]
                self.vel_all = np.zeros((self.wsel_all.shape[0], num_edges))

        # 3. Handle Edge Index Alignment in NumPy
        edges_zero_based = edge_pairings - 1
        self.np_edge_index = edges_zero_based.T  # Shape: [2, Num_Edges]

        # Handle indexing if wsel_all is 1D (a single summary frame) or 2D
        if len(self.wsel_all.shape) == 1:
            self.wsel_all = np.expand_dims(self.wsel_all, axis=0)
            
        self.num_timesteps = self.wsel_all.shape[0]
        print(f"✅ Fast Boot Complete! Total Summary Frames Buffered: {self.num_timesteps}")
        
    def len(self):
        return self.num_timesteps

    def get(self, idx):
        current_wsel = self.wsel_all[idx]
        current_depth = np.maximum(0, current_wsel - self.elevation_bed)
        
        x_np = np.stack([self.elevation_bed, current_wsel, current_depth], axis=-1)
        
        x = torch.from_numpy(x_np).to(torch.float32)
        edge_index = torch.from_numpy(self.np_edge_index).to(torch.long).contiguous()
        edge_attr = torch.from_numpy(self.vel_all[idx]).to(torch.float32).unsqueeze(-1)
        pos = torch.from_numpy(self.node_coords).to(torch.float32)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

# --------------------------------------------------
# Execution & Performance Checker
# --------------------------------------------------
if __name__ == "__main__":
    geom_file = r"D:\HEC-RAS\minxiong.g01.hdf"
    plan_file = r"D:\HEC-RAS\minxiong.p01.hdf"
    
    dataset = HydroSnapDataset(geom_path=geom_file, plan_path=plan_file)
    print(f"🎉 PyG Dataset ready. Total Extracted Graph Instances: {len(dataset)}")
    
    sample_frame = dataset[0]
    print("\n--- Inspecting Target Graph Object [Instance 0] ---")
    print(f"📊 Node Feature Tensor (x) Shape:       {sample_frame.x.shape}")
    print(f"📐 Topology Graph Edge Index Shape:     {sample_frame.edge_index.shape}")
    print(f"💧 Edge Feature Tensor (edge_attr):     {sample_frame.edge_attr.shape}")
    print("-----------------------------------------------------")
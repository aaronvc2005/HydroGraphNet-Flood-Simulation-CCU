import os
import numpy as np
import torch
from scipy.spatial import Delaunay
import rasterio

class MinxiongMeshBuilder:
    """
    Utility to load a GeoTIFF Digital Elevation Model (DEM) and convert it 
    into an unstructured GNN spatial mesh compatible with HydroGraphNet.
    """
    def __init__(self, dem_path: str = "Terrain.dem_minxiong.tif"):
        self.dem_path = dem_path
        self.elevation_grid = None
        self.transform = None
        self.crs = None
        self.res = None
        self.width = 0
        self.height = 0
        
    def load_dem(self):
        """Loads elevation data and spatial metadata from the GeoTIFF."""
        if not os.path.exists(self.dem_path):
            raise FileNotFoundError(f"DEM file not found at {self.dem_path}")
            
        with rasterio.open(self.dem_path) as src:
            self.width = src.width
            self.height = src.height
            self.crs = src.crs
            self.res = src.res
            self.transform = src.transform
            band = src.read(1)
            
            # Mask nodata values
            nodata = src.nodata
            if nodata is not None:
                self.elevation_grid = np.where(band == nodata, np.nan, band)
            else:
                self.elevation_grid = band.astype(float)
                
        print(f"[DEM Loaded] Size: {self.width}x{self.height} | Res: {self.res[0]}m | CRS: {self.crs}")
        return self.elevation_grid

    def build_unstructured_mesh(self, sample_step: int = 5, noise_scale: float = 0.2) -> dict:
        """
        Converts the regular grid DEM into an unstructured Delaunay triangular mesh.
        
        Args:
            sample_step: Step size to downsample the DEM (e.g. 5 means taking every 5th pixel)
                         to keep the graph size computationally efficient.
            noise_scale: Small random perturbation to point coordinates to simulate 
                         an unstructured, real-world triangular mesh layout.
        """
        if self.elevation_grid is None:
            self.load_dem()
            
        # 1. Subsample points from the DEM grid
        rows = np.arange(0, self.height, sample_step)
        cols = np.arange(0, self.width, sample_step)
        c_grid, r_grid = np.meshgrid(cols, rows)
        
        # Get coordinates in meters using the affine transform
        xs, ys = rasterio.transform.xy(self.transform, r_grid, c_grid)
        xs = np.array(xs)
        ys = np.array(ys)
        
        # Add slight spatial noise for unstructured triangulation (excluding boundaries)
        np.random.seed(42)
        x_noise = (np.random.rand(*xs.shape) - 0.5) * self.res[0] * sample_step * noise_scale
        y_noise = (np.random.rand(*ys.shape) - 0.5) * self.res[1] * sample_step * noise_scale
        
        # Keep boundaries clean
        x_noise[:, 0] = 0
        x_noise[:, -1] = 0
        y_noise[0, :] = 0
        y_noise[-1, :] = 0
        
        perturbed_xs = xs + x_noise
        perturbed_ys = ys + y_noise
        
        # Flatten and align coordinates
        points = np.stack([perturbed_xs.ravel(), perturbed_ys.ravel()], axis=-1)
        
        # Extract elevations at these sampled grid indices
        elevations_sampled = self.elevation_grid[r_grid.ravel(), c_grid.ravel()]
        
        # Handle nan values if present (e.g. outside boundary)
        valid_mask = ~np.isnan(elevations_sampled)
        points = points[valid_mask]
        elevations_sampled = elevations_sampled[valid_mask]
        
        # 2. Perform Delaunay Triangulation to build the unstructured mesh cells
        tri = Delaunay(points)
        cells = tri.simplices
        num_cells = cells.shape[0]
        
        # Compute cell centers and areas
        cell_coords = points[cells]  # Shape: [num_cells, 3, 2]
        cell_centers = cell_coords.mean(axis=1)  # Shape: [num_cells, 2]
        
        # Area = 0.5 * |x_A(y_B - y_C) + x_B(y_C - y_A) + x_C(y_A - y_B)|
        cell_areas = 0.5 * np.abs(
            cell_coords[:, 0, 0] * (cell_coords[:, 1, 1] - cell_coords[:, 2, 1])
            + cell_coords[:, 1, 0] * (cell_coords[:, 2, 1] - cell_coords[:, 0, 1])
            + cell_coords[:, 2, 0] * (cell_coords[:, 0, 1] - cell_coords[:, 1, 1])
        )
        
        # Compute cell elevations by interpolating node elevations
        cell_elevations = elevations_sampled[cells].mean(axis=1)
        
        # 3. Establish face connectivity and construct edges
        face_dict = {}
        for c_idx, cell in enumerate(cells):
            edges = [(cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[0])]
            for edge in edges:
                sorted_edge = tuple(sorted(edge))
                if sorted_edge not in face_dict:
                    face_dict[sorted_edge] = []
                face_dict[sorted_edge].append(c_idx)
                
        face_cells = []
        face_widths = []
        face_normals = []
        
        for edge, adj_cells in face_dict.items():
            pt1 = points[edge[0]]
            pt2 = points[edge[1]]
            width_val = np.linalg.norm(pt2 - pt1)
            midpoint = 0.5 * (pt1 + pt2)
            
            # Normal vector calculation
            tangent = pt2 - pt1
            normal = np.array([-tangent[1], tangent[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            
            if len(adj_cells) == 1:
                # Boundary face
                c1 = adj_cells[0]
                c2 = -1
                c1_center = cell_centers[c1]
                if np.dot(midpoint - c1_center, normal) < 0:
                    normal = -normal
            else:
                c1 = adj_cells[0]
                c2 = adj_cells[1]
                c1_center = cell_centers[c1]
                if np.dot(normal, cell_centers[c2] - c1_center) < 0:
                    normal = -normal
                    
            face_cells.append([c1, c2])
            face_widths.append(width_val)
            face_normals.append(normal)
            
        face_cells = np.array(face_cells)
        face_widths = np.array(face_widths)
        face_normals = np.array(face_normals)
        
        print(f"[Mesh Built Successfully]")
        print(f"  * Total Nodes (Triangular Cells): {num_cells}")
        print(f"  * Total Edges (Face connections): {len(face_cells)}")
        
        return {
            "cell_centers": cell_centers,
            "cell_elevations": cell_elevations,
            "cell_areas": cell_areas,
            "face_cells": face_cells,
            "face_widths": face_widths,
            "face_normals": face_normals,
            "num_cells": num_cells
        }
        
    def to_torch_tensors(self, mesh_data: dict, device: str = "cpu") -> dict:
        """Converts generated mesh data arrays into PyTorch Tensors for GNN training."""
        return {
            "cell_centers": torch.tensor(mesh_data["cell_centers"], dtype=torch.float32, device=device),
            "cell_elevations": torch.tensor(mesh_data["cell_elevations"], dtype=torch.float32, device=device),
            "cell_areas": torch.tensor(mesh_data["cell_areas"], dtype=torch.float32, device=device),
            "face_cells": torch.tensor(mesh_data["face_cells"], dtype=torch.long, device=device),
            "face_widths": torch.tensor(mesh_data["face_widths"], dtype=torch.float32, device=device),
            "face_normals": torch.tensor(mesh_data["face_normals"], dtype=torch.float32, device=device),
        }

if __name__ == "__main__":
    builder = MinxiongMeshBuilder()
    mesh = builder.build_unstructured_mesh(sample_step=8)

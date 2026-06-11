import h5py
import numpy as np
import json
import os

def extract_2d_hec_ras_to_graph(hdf_path, area_name="Minxiong"):
    print(f"Processing 2D HEC-RAS file: {hdf_path}")
    
    with h5py.File(hdf_path, 'r') as hf:
        # Base path pointing exactly to your specific 'Minxiong' area
        base_mesh_path = f'Geometry/2D Flow Areas/{area_name}'
        
        if base_mesh_path not in hf:
            raise KeyError(f"Could not find the area group: {base_mesh_path}")
            
        # 1. Extract Nodes (Cell Centroids) using HEC-RAS 'Cells Center Coordinate'
        centroids = hf[f"{base_mesh_path}/Cells Center Coordinate"][:]
        nodes = [{"node_index": i, "node_id": f"cell_{i}", "x_coord": float(c[0]), "y_coord": float(c[1])} 
                 for i, c in enumerate(centroids)]
        print(f"Extracted {len(nodes)} grid cell nodes.")

        # 2. Extract Edges (Cell-to-Cell Face Connectivity)
        face_cells = hf[f"{base_mesh_path}/Faces Cell Indexes"][:]
        edges = []
        for c1, c2 in face_cells:
            if c1 != -1 and c2 != -1:  # Filter out boundary faces
                edges.extend([[int(c1), int(c2)], [int(c2), int(c1)]])
        print(f"Generated {len(edges)} bidirectional graph edges.")
        
        # 3. Extract Time Stamps and Hydraulic Features
        res_path = f"Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/{area_name}"
        time_stamps = [t.decode('utf-8').strip() for t in hf['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp'][:]]
        
        feature_names = ['Water Surface', 'Depth']
        feature_matrix = np.stack([hf[f"{res_path}/{var}"][:] for var in feature_names if var in hf[res_path]], axis=-1)

    return {
        "nodes": nodes,
        "edge_index": np.array(edges).T.tolist(),  # Transposes directly to [2, E] shape for GNNs
        "time_stamps": time_stamps,
        "feature_names": feature_names,
        "x": feature_matrix.tolist()
    }

if __name__ == "__main__":
    # Point directly to your absolute file location
    hdf_file = r"D:\HEC-RAS\minxiong_region.p01.hdf"
    
    # Save the output next to your parser script folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_json = os.path.join(script_dir, "hydrographnet_2d_ready.json")
    
    try:
        dataset = extract_2d_hec_ras_to_graph(hdf_file, area_name="Minxiong")
        with open(output_json, 'w') as f:
            json.dump(dataset, f, indent=4)
        print(f"Success! Model-ready data saved to: {output_json}")
    except Exception as e:
        print(f"Execution failed: {e}")
import torch
import torch.nn as nn
from kan import KAN

class HydroGraphNet(nn.Module):
    """
    HydroGraphNet: Interpretable Physics-Informed Graph Neural Network
    designed for unstructured HEC-RAS 2D flood routing models.
    Uses Kolmogorov-Arnold Networks (KAN) for message-passing and decoding layers.
    """
    def __init__(
        self,
        in_node_features: int, # static (elev, coords) + dynamic (depth, velocity)
        edge_features_dim: int, # face width, normal vectors
        hidden_dim: int = 16,
        num_message_layers: int = 2,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_message_layers = num_message_layers
        
        # 1. Node Encoder KAN
        # Encodes node features to latent dimensions
        self.node_encoder = KAN(
            layers_hidden=[in_node_features, hidden_dim, hidden_dim],
            grid_size=grid_size,
            spline_order=spline_order
        )
        
        # 2. Message-passing KAN layers
        # Computes messages: m_ji = KAN( [h_j, h_i, e_ji] )
        self.msg_layers = nn.ModuleList()
        # Updates state: h_i = h_i + KAN( [h_i, m_i] )
        self.update_layers = nn.ModuleList()
        
        for _ in range(num_message_layers):
            # Input to message is: sender_node_emb + receiver_node_emb + edge_features
            self.msg_layers.append(
                KAN(
                    layers_hidden=[2 * hidden_dim + edge_features_dim, hidden_dim, hidden_dim],
                    grid_size=grid_size,
                    spline_order=spline_order
                )
            )
            # Input to update is: node_emb + aggregated_msg
            self.update_layers.append(
                KAN(
                    layers_hidden=[2 * hidden_dim, hidden_dim, hidden_dim],
                    grid_size=grid_size,
                    spline_order=spline_order
                )
            )
            
        # 3. State Decoder KAN
        # Predicts delta change: \Delta h (depth), \Delta u, \Delta v (velocities)
        self.decoder = KAN(
            layers_hidden=[hidden_dim, hidden_dim, 3], # Outputs: [dh, du, dv]
            grid_size=grid_size,
            spline_order=spline_order
        )

    def prepare_edges(self, face_cells: torch.Tensor, face_widths: torch.Tensor, face_normals: torch.Tensor):
        """
        Processes undirected face connectivity into directed edge indices and features.
        """
        num_cells = face_cells.max().item() + 1
        
        src_list = []
        dst_list = []
        edge_attr_list = []
        
        for f_idx, (c1, c2) in enumerate(face_cells):
            c1_val = c1.item()
            c2_val = c2.item()
            width = face_widths[f_idx].unsqueeze(0)
            normal = face_normals[f_idx] # Shape: [2]
            
            # Direct Edge 1: c1 -> c2
            if c2_val != -1:
                src_list.append(c1_val)
                dst_list.append(c2_val)
                # Edge features: width, normal_x, normal_y
                edge_attr_list.append(torch.cat([width, normal]))
                
                # Direct Edge 2: c2 -> c1 (opposite normal)
                src_list.append(c2_val)
                dst_list.append(c1_val)
                edge_attr_list.append(torch.cat([width, -normal]))
            else:
                # Boundary face (c2 is -1), self-loop or reflective boundary
                src_list.append(c1_val)
                dst_list.append(c1_val)
                edge_attr_list.append(torch.cat([width, normal]))
                
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=face_cells.device)
        edge_attr = torch.stack(edge_attr_list, dim=0)
        
        return edge_index, edge_attr

    def forward(
        self,
        node_features: torch.Tensor, # Shape: [num_cells, in_node_features]
        edge_index: torch.Tensor,    # Shape: [2, num_edges]
        edge_attr: torch.Tensor      # Shape: [num_edges, edge_features_dim]
    ) -> torch.Tensor:
        """
        Forward pass for a single time step prediction.
        """
        num_cells = node_features.size(0)
        
        # 1. Encode nodes
        h = self.node_encoder(node_features) # [num_cells, hidden_dim]
        
        # 2. Process message passing
        for m in range(self.num_message_layers):
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            
            # Construct message inputs
            h_src = h[src_nodes]
            h_dst = h[dst_nodes]
            msg_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)
            
            # Pass through message KAN
            messages = self.msg_layers[m](msg_input) # [num_edges, hidden_dim]
            
            # Aggregate messages using scatter-add (index_add)
            aggregated = torch.zeros(num_cells, self.hidden_dim, device=h.device)
            aggregated.index_add_(0, dst_nodes, messages) # [num_cells, hidden_dim]
            
            # Update node representation
            update_input = torch.cat([h, aggregated], dim=-1)
            h = h + self.update_layers[m](update_input) # Residual connection
            
        # 3. Decode outputs
        delta = self.decoder(h) # [num_cells, 3] -> [dh, du, dv]
        return delta

    def autoregressive_rollout(
        self,
        static_features: torch.Tensor, # Shape: [num_cells, static_dim] (e.g. elevation, coordinates)
        initial_state: torch.Tensor,   # Shape: [num_cells, 3] (initial h, u, v)
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """
        Performs multi-step autoregressive rollout during forecasting or rollout loss.
        """
        state = initial_state
        trajectory = [state]
        
        for _ in range(steps):
            # Input features = static + dynamic state
            node_features = torch.cat([static_features, state], dim=-1)
            
            # Predict rate of change
            delta = self.forward(node_features, edge_index, edge_attr)
            
            # Update state with physical constraints
            h_next = torch.clamp(state[:, 0:1] + delta[:, 0:1], min=0.0) # mass (depth) cannot be negative
            u_next = state[:, 1:2] + delta[:, 1:2]
            v_next = state[:, 2:3] + delta[:, 2:3]
            
            state = torch.cat([h_next, u_next, v_next], dim=-1)
            trajectory.append(state)
            
        return torch.stack(trajectory, dim=0) # [steps + 1, num_cells, 3]

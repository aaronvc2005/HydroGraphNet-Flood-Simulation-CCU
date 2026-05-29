import torch

# Automatically switches to GPU if available in the cloud
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Current training device: {device}")

# When sending your graph mesh to the model:
# model = HydroGraphNet().to(device)
# data = data.to(device)
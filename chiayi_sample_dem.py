import numpy as np
import rasterio
from rasterio.transform import from_origin

# 1. Define Spatial Extent (Sample bounding box for Chiayi City area in EPSG:3826)
# Coordinates are in meters (TWD97 TM2 Zone 121)
x_min, y_max = 187000, 2598000  # Top-left corner
pixel_size = 20                 # 20-meter spatial resolution
width, height = 500, 500        # 10km x 10km grid area

# Create coordinate matrices
cols, rows = np.meshgrid(np.arange(width), np.arange(height))
x_coords = x_min + cols * pixel_size
y_coords = y_max - rows * pixel_size

# 2. Model the Regional Topography (East-to-West Slope)
# West side (left) ~ 25m, East side (right) ~ 150m
east_to_west_gradient = (cols / width) * 125 + 25

# 3. Carve a Synthetic River Channel (Simulating the Bazhang River in the South)
# We model a meandering river pathway using a sine wave function
river_center_y = 2590000 + 1500 * np.sin((x_coords - x_min) / 3000)
distance_to_river = np.abs(y_coords - river_center_y)

# Channel profile: 200m wide, dropping up to 8 meters below bank level
river_width = 200 
river_trench = np.where(distance_to_river < river_width, 
                        -8 * (1 - (distance_to_river / river_width)**2), 
                        0)

# Combine the terrain layers
final_elevation = east_to_west_gradient + river_trench

# Add slight high-frequency surface roughness for hydrodynamic realism
np.random.seed(42)
noise = np.random.normal(0, 0.2, size=(height, width))
final_elevation += noise

# 4. Save to a HEC-RAS Compatible GeoTIFF File
transform = from_origin(x_min, y_max, pixel_size, pixel_size)

metadata = {
    'driver': 'GTiff',
    'height': height,
    'width': width,
    'count': 1,
    'dtype': 'float32',
    'crs': 'EPSG:3826',  # TWD97 / TM2 zone 121
    'transform': transform
}

output_filename = "chiayi_sample_dem.tif"
with rasterio.open(output_filename, 'w', **metadata) as dst:
    dst.write(final_elevation.astype(np.float32), 1)

print(f"Success! Generated '{output_filename}' ({width}x{height} pixels at {pixel_size}m resolution).")
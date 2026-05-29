import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

dem_path = "Terrain.dem_minxiong.tif"
output_plot = "minxiong_dem_plot.png"

print("==========================================================")
print("      MINXIONG TOPOGRAPHY DEM VISUALIZATION SCRIPT        ")
print("==========================================================")

if not os.path.exists(dem_path):
    print(f"Error: DEM file '{dem_path}' not found.")
    exit(1)

# Read the file
elevation = None
crs_info = "Unknown"
resolution = "Unknown"

if HAS_RASTERIO:
    print("Reading GeoTIFF using Rasterio...")
    with rasterio.open(dem_path) as src:
        width = src.width
        height = src.height
        crs_info = str(src.crs)
        resolution = f"{src.res[0]}m x {src.res[1]}m"
        band = src.read(1)
        nodata = src.nodata
        
        # Mask out nodata
        if nodata is not None:
            elevation = np.where(band == nodata, np.nan, band)
        else:
            elevation = band.astype(float)
else:
    print("Rasterio not available, trying PIL...")
    if HAS_PIL:
        img = Image.open(dem_path)
        width, height = img.size
        elevation = np.array(img).astype(float)
    else:
        print("Error: Neither rasterio nor PIL is available. Cannot proceed.")
        exit(1)

# Compute basic stats
valid_elev = elevation[~np.isnan(elevation)]
min_e = np.min(valid_elev)
max_e = np.max(valid_elev)
mean_e = np.mean(valid_elev)

print(f"Metadata:")
print(f"  * Dimensions: {width} x {height} pixels")
print(f"  * Coordinate System: {crs_info}")
print(f"  * Resolution: {resolution}")
print(f"  * Elevation Stats:")
print(f"      - Minimum: {min_e:.2f} meters")
print(f"      - Maximum: {max_e:.2f} meters")
print(f"      - Mean:    {mean_e:.2f} meters")

# Plot the DEM
plt.figure(figsize=(10, 8))
plt.imshow(elevation, cmap="terrain", aspect="equal")
plt.colorbar(label="Elevation (meters)")
plt.title(f"Digital Elevation Model (DEM) - Minxiong Township, Chiayi\n({width}x{height} pixels, Res: {resolution})", fontsize=12, fontweight="bold", pad=15)
plt.xlabel("X (columns)")
plt.ylabel("Y (rows)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
plt.close()

print(f"\nSuccess! Plotted elevation map and saved it as: {os.path.abspath(output_plot)}")

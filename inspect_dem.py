import os
import numpy as np

# Try importing rasterio
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Try importing PIL
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

dem_path = "Terrain.dem_minxiong.tif"

print("==========================================================")
print(f"Inspecting DEM File: {dem_path}")
print("==========================================================")

if not os.path.exists(dem_path):
    print(f"Error: File '{dem_path}' does not exist in current directory.")
    exit(1)

if HAS_RASTERIO:
    print("[Using Rasterio to read GeoTIFF metadata...]")
    with rasterio.open(dem_path) as src:
        print(f"  * Dimensions (width x height): {src.width} x {src.height} pixels")
        print(f"  * Number of Bands: {src.count}")
        print(f"  * Data Type: {src.dtypes[0]}")
        print(f"  * Coordinate Reference System (CRS): {src.crs}")
        print(f"  * Bounding Box (Bounds):")
        print(f"      Left:   {src.bounds.left}")
        print(f"      Bottom: {src.bounds.bottom}")
        print(f"      Right:  {src.bounds.right}")
        print(f"      Top:    {src.bounds.top}")
        print(f"  * Spatial Resolution: {src.res} meters")
        
        # Read the elevation band
        band = src.read(1)
        # Handle nodata if present
        nodata = src.nodata
        if nodata is not None:
            valid_band = band[band != nodata]
        else:
            valid_band = band
            
        print(f"  * Elevation Statistics:")
        print(f"      Min Elevation:  {np.min(valid_band):.3f} meters")
        print(f"      Max Elevation:  {np.max(valid_band):.3f} meters")
        print(f"      Mean Elevation: {np.mean(valid_band):.3f} meters")
        print(f"      Std Elevation:  {np.std(valid_band):.3f} meters")

elif HAS_PIL:
    print("[Rasterio not installed, falling back to PIL...]")
    img = Image.open(dem_path)
    print(f"  * Image Format: {img.format}")
    print(f"  * Image Size (width x height): {img.size[0]} x {img.size[1]} pixels")
    print(f"  * Image Mode: {img.mode}")
    
    # Read band as numpy array
    band = np.array(img)
    print(f"  * Elevation Statistics:")
    print(f"      Min Value:  {np.min(band):.3f}")
    print(f"      Max Value:  {np.max(band):.3f}")
    print(f"      Mean Value: {np.mean(band):.3f}")

else:
    print("Error: Neither 'rasterio' nor 'Pillow' is installed. Cannot read the .tif file.")


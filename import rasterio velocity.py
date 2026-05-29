import rasterio

# UPDATED: Path fixed with your maximum velocity filename
tif_path = r"D:\HEC-RAS\Plan 01\Velocity (Max).Terrain (2).Terrain.dem_minxiong.tif"

with rasterio.open(tif_path) as src:
    print("--- 📊 Velocity GeoTIFF Structural Verification ---")
    print(f"📐 Image Dimensions (Pixels): {src.width}x{src.height}")
    print(f"🗺️ Coordinate Reference (CRS): {src.crs}")
    
    # Read pixel array data to check for velocity speeds
    raster_array = src.read(1)
    print(f"🏃‍♂️ Peak Velocity Value Found:  {raster_array.max()} m/s")
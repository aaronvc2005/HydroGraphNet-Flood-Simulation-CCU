import rasterio

# UPDATED: Path fixed with your maximum WSEL filename
tif_path = r"D:\HEC-RAS\Plan 01\WSE (Max).Terrain (2).Terrain.dem_minxiong.tif"

with rasterio.open(tif_path) as src:
    print("--- 📊 WSEL GeoTIFF Structural Verification ---")
    print(f"📐 Image Dimensions (Pixels): {src.width}x{src.height}")
    print(f"🗺️ Coordinate Reference (CRS): {src.crs}")
    
    # Read pixel array data to check water surface elevations
    raster_array = src.read(1)
    print(f"🏔️ Highest Water Elevation Found: {raster_array.max()} meters above sea level")
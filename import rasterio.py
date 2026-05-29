import rasterio

# UPDATED: Path fixed with your exact HEC-RAS exported file name
tif_path = r"D:\HEC-RAS\Plan 01\Depth (Max).Terrain (2).Terrain.dem_minxiong.tif"

with rasterio.open(tif_path) as src:
    print("--- 📊 GeoTIFF Structural Verification ---")
    print(f"📐 Image Dimensions (Pixels): {src.width}x{src.height}")
    print(f"💧 Total Data Bands Found:    {src.count}")
    print(f"🗺️ Coordinate Reference (CRS): {src.crs}")
    print(f"📌 Spatial Bounding Box:       {src.bounds}")
    
    # Read pixel array data to check for valid numerical flood depths
    raster_array = src.read(1)
    print(f"📈 Peak Depth Value Found:    {raster_array.max()} meters")
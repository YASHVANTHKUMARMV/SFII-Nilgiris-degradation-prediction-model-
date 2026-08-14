import ee
from config import get_roi, export_image, SCALE

def generate_dem_terrain():
    roi = get_roi()
    
    # 1. Copernicus DEM
    copernicus_dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select('DEM').mosaic().clip(roi)
    
    # 2. SRTM DEM
    srtm_dem = ee.Image("USGS/SRTMGL1_003").select('elevation').clip(roi)
    
    # 3. Elevation, 4. Slope, 5. Aspect
    # Using Copernicus DEM as base for terrain features
    terrain = ee.Terrain.products(copernicus_dem)
    elevation = terrain.select('elevation')
    slope = terrain.select('slope')
    aspect = terrain.select('aspect')
    
    # 6. Curvature (Approximation using neighborhood operations in GEE, 
    # since native curvature isn't directly available in ee.Terrain.products)
    # We can use a kernel or edge detection as a proxy or just export DEM and compute locally if needed,
    # but for an automated workflow, we can use a Laplacian filter to estimate curvature.
    laplacian_kernel = ee.Kernel.laplacian8()
    curvature = elevation.convolve(laplacian_kernel).rename('curvature')

    # Exports
    exports = [
        (copernicus_dem, "Copernicus_DEM"),
        (srtm_dem, "SRTM_DEM"),
        (elevation, "Elevation"),
        (slope, "Slope"),
        (aspect, "Aspect"),
        (curvature, "Curvature")
    ]
    
    tasks = []
    for img, name in exports:
        task = export_image(img, name, roi, scale=30) # DEM usually exported at 30m
        tasks.append(task)
        
    return tasks

if __name__ == "__main__":
    ee.Initialize()
    generate_dem_terrain()

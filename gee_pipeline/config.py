import ee

# Initialize the Earth Engine module (must be authenticated beforehand)
# ee.Initialize()

# Configuration constants
ROI_NAME = "The Nilgiris"
START_YEAR = 2018
END_YEAR = 2024
START_DATE = f'{START_YEAR}-01-01'
END_DATE = f'{END_YEAR}-12-31'

# Scale for exports
SCALE = 10  # 10m resolution for Sentinel datasets

def get_roi():
    """
    Returns the Region of Interest (Nilgiris district) as an ee.Geometry.
    Uses the FAO GAUL dataset as default.
    """
    gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")
    nilgiris = gaul.filter(ee.Filter.eq('ADM2_NAME', ROI_NAME))
    return nilgiris.geometry()

def export_image(image, description, region, scale=SCALE, folder='SFII_Datasets'):
    """
    Helper function to export an ee.Image to Google Drive.
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=region,
        scale=scale,
        maxPixels=1e13
    )
    task.start()
    print(f"Started export task: {description}")
    return task

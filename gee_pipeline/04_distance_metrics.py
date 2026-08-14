import ee
from config import get_roi, export_image, SCALE

def generate_distance_metrics():
    roi = get_roi()
    
    # Using TIGER roads as an example, but for India (Nilgiris), 
    # we should use a global dataset. 
    # Global roads from GRIP4 (Global Roads Inventory Project)
    roads = ee.FeatureCollection("projects/sat-io/open-datasets/GRIP4/Asia")
    
    # Distance to Roads
    distance_to_roads = roads.distance(searchRadius=10000, maxError=50).clip(roi).rename('Distance_to_Roads')
    
    # For Settlements, we can use the Global Human Settlement Layer (GHSL)
    ghsl = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_C/2018")
    settlements = ghsl.gt(0) # Binary mask of built-up areas
    
    # Distance to Settlements
    # fastDistanceTransform is computationally cheaper for images
    distance_to_settlements = settlements.fastDistanceTransform(256).multiply(SCALE) \
        .clip(roi).rename('Distance_to_Settlements')
        
    exports = [
        (distance_to_roads, "Distance_to_Roads"),
        (distance_to_settlements, "Distance_to_Settlements")
    ]
    
    tasks = []
    for img, name in exports:
        task = export_image(img, name, roi, scale=SCALE)
        tasks.append(task)
        
    return tasks

if __name__ == "__main__":
    ee.Initialize()
    generate_distance_metrics()

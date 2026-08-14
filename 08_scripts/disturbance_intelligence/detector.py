import os
import logging
from typing import List, Dict

logger = logging.getLogger("Disturbance.Detector")

def check_existing_layers(output_dir: str, required_variables: List[str], sensor: str = "LANDTRENDR_NIL_30m_1985_2024_UTM43N") -> Dict[str, bool]:
    """
    Checks whether the required disturbance layers already exist.
    
    Args:
        output_dir: Directory where the disturbance history files are stored.
        required_variables: List of variable prefixes (e.g., ['DISTYR', 'DISTMAG']).
        sensor: The suffix string matching the naming convention.
        
    Returns:
        Dict mapping variable name to boolean indicating existence.
    """
    existence_map = {}
    all_exist = True
    
    for var in required_variables:
        expected_filename = f"{var}_{sensor}.tif"
        expected_path = os.path.join(output_dir, expected_filename)
        
        if os.path.exists(expected_path):
            existence_map[var] = True
            logger.info(f"Detected existing layer: {expected_filename}")
        else:
            existence_map[var] = False
            all_exist = False
            logger.info(f"Missing required layer: {expected_filename}")
            
    # Also check parquet
    parquet_path = os.path.join(output_dir, "disturbance_history_db.parquet")
    if os.path.exists(parquet_path):
        existence_map['parquet_db'] = True
        logger.info(f"Detected existing database: disturbance_history_db.parquet")
    else:
        existence_map['parquet_db'] = False
        all_exist = False
        logger.info(f"Missing required database: disturbance_history_db.parquet")
            
    return existence_map, all_exist

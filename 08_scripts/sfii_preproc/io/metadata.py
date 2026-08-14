import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("SFII_Preproc.Metadata")

def generate_processing_metadata(
    output_dir: str, 
    dataset_name: str, 
    stats: Dict[str, Any]
) -> None:
    """
    Generates a JSON metadata file for the processed dataset.
    
    Args:
        output_dir (str): Directory where the metadata file will be saved.
        dataset_name (str): Name of the dataset (e.g., 's2_monthly_stacked').
        stats (Dict[str, Any]): Dictionary of processing statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {
        "dataset_name": dataset_name,
        "processing_timestamp": datetime.now().isoformat(),
        "processing_version": "1.0.0",
        "statistics": stats
    }
    
    metadata_file = os.path.join(output_dir, f"{dataset_name}_metadata.json")
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Metadata written to {metadata_file}")

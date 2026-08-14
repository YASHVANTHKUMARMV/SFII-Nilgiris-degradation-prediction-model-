import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("SFII_Preproc.Checkpoints")

class CheckpointManager:
    """Manages saving and loading of pipeline state to allow resuming."""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.state_file = os.path.join(self.checkpoint_dir, "pipeline_state.json")
        self.state = self.load_state()

    def load_state(self) -> Dict[str, Any]:
        """Loads the current checkpoint state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode checkpoint file {self.state_file}. Starting fresh.")
                return {}
        return {}

    def save_state(self) -> None:
        """Saves the current checkpoint state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)

    def mark_completed(self, step_name: str) -> None:
        """Marks a pipeline step as completed."""
        self.state[step_name] = True
        self.save_state()
        logger.info(f"Checkpoint saved: {step_name} marked as completed.")

    def is_completed(self, step_name: str) -> bool:
        """Checks if a step is already completed."""
        return self.state.get(step_name, False)

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value
        self.save_state()

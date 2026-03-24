import logging
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from config import FeatureConfig  # Use protocol
from evaluation.utils import numpy_to_pil

from config import N_LATENTS_VAL

logger = logging.getLogger(__name__)

class FeatureLoader:
    """Generic loader for feature-related images and data."""
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize with a config object adhering to FeatureConfig protocol.
        
        Args:
            config: Object with features_dir and number_of_top_k_images attributes
            features_dir_key: Key to access features directory if different from default
        """
        self.features_dir = config.features_dir
        if config.number_of_top_k_images is None:
            self.number_of_top_k_images = float("inf")
        else:
            self.number_of_top_k_images = config.number_of_top_k_images

        self.top_k_random_sample = config.top_k_random_sample
        self.top_k_random_sample_seed = config.top_k_random_sample_seed

        self.features_dir = Path(self.features_dir)
        
        if not self.features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_dir}")
        
        self.top_k_dir = self.features_dir  # Can be customized later if needed

    def list_features(self) -> List[str]:
        """List all feature names from .pt files."""
        # Get all latent files
        latent_files = list(self.top_k_dir.glob("latent_*.pt"))
        
        # Extract latent numbers and sort numerically
        def extract_latent_number(file_path):
            try:
                # Extract the number from the filename (latent_X.pt)
                return int(file_path.stem.split('_')[1])
            except (IndexError, ValueError):
                # If parsing fails, return a large number to sort it last
                return float('inf')
        
        # Sort files by the numeric value of the latent ID
        latent_files.sort(key=extract_latent_number)
        
        # Return the sorted stem names
        return [f.stem for f in latent_files]
        #return sorted([f.stem for f in self.top_k_dir.glob("latent_*.pt")])

    def _load_feature_data(self, feature_name: str, key: str) -> List[Any]:
        """Generic method to load data from .pt files."""
        top_k_path = self.top_k_dir / f"{feature_name}.pt"
        try:
            data = torch.load(top_k_path, map_location="cpu", weights_only=False)
            return data.get(key, [])[0:self.number_of_top_k_images]
        except Exception as e:
            logger.error(f"Failed to load {key} for {feature_name}: {e}")
            return []

    def load_top_k_ids(self, feature_name: str) -> List[int]:
        """Load top-k image IDs for a feature."""
        return self._load_feature_data(feature_name, "top_ids")

    def load_heatmaps(self, feature_name: str) -> List[np.ndarray]:
        """Load heatmaps for a feature."""
        return self._load_feature_data(feature_name, "heatmaps")

    def load_all_top_k_ids(self) -> Dict[str, List[int]]:
        """Load top-k image IDs for all features."""
        return {feature: self.load_top_k_ids(feature) for feature in self.list_features()}

    def load_all_top_k_ids_and_heatmaps(self, validation=False) -> Dict[str, Dict[str, List[Any]]]:
        """Load top-k image IDs and heatmaps for all features."""
        all_data = {}
        for i, feature in enumerate(self.list_features()):
            # If index latent is in range, load data
            top_k_path = self.top_k_dir / f"{feature}.pt"
            try:
                data = torch.load(top_k_path, map_location="cpu", weights_only=False)
                top_ids_raw = data.get("top_ids", [])
                top_ids = top_ids_raw.tolist() if hasattr(top_ids_raw, 'tolist') else list(top_ids_raw)
                heatmaps = data.get("heatmaps", [])

                if self.top_k_random_sample:
                    # draw k **distinct** indices uniformly at random from the slice [start, end)
                    rng = np.random.default_rng(seed=self.top_k_random_sample_seed)  # reproducible? add seed=…
                    sample_size = min(self.number_of_top_k_images, len(heatmaps))
                    random_indices = rng.choice(np.arange(0, len(heatmaps)), size=sample_size, replace=False)

                    top_ids = [top_ids[i] for i in random_indices]
                    heatmaps = [heatmaps[i] for i in random_indices]
                    all_data[feature] = {
                    "top_ids": top_ids,
                    "heatmaps": heatmaps,
                    }
                else:
                    all_data[feature] = {
                    "top_ids": top_ids[0:self.number_of_top_k_images],
                    "heatmaps": heatmaps[0:self.number_of_top_k_images],
                    }
                
            except Exception as e:
                logger.error(f"Failed to load data for {feature}: {e}")
            if validation:
                # To make this faster, only read N_LATENTS_VAL if during validation
                if i >= N_LATENTS_VAL:
                    break
        return all_data

    def load_feature_images(self, feature_name: str) -> Dict[str, List[Image.Image]]:
        """Load top-k images for a feature."""
        result = {"examples": []}
        heatmaps = self._load_feature_data(feature_name, "overlayed_heatmaps")
        result["examples"] = [numpy_to_pil(hm) for hm in heatmaps if hm is not None]
        return result
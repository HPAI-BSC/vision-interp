from dataclasses import asdict, dataclass, field
from pathlib import Path
import logging
import yaml
from typing import Literal, Dict, Any, Protocol, Optional
import os
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

REPO_DIR = os.getenv('REPO_DIR')
N_LATENTS_VAL = 500 # number of latents for validation


def get_outputs_path(subject_model_name, layer, run_type):
    # Given a subject model name and a layer, return the path to the outputs
    outputs_path = get_paths(subject_model_name, layer, path_to_get="outputs_path")
    if run_type == "validation":
        # Workaround to add validation folder to the path
        outputs_path = '/'.join(outputs_path.split('/')[:-2]) + '/validation/' + '/'.join(outputs_path.split('/')[-2:])
    return outputs_path

def get_top_k_images_path(subject_model_name, layer, subset="test", dataset_size=50000):
    # Given a subject model name and a layer, return the path to the top k images
    top_k_images_path = get_paths(subject_model_name, layer, path_to_get="top_k_images_id")
    top_k_images_path = top_k_images_path.replace('test', subset)
    top_k_images_path = top_k_images_path.replace('50000', str(dataset_size))
    return top_k_images_path
    
def get_paths(subject_model_name, layer, path_to_get: Literal["sae_path", "top_k_images_id", "outputs_path"]="sae_path"):
    """
    Extract the sae_path for a specific model name and layer number.
    
    Args:
        config_str (str): The configuration string to parse
        subject_model_name (str): The model name to search for
        layer (int): The layer number to search for
        
    Returns:
        str or None: The path if found
    """
    config_str = f"{REPO_DIR}/config/saes.yaml"
    try:
        with open(config_str, 'r') as file:
            config = yaml.safe_load(file)
        for model_id in config.keys():
            model_config = config[model_id]
            # Check if model name matches
            if model_config['model_name'] == subject_model_name:
                # Search for the target layer in SAEs list
                for sae in model_config['saes']:
                    if sae['layer'] == layer:
                        return sae[path_to_get]
    except Exception as e:
        logger.error(f"Error getting paths: {e}")

# Generic Config Protocol for FeatureLoader
class FeatureConfig(Protocol):
    features_dir: Path
    number_of_top_k_images: int = None

@dataclass
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 5000
    n: int = 1
    stop_token_ids: Optional[list[int]] = None # DO NOT TOUCH THIS

@dataclass
class ModelConfig:
    llm_path: str
    tensor_parallel_size: int = 4
    trust_remote_code: bool = True
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    limit_mm_per_prompt: int = None

@dataclass
class RefinementConfig:
    prompt_refine: Path
    model_refinement: ModelConfig


@dataclass
class SimulatorConfig:
    subject_model: str
    layer: str
    explainer_method_name: str
    features_dir: Path
    explanations_path: Path = None
    explanations_base_path: Path = None
    explanation_column: str = "explanation"
    number_of_top_k_images: int = None
    sam_model_path: Path = None
    grounding_dino_path: Path = None
    mask_image: bool = False
    run_type: str = "test"
    top_k_random_sample: bool = True
    top_k_random_sample_seed: int = 42


def load_yaml_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    try:
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        raise

def create_simulator_config_from_dict(config_dict: dict, latents_split: str = "validation", top_k_images_split: str = "test") -> SimulatorConfig:
    """Convert dictionary to SimulatorConfig object."""
    for key in ["explanations_path", "explanations_base_path", "sam_model_path", "grounding_dino_path", "explainer_method_name"]:
        if key in config_dict and config_dict[key]:
            config_dict[key] = Path(config_dict[key])
    # Path from where we read top k ids and heatmaps arrays
    # For evaluating explanations (simulation), we always use the test split
    config_dict["features_dir"] = Path(get_top_k_images_path(config_dict["subject_model"], config_dict["layer"], top_k_images_split))
    # Path from where we read the explanations
    config_dict["explanations_base_path"] = Path(get_outputs_path(config_dict["subject_model"], config_dict["layer"], latents_split)) / config_dict["explainer_method_name"]

    
    return SimulatorConfig(**config_dict)
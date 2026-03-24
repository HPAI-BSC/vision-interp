# %%
import os
import sys
import json
import glob
import torch
import argparse
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
load_dotenv()

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_DIR = _EVAL_DIR.parent
sys.path.insert(0, str(_REPO_DIR))
sys.path.insert(0, str(_REPO_DIR / 'src'))

REPO_DIR = os.getenv('REPO_DIR', str(_REPO_DIR))

from diffusers import StableDiffusion3Pipeline
from config import get_top_k_images_path, get_outputs_path
from feature_loader import FeatureLoader
from config import N_LATENTS_VAL

# Ensure the environment variables are applied
import transformers
transformers.utils.logging.set_verbosity_error()

import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


PREFIX_PROMPT = "A realistic image of "

# Based on huggingface page
NUM_INFERENCE_STEPS_MODEL = {
    'stabilityai/stable-diffusion-3.5-medium': 28,
    'stabilityai/stable-diffusion-3.5-large': 40,
}

GUIDANCE_SCALE_MODEL = {
    'stabilityai/stable-diffusion-3.5-medium': 4.5,
    'stabilityai/stable-diffusion-3.5-large': 3.5,
}

EXPLANATION_COLUMN = "explanation"
NUMBER_OF_TOP_K_IMAGES = 5

DATA_DIR = os.environ.get('DATA_DIR', os.path.join(REPO_DIR, 'data'))

@dataclass
class ImageGenerationConfig:
    subject_model: str = None
    layer: str = None
    explainer_method_name: str = None
    explanation_column: str = None
    features_dir: Path = None
    explanations_base_path: Path = None
    diffusion_model_id: str = None
    num_images: int = None
    guidance_scale: float = None
    steps: int = None
    seed: Optional[int] = None
    number_of_top_k_images: int = NUMBER_OF_TOP_K_IMAGES
    batch_size: int = None
    top_k_random_sample: bool = True
    top_k_random_sample_seed: int = 42

def create_image_generator_config_from_dict(config_dict: dict, latents_split: str = "validation", top_k_images_split: str = "test") -> ImageGenerationConfig:
    """Convert dictionary to SimulatorConfig object."""
    for key in ["explanations_path", "explanations_base_path", "explainer_method_name"]:
        if key in config_dict and config_dict[key]:
            config_dict[key] = Path(config_dict[key])
    # Path from where we read top k ids and heatmaps arrays
    config_dict["features_dir"] = Path(get_top_k_images_path(config_dict["subject_model"], config_dict["layer"], top_k_images_split))
    # Path from where we read the explanations (per-latent subdirectory structure)
    outputs_path = get_outputs_path(config_dict["subject_model"], config_dict["layer"], latents_split)
    config_dict["explanations_base_path"] = Path(outputs_path) / config_dict["explainer_method_name"]

    return ImageGenerationConfig(**config_dict)

def setup_diffusion_model(diffusion_model_id="stabilityai/stable-diffusion-3.5-medium", device="cuda"):
    """
    Set up the diffusion model for image generation.
    
    Args:
        model_id: Hugging Face model ID for the diffusion model
        device: Device to run the model on ("cuda" or "cpu")
        
    Returns:
        Configured diffusion pipeline
    """
    # Check if CUDA is available when requested
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    # Set up the pipeline with optimizations
    pipe = StableDiffusion3Pipeline.from_pretrained(
        diffusion_model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    # Move to device
    pipe = pipe.to(device)
    
    # Enable memory optimization if on CUDA
    if device == "cuda":
        pipe.enable_attention_slicing()
    
    return pipe

class ImageGeneration:
    """
    Main simulator class that compares feature activation heatmaps with
    segmentation masks produced by text-guided segmentation.
    """
    def __init__(self, config: ImageGenerationConfig, run_type: str, max_latents_test: int = None):
        """
        Initialize the simulator with the provided configuration.
        
        Args:
            config: ImageGenerationConfig object with paths and parameters
            run_type: Type of run to perform (validation or test)
            save_images: Whether to save generated images (with masks, heatmaps, etc.)
        """
        logger.info(f"Initializing Simulator with config: {config}")
        self.config = config

        self.run_type = run_type
        self.diffusion_model_id = config.diffusion_model_id
        self.num_images = config.num_images
        self.guidance_scale = config.guidance_scale
        self.steps = config.steps
        self.seed = config.seed
        self.batch_size = config.batch_size

        self.output_dir = Path(config.explanations_base_path)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize FeatureLoader with the config
        self.loader = FeatureLoader(config)

        # We select different latents for validation and test
        all_features = list(self.loader.list_features())
        if run_type == "validation":
            features = [feature for i, feature in enumerate(all_features) if i < N_LATENTS_VAL]
        elif run_type == "test":
            end = N_LATENTS_VAL + max_latents_test if max_latents_test is not None else len(all_features)
            features = [feature for i, feature in enumerate(all_features) if N_LATENTS_VAL <= i < end]
        else:
            features = all_features
        self.features = features

        # Filter features if subset is provided
        logger.info(f"Processing {len(self.features)} features from provided subset")

        self.pipe = setup_diffusion_model(self.diffusion_model_id)
        
        self.results_dir = self.output_dir
        self.results_dir.mkdir(exist_ok=True)
        logger.info(f"Diffusion model initialized, results will be saved in: {self.results_dir}")

    def generate_images(self, prompt):
        """
        Generate images using the diffusion model.
        
        Args:
            pipe: Diffusion pipeline
            prompt: Text prompt for generation
            negative_prompt: Negative text prompt (optional)
            num_images: Number of images to generate
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
            
        Returns:
            List of generated images
        """
        # Set the seed if provided
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self.pipe.device).manual_seed(self.seed)
        
        # Generate the images
        result = self.pipe(
            prompt=prompt,
            #negative_prompt=negative_prompt,
            height=512,
            width=512,
            num_images_per_prompt=self.num_images,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.steps,
            generator=generator
        )
        print(f'Result: {result}')
        return result.images
    
    def save_images(self, images, latent_dir, prefix, latent_id):
        """
        Save generated images to disk using image_gen_eval.py-compatible filenames.

        Args:
            images: List of PIL images
            latent_dir: Directory to save images
            prefix: Prefix for filenames (e.g., "positive")
            latent_id: ID of the latent being visualized
        """
        os.makedirs(latent_dir, exist_ok=True)
        for i, image in enumerate(images):
            filename = f"{prefix}_latent_{latent_id}_{i}.png"
            filepath = os.path.join(latent_dir, filename)
            image.save(filepath)
            logger.info(f"Saved image to {filepath}")


    def load_explanation(self, latent_id: int) -> str:
        """
        Load the concept explanation for a specific latent from either a single JSON or individual files.
        
        Args:
            latent_id: ID of the latent feature
            
        Returns:
            The concept explanation associated with the latent
        """
        logger.debug(f"Loading explanation for latent_id: {latent_id}")

        # Fallback to individual JSON files if base path is configured
        if self.config.explanations_base_path:
            latent_id_str = f"latent_{latent_id}"
            explanation_path = self.config.explanations_base_path / latent_id_str / "explanations" / f"explanation.json"
            if explanation_path.exists():
                with open(explanation_path, "r") as f:
                    explanation_data = json.load(f)
                    # TODO: reading the first explanation for now
                    concept_name = explanation_data[self.config.explanation_column][0]
                logger.info(f"Loaded concept name from file: {concept_name}")
                return concept_name
        
        logger.warning(f"No explanation found for latent_id: {latent_id}")
        return "generic object"  # Return a default value instead of "unknown_concept"


    def process_latents(self, feature_names: List[str]) -> Dict:
        logger.info(f"Processing features: {feature_names}")
        results = {}

        text_prompts = []
        latent_ids = []
        latent_dirs = []

        for feature_name in feature_names:
            try:
                latent_id = int(feature_name.split("latent_")[-1]) if "latent_" in feature_name else int(feature_name)
            except ValueError:
                logger.warning(f"Could not extract latent_id from feature_name {feature_name}")
                return results

            explanation = self.load_explanation(latent_id)
            text_prompt = PREFIX_PROMPT + explanation.lower()
            text_prompts.append(text_prompt)
            latent_ids.append(latent_id)

            latent_dir = self.config.explanations_base_path / f"latent_{latent_id}" / "generated_images"
            latent_dirs.append(latent_dir)

        logger.info(f"Prompts: {text_prompts}")

        images_batch = self.generate_images(text_prompts)

        for batch_idx, i in enumerate(range(0, len(images_batch), self.num_images)):
            images_latent = images_batch[i:i+self.num_images]
            latent_dir = latent_dirs[batch_idx]
            latent_id = latent_ids[batch_idx]
            self.save_images(images_latent, latent_dir, "positive", latent_id)

        return results

    def process_all_latents(self):
        """Process all available latents or the subset if specified."""
        total_features = len(self.features)
        logger.info(f"Processing {total_features} features")
        
        all_results = {}
        for idx in range(0, total_features, self.batch_size):
            feature_names = self.features[idx:idx+self.batch_size] if idx+self.batch_size <= total_features else self.features[idx:]
            logger.info(f"Processing features {idx+1}/{total_features}: {feature_names}")
            results = self.process_latents(feature_names)
            all_results.update(results)
        
        logger.info(f"Completed generating images for all latents")
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Image Generation")
    parser.add_argument("--subject_model", type=str, default="google/gemma-3-4b-it", help="Subject model")
    parser.add_argument("--layer", type=str, default="mid", help="Layer")
    parser.add_argument("--explainer_method_name", type=str, default=None, help="Explainer method name")
    parser.add_argument("--diffusion_model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium", help="Diffusion model id")
    parser.add_argument("--run_type", type=str, choices=["validation", "test"], default="test", help="Type of run to perform (validation or test)")
    parser.add_argument("--num_images", type=int, default=3,
                        help="Number of images to generate per latent and type")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--coeff", type=float, default=None,
                        help="Coefficient for the explanation")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--max_latents_test", type=int, default=None,
                        help="Number of latents to process in test split; defaults to all latents")
    args = parser.parse_args()

    assert args.explainer_method_name is not None, "Explainer method name is required"

    config = ImageGenerationConfig(
        subject_model=args.subject_model,
        layer=args.layer,
        explainer_method_name=args.explainer_method_name,
        explanation_column=EXPLANATION_COLUMN,
        diffusion_model_id=args.diffusion_model_id,
        num_images=args.num_images,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
        seed=args.seed,
        batch_size=args.batch_size,
        #n_latents=args.n_latents
    )
    config_dict = asdict(config)
    logger.info(f"Loaded configuration: {config_dict}")

    if args.coeff is not None:
        config_dict["explainer_method_name"] = config_dict["explainer_method_name"] + str(args.coeff)
    config = create_image_generator_config_from_dict(config_dict, latents_split=args.run_type)

    simulator = ImageGeneration(config, args.run_type, max_latents_test=args.max_latents_test)
    simulator.process_all_latents()
    logger.info("Simulation completed successfully")

if __name__ == "__main__":
    main()

"""
Usage:

python -m evaluation.image_generation \
  --subject_model google/gemma-3-4b-it \
  --layer mid \
  --explainer_method_name HF_easy2_masks_TOPK-5_MODEL-gemma-3-27b-it \
  --num_images 3 \
  --max_latents_test 2
"""
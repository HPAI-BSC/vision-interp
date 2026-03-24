import sys
from pathlib import Path
_EVAL_DIR = Path(__file__).resolve().parent
_REPO_DIR = _EVAL_DIR.parent
sys.path.insert(0, str(_REPO_DIR / 'src'))  # for config, feature_loader, demo_config
sys.path.insert(0, str(_EVAL_DIR))           # for evaluation/utils.py
sys.path.insert(0, str(_REPO_DIR))           # for evaluation/ package

from PIL import Image, ImageDraw, ImageFont
from feature_loader import FeatureLoader
from lang_sam import LangSAM
import numpy as np
from typing import List, Dict, Optional
import os
import json
from datasets import load_dataset
import torch
import logging
import argparse
from dataclasses import dataclass, asdict
from evaluation.utils import create_masked_image, calculate_iou

from config import SimulatorConfig, create_simulator_config_from_dict
from config import N_LATENTS_VAL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

IMAGE_SIZE = 224
SAM_MODEL_PATH = "facebook/sam2.1-hiera-large"
GROUNDING_DINO_PATH = "IDEA-Research/grounding-dino-base"
NUMBER_OF_TOP_K_IMAGES = 5
EXPLANATION_COLUMN = "explanation"

@dataclass
class BaselineSimulatorConfig:
    subject_model: str
    layer: str
    explainer_method_name: str
    explanation_column: str
    sam_model_path: Path
    grounding_dino_path: Path
    number_of_top_k_images: int


class SegmentationModel:
    """
    Wrapper for the LangSAM segmentation model that provides text-based
    image segmentation capabilities.
    """
    def __init__(self, sam_model_path: Path, grounding_dino_path: Path):
        logger.info("Initializing SegmentationModel with LangSAM")
        # Use provided paths from config or default to the model's own defaults
        sam_path = str(sam_model_path) if sam_model_path else None
        dino_path = str(grounding_dino_path) if grounding_dino_path else None

        self.model = LangSAM(
            sam_type="sam2.1_hiera_large",
            sam_ckpt_path=None,
            gdino_model_ckpt_path=None,
            gdino_processor_ckpt_path=None
        )

    def get_segmentation_mask(self, image: Image.Image, text_prompt: str) -> np.ndarray:
        # Add validation to ensure text_prompt is not None
        if text_prompt is None or not text_prompt.strip():
            logger.warning("Empty or None text prompt provided, using default")
            text_prompt = "object"

        logger.info(f"Generating segmentation mask for prompt: {text_prompt}")
        image = image.convert("RGB")
        results = self.model.predict([image], [text_prompt])
        if not results or 'masks' not in results[0] or len(results[0]['masks']) == 0:
            logger.warning(f"No masks generated for prompt '{text_prompt}' in image of size {image.size}")
            return np.zeros((image.height, image.width), dtype=np.uint8)
        mask = results[0]['masks'][0]
        return mask.astype(np.uint8)

def load_jsonl(file_path):
    """
    Load a JSONL file (JSON Lines format - one JSON object per line).

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of parsed JSON objects
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

class Simulator:
    """
    Main simulator class that compares feature activation heatmaps with
    segmentation masks produced by text-guided segmentation.
    """
    def __init__(self, config: SimulatorConfig, run_type: str, save_images: bool = False, max_latents_test: int = None):
        """
        Initialize the simulator with the provided configuration.

        Args:
            config: SimulatorConfig object with paths and parameters
            run_type: Type of run to perform (validation or test)
            save_images: Whether to save generated images (with masks, heatmaps, etc.)
        """
        logger.info(f"Initializing Simulator with config: {config}")
        self.config = config

        self.run_type = run_type
        self.save_images = save_images


        #self.output_dir = Path(config.output_dir)
        self.output_dir = Path(config.explanations_base_path)
        #self.output_dir.mkdir(exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize FeatureLoader with the config
        self.loader = FeatureLoader(config)
        self.top_k_ids_and_heatmaps = self.loader.load_all_top_k_ids_and_heatmaps(validation=run_type=="validation")
        self.segmentation_model = SegmentationModel(config.sam_model_path, config.grounding_dino_path)
        # We always load the test set of imagenet for simulation
        self.imagenet = load_dataset("ILSVRC/imagenet-1k", split="test", trust_remote_code=True)

        # We select different latents for validation and test
        if run_type == "validation":
            latents_range = list(range(0, N_LATENTS_VAL))
        elif run_type == "test":
            end = N_LATENTS_VAL + max_latents_test if max_latents_test is not None else len(self.top_k_ids_and_heatmaps)
            latents_range = list(range(N_LATENTS_VAL, end))

        features = [feature for i, feature in enumerate(self.loader.list_features()) if i in latents_range]
        print('features', features)

        self.features = features

        # Filter features if subset is provided
        #self.top_k_ids_and_heatmaps = {k: v for k, v in self.top_k_ids_and_heatmaps.items() if int(k.split("latent_")[-1]) in self.latents_range}
        logger.info(f"Processing {len(features)} features from provided subset")

        self.results_dir = self.output_dir
        self.results_dir.mkdir(exist_ok=True)
        logger.info(f"Simulator initialized, results will be saved in: {self.results_dir}")


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

    def get_image(self, image_id: int) -> Image.Image:
        logger.debug(f"Loading image with ID: {image_id}")
        image = self.imagenet[image_id]['image']
        return image

    def load_heatmap(self, feature_name: str, image_id: int, image: Image.Image) -> np.ndarray:
        logger.debug(f"Loading heatmap for feature: {feature_name}, image_id: {image_id}")
        image_array = np.array(image) if isinstance(image, Image.Image) else image
        feature_data = self.top_k_ids_and_heatmaps.get(feature_name, {})
        top_ids = feature_data.get("top_ids", [])
        heatmaps = feature_data.get("heatmaps", [])
        if image_id not in top_ids:
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)  # Return zeros with float type
        idx = top_ids.index(image_id)
        if idx >= len(heatmaps):
            return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        heatmap = heatmaps[idx]
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.numpy()
        #masked_image = create_masked_image(image_array, heatmap)
        # if heatmap.shape != (16, 16):
        #     return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        # Normalize heatmap to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_resized = heatmap_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        heatmap_array = np.array(heatmap_resized) / 255.0  # Keep continuous values
        return heatmap_array  # Return continuous values instead of binary



    def create_superposition(self, image: Image.Image, heatmap: np.ndarray, seg_mask: np.ndarray, latent_id: int, concept_name: str, iou_score: float, precision: float, recall: float) -> Dict[str, Path]:
        logger.info(f"Creating superposition for latent_id: {latent_id}")
        latent_dir = self.results_dir / f"latent_{latent_id}" / "segmentation_results"
        latent_dir.mkdir(exist_ok=True)

        image_array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32)
        if image_array.ndim == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        heatmap = heatmap.astype(np.float32)
        seg_mask = seg_mask.astype(np.float32)
        if seg_mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
            seg_mask = np.array(Image.fromarray(seg_mask).resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST))

        heatmap_color = np.zeros_like(image_array)
        seg_color = np.zeros_like(image_array)
        heatmap_color[:, :, 0] = heatmap * 255  # Use continuous heatmap values
        seg_color[:, :, 2] = seg_mask * 255

        superposition_paths = {}
        img_heatmap = (image_array * 0.5 + heatmap_color * 0.5).clip(0, 255).astype(np.uint8)
        superposition_paths["image_heatmap"] = latent_dir / "image_heatmap.png"
        if self.save_images:
            Image.fromarray(img_heatmap).save(superposition_paths["image_heatmap"])

        img_seg = (image_array * 0.5 + seg_color * 0.5).clip(0, 255).astype(np.uint8)
        superposition_paths["image_seg"] = latent_dir / "image_seg.png"
        if self.save_images:
            Image.fromarray(img_seg).save(superposition_paths["image_seg"])

        img_both = (image_array * 0.3 + heatmap_color * 0.35 + seg_color * 0.35).clip(0, 255).astype(np.uint8)
        img_both_pil = Image.fromarray(img_both)
        draw = ImageDraw.Draw(img_both_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        text_color = (255, 255, 255)
        outline_color = (0, 0, 0)
        concept_position = (10, 10)
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                draw.text((concept_position[0] + dx, concept_position[1] + dy), concept_name, font=font, fill=outline_color)
        draw.text(concept_position, concept_name, font=font, fill=text_color)

        metrics_text = f"IoU: {iou_score:.3f}\nPrec: {precision:.3f}\nRec: {recall:.3f}"
        metrics_position = (10, 30)
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                draw.text((metrics_position[0] + dx, metrics_position[1] + dy), metrics_text, font=font, fill=outline_color)
        draw.text(metrics_position, metrics_text, font=font, fill=text_color)

        superposition_paths["image_both"] = latent_dir / "image_both.png"
        if self.save_images:
            img_both_pil.save(superposition_paths["image_both"])

        return superposition_paths

    def process_latents(self, feature_name: str) -> Dict:
        logger.info(f"Processing feature: {feature_name}")
        results = {}
        feature_data = self.top_k_ids_and_heatmaps.get(feature_name, {})
        top_k_ids = feature_data.get("top_ids", [])
        if not top_k_ids:
            logger.warning(f"No top_k_ids found for feature: {feature_name}")
            return results

        try:
            latent_id = int(feature_name.split("latent_")[-1]) if "latent_" in feature_name else int(feature_name)
        except ValueError:
            logger.warning(f"Could not extract latent_id from feature_name {feature_name}")
            return results

        latent_dir = self.results_dir / f"latent_{latent_id}" / "segmentation_results"
        latent_dir.mkdir(exist_ok=True)

        text_prompt = self.load_explanation(latent_id)

        if not text_prompt or not text_prompt.strip() or "Unable to produce descriptions" in text_prompt:
            logger.warning(f"Empty explanation for latent_id {latent_id}, skipping...")
            results[f"latent_{latent_id}"] = {
                "latent_id": latent_id,
                "images": []
            }
            return results

        image_results = []
        for idx, top_image_id in enumerate(top_k_ids):
            logger.info(f"Processing image {idx + 1}/{len(top_k_ids)} (ID: {top_image_id}) for latent {latent_id}")

            image = self.get_image(top_image_id)
            seg_mask = self.segmentation_model.get_segmentation_mask(image, text_prompt)
            heatmap = self.load_heatmap(feature_name, top_image_id, image)  # Continuous heatmap

            if seg_mask.shape != heatmap.shape:
                seg_mask = np.array(Image.fromarray(seg_mask).resize(heatmap.shape[::-1], Image.NEAREST))

            # Calculate weighted IoU, precision, and recall
            iou_score, precision, recall = calculate_iou(heatmap, seg_mask)

            # Save segmentation mask and heatmap
            seg_mask_path = latent_dir / f"seg_mask_image_{top_image_id}.png"
            heatmap_path = latent_dir / f"heatmap_image_{top_image_id}.png"
            if self.save_images:
                Image.fromarray((seg_mask * 255).astype(np.uint8)).save(seg_mask_path)
                Image.fromarray((heatmap * 255).astype(np.uint8)).save(heatmap_path)

            # Create superposition images
            superposition_paths = self.create_superposition(
                image, heatmap, seg_mask, latent_id, text_prompt, iou_score, precision, recall
            )

            if self.save_images:
                new_superposition_paths = {}
                for key, path in superposition_paths.items():
                    new_path = latent_dir / f"{key}_image_{top_image_id}.png"
                    os.rename(path, new_path)
                    new_superposition_paths[key] = new_path

            result = {
                "latent_id": latent_id,
                "image_id": top_image_id,
                "concept_name": text_prompt,
                "iou_score": float(iou_score),
                "precision": float(precision),
                "recall": float(recall)
            }
            image_results.append(result)

        results[f"latent_{latent_id}"] = {
            "latent_id": latent_id,
            "images": image_results
        }

        output_path = latent_dir / "result.json"
        with open(output_path, "w") as f:
            json.dump(results[f"latent_{latent_id}"], f, indent=2)

        logger.info(f"Completed processing {len(top_k_ids)} images for {feature_name}")
        return results

    def process_all_latents(self):
        """Process all available latents or the subset if specified."""
        total_features = len(self.features)
        logger.info(f"Processing {total_features} features")

        all_results = {}
        for idx, feature_name in enumerate(self.features):
            logger.info(f"Processing feature {idx+1}/{total_features}: {feature_name}")
            results = self.process_latents(feature_name)
            all_results.update(results)

        # Save combined results
        combined_results_path = self.results_dir / "combined_segmentation_results.json"
        with open(combined_results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Completed processing all features. Combined results saved to {combined_results_path}")
        return all_results

def main():
    parser = argparse.ArgumentParser(description="Baseline Simulator for Feature Interpretation")
    parser.add_argument("--subject_model", type=str, default="google/gemma-3-4b-it", help="Subject model")
    parser.add_argument("--layer", type=str, default="mid", help="Layer")
    parser.add_argument("--explainer_method_name", type=str, default=None, help="Explainer method name")
    parser.add_argument("--run_type", type=str, choices=["validation", "test"], default="test", help="Type of run to perform (validation or test)")
    parser.add_argument("--save_images", action="store_true", default=False, help="Whether to save generated images (with masks, heatmaps, etc.)")
    parser.add_argument("--max_latents_test", type=int, default=None, help="Number of latents to process in test split; defaults to all latents")
    args = parser.parse_args()

    assert args.explainer_method_name is not None, "Explainer method name is required"
    if args.run_type == "test":
        assert args.save_images == False, "Cannot run test set with saving images"

    config = BaselineSimulatorConfig(
        subject_model=args.subject_model,
        layer=args.layer,
        explainer_method_name=args.explainer_method_name,
        explanation_column=EXPLANATION_COLUMN,
        sam_model_path=SAM_MODEL_PATH,
        grounding_dino_path=GROUNDING_DINO_PATH,
        number_of_top_k_images=NUMBER_OF_TOP_K_IMAGES
    )
    config_dict = asdict(config)
    logger.info(f"Loaded configuration: {config_dict}")

    config = create_simulator_config_from_dict(config_dict, latents_split=args.run_type)

    simulator = Simulator(config, args.run_type, args.save_images, max_latents_test=args.max_latents_test)
    simulator.process_all_latents()
    logger.info("Simulation completed successfully")

if __name__ == "__main__":
    main()

"""
Usage:
python -m evaluation.baseline_simulator \
        --explainer_method_name steering2_google_gemma-3-27b-it_sampling-False_blank_input_200 \
        --layer mid \
        --run_type test \
        --max_latents_test 1000
"""
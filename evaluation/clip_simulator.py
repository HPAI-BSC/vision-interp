import sys
from pathlib import Path
_EVAL_DIR = Path(__file__).resolve().parent
_REPO_DIR = _EVAL_DIR.parent
sys.path.insert(0, str(_REPO_DIR / 'src'))  # for config, feature_loader, demo_config
sys.path.insert(0, str(_EVAL_DIR))           # for evaluation/utils.py
sys.path.insert(0, str(_REPO_DIR))           # ensure utils/ package takes precedence over evaluation/utils.py

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation,zoom
import json
import logging
import numpy as np
from PIL import Image
from typing import List, Dict
import torch
from feature_loader import FeatureLoader
from datasets import load_from_disk, load_dataset
#from configs import load_config, PipelineConfig
#from torch.nn import CosineSimilarity
from config import create_simulator_config_from_dict
import os
import torch.nn.functional as F
from torchvision import transforms
from config import N_LATENTS_VAL

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)



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


class CLIPSimulator:
    def __init__(self, config, output_dir: str, run_type: str, explanations_json_path: str = None, scorer: str = 'clip',
                 compute_patch_ce: bool = False, max_latents_test: int = None):
        """
        Initialize the simulator with paths to ImageNet dataset, output directory, and optional explanations JSON.

        Args:
            output_dir: Directory where results will be saved
            explanations_json_path: Optional path to a single JSON file containing all explanations
        """
        logger.info(f"Initializing Simulator, output_dir: {output_dir}")
        self.results_dir = Path(output_dir)
        self.run_type = run_type
        self.explanations_json_path = Path(explanations_json_path) if explanations_json_path else None
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.explanations_base_path = Path(config.explanations_base_path)
        #self.loader = FeatureLoader(load_config(Path("configs/simulator.yaml"), PipelineConfig))
        self.loader = FeatureLoader(config)
        self.imagenet = load_dataset("ILSVRC/imagenet-1k", split="test", trust_remote_code=True)
        self.top_k_ids_and_heatmaps = self.loader.load_all_top_k_ids_and_heatmaps(validation=run_type=="validation")
        self.config = config

        # We select different latents for validation and test
        if run_type == "validation":
            latents_range = list(range(0, N_LATENTS_VAL))
        elif run_type == "test":
            end = N_LATENTS_VAL + max_latents_test if max_latents_test is not None else len(self.top_k_ids_and_heatmaps)
            latents_range = list(range(N_LATENTS_VAL, end))

        features = [feature for i, feature in enumerate(self.loader.list_features()) if i in latents_range]
        print('len features', len(features))

        self.features = features

        # Adjust results_dir based on input
        if "steering" in str(config.explanations_base_path):
            tag = "steering"
        elif "topk" in str(config.explanations_base_path) or "TOPK" in str(config.explanations_base_path):
            tag = "topk"
        else:
            print(config.explanations_base_path)
        if config.mask_image:
            self.mask_img = True
            tag = tag + "_masked"
        else:
            self.mask_img = False
            tag = tag + "_unmasked"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Simulator initialized, results will be saved in: {self.results_dir}")

        self.scorer = scorer
        #self.sae_id = config.sae_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.compute_patch_ce = compute_patch_ce
        self.n_images_to_consider = int(config.number_of_top_k_images)
        self.model, self.processor = self.load_clip()

    def compute_per_patch_attribution(self, image, text_prompt, heatmap):
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        image_shape = image_tensor.shape
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract image features (excluding [CLS])
        with torch.no_grad():
            image_features = self.model.vision_model(pixel_values=inputs["pixel_values"]).last_hidden_state
            image_patches = image_features[:, 1:, :]  # Remove [CLS], shape: [1, 256, 1024]

            # Project patches using visual projection
            vision_projection = self.model.visual_projection
            image_patches_projected = vision_projection(image_patches)
            image_patches_projected = F.normalize(image_patches_projected, dim=-1)

            # Encode text
            text_features = self.model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).pooler_output  # We get the text [cls] token -> [1,768]
            text_projection = self.model.text_projection
            text_projected = text_projection(text_features)
            text_projected = F.normalize(text_projected, dim=-1)

            # Compute similarities
            similarities = (image_patches_projected @ text_projected.T).squeeze()  # Shape: [256]

        # Generate heatmap
        pred_heatmap = similarities.reshape(16, 16).cpu().numpy()  # Shape 256
        pred_heatmap = (pred_heatmap - pred_heatmap.min()) / (pred_heatmap.max() - pred_heatmap.min() + 1e-8)

        upsampled_heatmap_224 = torch.nn.functional.interpolate(
            torch.tensor(pred_heatmap).unsqueeze(0).unsqueeze(0),
            size=224,
            mode='bicubic',
            align_corners=False
        ).squeeze().numpy()

        upsampled_heatmap = torch.nn.functional.interpolate(
            torch.tensor(pred_heatmap).unsqueeze(0).unsqueeze(0),
            size=image_shape[1:],
            mode='bicubic',
            align_corners=False
        ).squeeze().numpy()

        heatmap_ten = torch.from_numpy(heatmap).float()

        gt_heatmap = F.interpolate(
            heatmap_ten.unsqueeze(0).unsqueeze(0),
            size=image_shape[1:],
            mode='bilinear',
            align_corners=False
        ).squeeze()

        gt_lowres = F.adaptive_avg_pool2d(
            heatmap_ten.unsqueeze(0).unsqueeze(0),
            (16, 16)
        ).squeeze()

        upsampled_heatmap = torch.from_numpy(upsampled_heatmap)
        pred_heatmap = torch.from_numpy(pred_heatmap)
        upsampled_heatmap_224 = torch.from_numpy(upsampled_heatmap_224)
        loss_lowres = F.cross_entropy(pred_heatmap, gt_lowres)

        return loss_lowres.item()

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

    def load_heatmap(self, feature_name: str, image_id: int) -> np.ndarray:
        logger.debug(f"Loading heatmap for feature: {feature_name}, image_id: {image_id}")
        feature_data = self.top_k_ids_and_heatmaps.get(feature_name, {})
        top_ids = feature_data.get("top_ids", [])
        heatmaps = feature_data.get("heatmaps", [])
        if image_id not in top_ids:
            return np.zeros((224, 224), dtype=np.float32)  # Return zeros with float type
        idx = top_ids.index(image_id)
        if idx >= len(heatmaps):
            return np.zeros((224, 224), dtype=np.float32)
        heatmap = heatmaps[idx]
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.numpy()
        """
        if heatmap.shape != (16, 16):
            print('c')
            return np.zeros((224, 224), dtype=np.float32)
        """
        # Normalize heatmap to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_resized = heatmap_img.resize((224, 224), Image.NEAREST)
        heatmap_array = np.array(heatmap_resized) / 255.0  # Keep continuous values
        return heatmap_array  # Return continuous values instead of binary

    def load_clip(self):
        from transformers import CLIPProcessor, CLIPModel
        model_directory = 'openai/clip-vit-large-patch14'

        model = CLIPModel.from_pretrained(model_directory)
        processor = CLIPProcessor.from_pretrained(model_directory)

        model.to(self.device)
        return model, processor


    def mask_image_from_heatmap(self, image: Image.Image, heatmap: np.ndarray, patch_size: int = 1) -> np.ndarray:
        """
            Mask the image using an activation heatmap.

            The heatmap is a binary NumPy array (0 for inactive, 1 for active).
            For each active pixel, a patch of size patch_size x patch_size is expanded
            and the union of these patches is used to preserve parts of the image.

            Args:
                image (PIL.Image.Image): The input image.
                heatmap (np.ndarray): A binary array with shape (H, W) indicating activated pixels.
                patch_size (int): The size of the square patch to expand around each active pixel.

            Returns:
                PIL.Image.Image: The masked image with non-preserved regions set to zero.
        """
        binary_mask = (heatmap != 0.0).astype(int)
        if patch_size == 1:
            # No binary dilation
            mask = binary_mask
        else:
            patch = np.ones((patch_size, patch_size), dtype=bool)
            mask = binary_dilation(binary_mask, structure=patch)

        image_np = np.array(image)

        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)

        if image_np.ndim == 3:
            mask_expanded = np.repeat(mask[:, :, np.newaxis], image_np.shape[2], axis=2)
        else:
            mask_expanded = mask

        if image_np.shape != mask_expanded.shape:
            zoom_factors = image_np.shape / np.array(mask_expanded.shape)

            # Apply zoom (different scaling for each axis)
            mask_expanded = zoom(mask_expanded, zoom_factors, order=1)  # order=1 for bilinear interpolation

        masked_image_np = image_np * mask_expanded.astype(image_np.dtype)
        return masked_image_np

    def process_latents(self, feature_name: str) -> Dict:
        """
        Process a specific latent feature by generating and comparing heatmaps and segmentation masks.

        Args:
            feature_name: Name of the feature to process

        Returns:
            Dictionary containing the analysis results
        """
        logger.info(f"Processing feature: {feature_name}")
        results = {}
        feature_data = self.top_k_ids_and_heatmaps.get(feature_name, {})
        top_k_ids = feature_data.get("top_ids", [])

        if not top_k_ids:
            logger.warning(f"No top_k_ids found for {feature_name}")
            return results

        # Extract the latent ID from the feature name
        try:
            latent_id = int(feature_name.split("latent_")[-1]) if "latent_" in feature_name else int(feature_name)
        except ValueError:
            logger.warning(f"Could not extract latent_id from feature_name {feature_name}")
            return results

        # Create directory for this latent's results
        latent_dir = self.results_dir / f"latent_{latent_id}" / "clip_scores"

        # Check if parent directory exists
        if not latent_dir.parent.exists():
            raise FileNotFoundError(f"Base directory does not exist for latent {latent_id}: {latent_dir.parent}")

        latent_dir.mkdir(exist_ok=True)

        # Use the top-ranking image for this feature
        top_image_ids = []
        if self.n_images_to_consider == 1:
            top_image_id = top_k_ids[0]
            top_image_ids.append(top_image_id)
            image = self.get_image(top_image_id)
        else:
            image,heatmap = [],[]
            for i in range(min(self.n_images_to_consider, len(top_k_ids))):
                image_id = top_k_ids[i]
                top_image_ids.append(image_id)
                image_sample = self.get_image(image_id)
                heatmap_sample = self.load_heatmap(feature_name, image_id)
                if self.mask_img:
                    image_sample = self.mask_image_from_heatmap(image_sample, heatmap_sample, patch_size=1)
                image.append(image_sample)
                heatmap.append(heatmap_sample)

        # Save images used for CLIP scoring to a tmp folder
        # tmp_dir = latent_dir / "tmp_images"
        # print('tmp_dir', tmp_dir)
        # tmp_dir.mkdir(exist_ok=True)
        # images_to_save = image if isinstance(image, list) else [image]
        # for idx, img in enumerate(images_to_save):
        #     if isinstance(img, np.ndarray):
        #         img = Image.fromarray(img.astype(np.uint8))
        #     img.save(tmp_dir / f"image_{idx}.png")

        # Get the concept explanation and generate masked image
        text_prompt = [self.load_explanation(latent_id)]


        if self.scorer == "clip":
            with torch.no_grad():
                try:
                    inputs = self.processor(text=text_prompt, images=image, return_tensors="pt", truncation=True, padding=True)
                except Exception as e:
                    logger.warning(f"Could not process inputs for latent {latent_id}.")
                    logger.warning(e)
                    return results
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)

                image_text_similarity = outputs.logits_per_image

                logit_scale = self.model.logit_scale
                cosine_sim = image_text_similarity / logit_scale.exp()
                cosine_sim = cosine_sim.detach().cpu().numpy().flatten().tolist()

                logging.info(f"Latent id {latent_id}, num images considered {self.n_images_to_consider}. Similarity: {cosine_sim}")

                if self.compute_patch_ce:
                    ce = self.compute_per_patch_attribution(image, text_prompt, heatmap)
                else:
                    ce = np.inf

        else:
            raise NotImplementedError

        # Compile results
        result = {
            "latent_id": latent_id,
            "top_image_id": top_image_ids,
            "concept_name": text_prompt,
            "cos_sim_score": cosine_sim,
            "patch_ce": ce,
        }
        results[f"latent_{latent_id}"] = result

        # Save results to JSON file
        output_path = latent_dir / "result.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved result to {output_path}")

        return results


def main():
    """
    Main entry point of the program that initializes the simulator and
    processes all latent features.
    """
    parser = argparse.ArgumentParser(description="CLIP-based SAE feature explanation evaluator")
    parser.add_argument("--subject_model", type=str, default="google/gemma-3-4b-it",
                        help="HuggingFace model ID of the subject model")
    parser.add_argument("--layer", type=str, default="mid",
                        help="Which layer to evaluate (e.g. 'mid' or 'later')")
    parser.add_argument("--explainer_method_name", type=str, required=True,
                        help="Name of the explanation method to evaluate")
    parser.add_argument("--run_type", type=str, default="test", choices=["validation", "test"],
                        help="Split to evaluate")
    parser.add_argument("--number_of_top_k_images", type=int, default=5,
                        help="Number of top-k images to use per latent")
    parser.add_argument(
                        "--mask_image",
                        type=bool,
                        default=True,
                        help="Mask images by heatmap activation before CLIP scoring (default: True)")
    parser.add_argument("--max_latents_test", type=int, default=None,
                        help="Last latent index (exclusive) for test split; defaults to all latents")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose logging output")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)

    config_dict = {
        "subject_model": args.subject_model,
        "layer": args.layer,
        "explainer_method_name": args.explainer_method_name,
        "run_type": args.run_type,
        "number_of_top_k_images": args.number_of_top_k_images,
        "mask_image": args.mask_image,
    }
    config = create_simulator_config_from_dict(config_dict, latents_split=args.run_type)

    simulator = CLIPSimulator(
        config = config,
        output_dir=config.explanations_base_path,
        explanations_json_path=None,
        run_type=config.run_type,
        scorer="clip",
        compute_patch_ce=False,
        max_latents_test=args.max_latents_test
    )

    # Process each feature and save the results
    for feature_name in tqdm(simulator.features, desc="Processing latents"):
        _ = simulator.process_latents(feature_name)
        logger.info(f"Completed processing {feature_name}. Results saved to {simulator.results_dir}.")

    logger.info(f"Done! Bye.")


if __name__ == "__main__":
    main()

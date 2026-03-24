import os
import sys
import json
import glob
import torch
import argparse
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from dotenv import dotenv_values

_EVAL_DIR = Path(__file__).resolve().parent
_REPO_DIR = _EVAL_DIR.parent
sys.path.insert(0, str(_REPO_DIR))           # for utils/ package
sys.path.insert(0, str(_REPO_DIR / 'src'))  # for config, feature_loader, demo_config

REPO_DIR = os.environ.get('REPO_DIR', str(Path(__file__).resolve().parent.parent))
sys.path.append(REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, 'src'))

env_vars = dotenv_values(os.path.join(REPO_DIR, '.env'))
# Set environment variables from env_vars
for key, value in env_vars.items():
    os.environ[key] = value

# Import necessary utilities
from utils.utils import load_model, get_model_img_size, get_model_patch_size, resolve_attr
from utils.sae_utils import get_patch_sae_codes
from utils.hf_hook_utils import get_module_output
from dictionary_learning.utils import load_dictionary
import demo_config
from demo_config import get_activation_dim
from datasets import load_dataset

from dataclasses import dataclass, asdict
import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

from config import N_LATENTS_VAL, get_paths, get_outputs_path, get_top_k_images_path
from feature_loader import FeatureLoader

DATA_DIR = os.environ.get('DATA_DIR')
SAES_DIR = os.environ.get('SAES_DIR')
EXPLANATION_COLUMN = "explanation"

@dataclass
class ImageGenerationConfig:
    subject_model: str
    layer: str
    explainer_method_name: str
    explanation_column: str


def load_explanations(base_dir):
    """
    Load explanations from per-latent explanation.json files.

    Args:
        base_dir: Base directory containing latent_* subdirectories

    Returns:
        Dictionary mapping latent IDs to their explanations
    """
    latent_dirs = glob.glob(os.path.join(base_dir, "latent_*"))

    latent_explanations = {}
    for latent_dir in latent_dirs:
        try:
            latent_id = int(os.path.basename(latent_dir).split('_')[1])
        except (IndexError, ValueError):
            continue
        explanation_path = os.path.join(latent_dir, "explanations", "explanation.json")
        if os.path.exists(explanation_path):
            with open(explanation_path, 'r') as f:
                latent_explanations[latent_id] = json.load(f)

    print(f"Loaded explanations for {len(latent_explanations)} latents")
    return latent_explanations

def load_generated_images(base_dir, latent_id, image_type="positive"):
    """
    Load generated images for a specific latent.

    Args:
        base_dir: Base directory containing latent_* subdirectories
        latent_id: ID of the latent to load images for
        image_type: Type of images to load ("positive" or "negative")

    Returns:
        List of PIL images
    """
    latent_dir = os.path.join(base_dir, f"latent_{latent_id}", "generated_images")
    image_files = glob.glob(os.path.join(latent_dir, f"{image_type}_latent_{latent_id}_*.png"))

    images = []
    for image_file in sorted(image_files):
        try:
            img = Image.open(image_file)
            images.append(img)
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")

    return images

def load_random_imagenet_images(dataset, num_images=5, seed=None):
    """
    Load random images from ImageNet test set.

    Args:
        dataset_path: Path to the ImageNet dataset
        num_images: Number of random images to load
        seed: Random seed for reproducibility

    Returns:
        List of PIL images
    """
    if seed is not None:
        random.seed(seed)

    # Load the dataset
    try:
        # Select random indices
        indices = random.sample(range(len(dataset)), num_images)
        print('indices', indices)

        # Load the images
        images = [dataset[idx]['image'] for idx in indices]
        return images

    except Exception as e:
        print(f"Error loading ImageNet dataset: {e}")
        return []

def get_sae_activations(model, images, tokenizer, processor, sae, cfg):
    """
    Get SAE activations for a batch of images.

    Args:
        model: The vision model (with model.submodule set)
        images: List of PIL images
        tokenizer: Tokenizer for the model
        processor: Processor for the model
        sae: Sparse autoencoder model
        cfg: Configuration object

    Returns:
        Tensor of SAE activations
    """
    # Process images for the model
    #images_batch = [{'image': [image]} for image in images]  # each image wrapped in a list
    images_batch = [{'image': [image]} for image in images] # Prepare input for get_module_output

    # Get model activations
    batch_layer_out = get_module_output(model, images_batch, tokenizer, cfg, processor)

    # Get SAE codes
    num_patches = get_model_img_size(cfg.model_name) // get_model_patch_size(cfg.model_name)
    sae_codes = get_patch_sae_codes(sae, batch_layer_out, num_patches=num_patches)

    return sae_codes

def compute_activation_statistics(sae_codes, target_latent_id=None):
    """
    Compute statistics for SAE activations.

    Args:
        sae_codes: Tensor of SAE activations [batch_size, height, width, num_latents]
        target_latent_id: If provided, only analyze this specific latent

    Returns:
        Dictionary of activation statistics, including lists of per-image max and mean values.
    """
    # If target_latent_id is provided, extract only that latent's activations
    if target_latent_id is not None:
        latent_activations = sae_codes[:, :, :, target_latent_id]

        max_values = torch.amax(latent_activations, dim=(1, 2))
        mean_values = torch.mean(latent_activations, dim=(1, 2))

        # Compute statistics for the target latent
        stats = {
            "mean_of_max": float(max_values.mean().item()),
            "mean_of_mean": float(mean_values.mean().item()),
            "std_of_max": float(max_values.std().item()),
            "std_of_mean": float(mean_values.std().item()),
            "sparsity": float((latent_activations > 0).float().mean().item()),
            "max_values": max_values.cpu().tolist(),  # Store all max values per image
            "mean_values": mean_values.cpu().tolist(),  # Store all mean values per image
        }
        return {target_latent_id: stats}
    else:
        # Compute statistics across all latents (not used in the main loop for this version)
        # Average across spatial dimensions first
        spatial_mean = sae_codes.mean(dim=(1, 2))  # [batch_size, num_latents]

        # Then compute statistics across the batch
        batch_mean = spatial_mean.mean(dim=0)  # [num_latents]
        batch_max = spatial_mean.max(dim=0)[0]  # [num_latents]
        batch_min = spatial_mean.min(dim=0)[0]  # [num_latents]
        batch_std = spatial_mean.std(dim=0)  # [num_latents]

        # Compute sparsity (fraction of activations > 0)
        sparsity = (sae_codes > 0).float().mean(dim=(0, 1, 2))  # [num_latents]

        # Convert to Python types for JSON serialization
        stats = {
            "mean": batch_mean.cpu().tolist(),
            "max": batch_max.cpu().tolist(),
            "min": batch_min.cpu().tolist(),
            "std": batch_std.cpu().tolist(),
            "sparsity": sparsity.cpu().tolist(),
        }
        return stats

def compute_auroc(positive_values, negative_values):
    """
    Compute the Area Under the ROC Curve (AUROC) for distinguishing positive from negative examples.

    Args:
        positive_values: List of activation values for positive examples
        negative_values: List of activation values for negative examples

    Returns:
        AUROC score, fpr, tpr
    """
    if not positive_values or not negative_values:
        print("Warning: Cannot compute AUROC with empty lists.")
        return 0.0, np.array([0, 1]), np.array([0, 1]) # Return default values

    # Create labels (1 for positive, 0 for negative)
    y_true = np.array([1] * len(positive_values) + [0] * len(negative_values))

    # Combine scores
    y_scores = np.array(positive_values + negative_values)

    # Compute ROC curve and AUROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = auc(fpr, tpr)

    return auroc, fpr, tpr

def plot_activation_comparison(positive_stats, negative_stats, latent_id, save_dir=None):
    """
    Plot comparison of activations between positive and negative images.

    Args:
        positive_stats: Statistics for positive images
        negative_stats: Statistics for negative images
        latent_id: ID of the latent being analyzed
        save_dir: Directory to save the plot (if None, display instead)

    Returns:
        None
    """
    plt.figure(figsize=(12, 8))

    # Extract data for plotting
    metrics = ["mean_of_max", "mean_of_mean", "sparsity"]
    pos_values = [positive_stats[latent_id][metric] for metric in metrics]
    neg_values = [negative_stats[latent_id][metric] for metric in metrics]

    # Create bar plot
    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width/2, pos_values, width, label='Generated Images')
    plt.bar(x + width/2, neg_values, width, label='Random ImageNet Images')

    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title(f'Activation Statistics for Latent {latent_id}')
    plt.xticks(x, metrics)
    plt.legend()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"latent_{latent_id}_comparison.png"))
        plt.close()
    else:
        plt.show()

def plot_roc_curve(fpr, tpr, auroc, title_suffix, save_dir=None):
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auroc: Area under ROC curve
        title_suffix: Suffix for the plot title (e.g., "Global Max Activations")
        save_dir: Directory to save the plot

    Returns:
        None
    """
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({title_suffix})')
    plt.legend(loc="lower right")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"roc_curve_{title_suffix.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()
    else:
        plt.show()

def plot_activation_distributions(positive_values, negative_values, title_suffix, save_dir=None):
    """
    Plot distributions of activations for positive and negative images.

    Args:
        positive_values: List of activation values for positive images
        negative_values: List of activation values for negative images
        title_suffix: Suffix for the plot title (e.g., "Global Max Activations")
        save_dir: Directory to save the plot

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))

    # Plot histograms
    sns.histplot(positive_values, kde=True, color='blue', alpha=0.5, label='Generated Images')
    sns.histplot(negative_values, kde=True, color='red', alpha=0.5, label='Random ImageNet Images')

    # Add vertical lines for means
    plt.axvline(np.mean(positive_values), color='blue', linestyle='--',
                label=f'Generated Mean: {np.mean(positive_values):.4f}')
    plt.axvline(np.mean(negative_values), color='red', linestyle='--',
                label=f'Random Mean: {np.mean(negative_values):.4f}')

    plt.xlabel(f'Activation Value')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Activations ({title_suffix})')
    plt.legend()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"distribution_{title_suffix.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(save_dir, filename))
        plt.close()
    else:
        plt.show()

def main(args):
    data_dir = DATA_DIR or os.path.join(REPO_DIR, "data")

    # Derive sae_path and sae_id from subject_model and layer
    sae_path = get_paths(args.subject_model, args.layer, "sae_path")
    outputs_path = get_outputs_path(args.subject_model, args.layer, args.run_type)
    sae_id = str(outputs_path).replace(os.path.join(data_dir, 'outputs') + os.sep, '').replace(
        os.path.join(data_dir, 'outputs') + '/', '')

    # Construct paths
    base_dir = os.path.join(data_dir, "outputs", sae_id, args.explanation_mode)
    results_dir = os.path.join(base_dir, 'image_generation_evaluation_results')

    # Load explanations
    latent_explanations = load_explanations(base_dir)

    # Get ordered feature list via FeatureLoader (same as other evaluators)
    class FeatureConfig:
        features_dir = get_top_k_images_path(args.subject_model, args.layer, "test")
        number_of_top_k_images = None
        top_k_random_sample = True
        top_k_random_sample_seed = 42

    all_features = FeatureLoader(FeatureConfig()).list_features()
    if args.run_type == "validation":
        selected_features = [f for i, f in enumerate(all_features) if i < N_LATENTS_VAL]
    else:  # test
        end = N_LATENTS_VAL + args.max_latents_test if args.max_latents_test is not None else len(all_features)
        selected_features = [f for i, f in enumerate(all_features) if N_LATENTS_VAL <= i < end]
    selected_ids = [int(f.split("latent_")[-1]) for f in selected_features]

    # Load SAE
    sae, sae_config = load_dictionary(sae_path, args.device)

    # Extract model configuration from SAE config
    model_name = sae_config['trainer']['lm_name']
    model_path = demo_config.LLM_CONFIG[model_name].model_path
    model_type = demo_config.LLM_CONFIG[model_name].model_type

    # Extract SAE configuration
    dict_size = sae_config['trainer']['dict_size']
    submodule_name = sae_config['trainer']['submodule_name']
    submodule_parts = submodule_name.split('_')
    submodel = submodule_parts[0]
    site = submodule_parts[1]
    io = submodule_parts[2]
    layer = int(submodule_parts[4])

    # Set up configuration
    class Config:
        def __init__(self):
            self.model_name = model_name
            self.model_path = model_path
            self.model_type = model_type
            self.submodel = submodel
            self.site = site
            self.io = io
            self.layer = layer
            self.activation_dim = get_activation_dim(model_name, submodel)
            self.dtype = demo_config.LLM_CONFIG[model_name].dtype
            self.device = args.device
            self.get_full_model = False
            self.tokens_to_remove = None

    cfg = Config()

    # Load model, tokenizer, and processor
    print(f"Loading model {model_name}...")
    model, tokenizer, processor = load_model(model_path, cfg, device=cfg.device)

    # Attach submodule for hook-based activation extraction.
    # load_model with submodel='enc' already extracts the vision encoder, so
    # model IS the vision encoder (e.g. SiglipVisionTransformer or InternVisionModel).
    # Only set submodule if load_vision_model hasn't already done so (e.g. Qwen2-VL).
    if submodel == 'enc' and not hasattr(model, 'submodule'):
        model.submodule = resolve_attr(model, f"encoder.layers[{layer}]")

    # Move SAE to device and set dtype
    sae.to(args.device)
    sae.to(cfg.dtype)

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Load ImageNet test set
    print("Loading ImageNet test set from HuggingFace...")
    dataset = load_dataset("ILSVRC/imagenet-1k", split="test", trust_remote_code=True)

    # Initialize lists to store all activations for global AUROC
    all_positive_max_activations = []
    all_positive_mean_activations = []
    all_random_max_activations = []
    all_random_mean_activations = []

    # Process each latent
    results = {}

    for latent_id in tqdm(selected_ids):
        # Get the explanation
        explanation = latent_explanations[latent_id]['explanation'][0]
        print(f"\nLatent {latent_id}: {explanation}")

        # Load positive images (generated from explanations)
        positive_images = load_generated_images(base_dir, latent_id, "positive")

        if not positive_images:
            print(f"Skipping latent {latent_id} - missing generated images")
            continue

        # Load random ImageNet images as negative examples
        # Use latent_id as seed for reproducibility
        random_images = load_random_imagenet_images(
            dataset,
            num_images=len(positive_images),
            seed=latent_id
        )

        if not random_images:
            print(f"Skipping latent {latent_id} - could not load random images")
            continue

        print(f"Processing {len(positive_images)} generated images and {len(random_images)} random ImageNet images")

        # Get SAE activations for positive images
        positive_sae_codes = get_sae_activations(
            model, positive_images, tokenizer, processor, sae, cfg
        )

        # Get SAE activations for random images
        random_sae_codes = get_sae_activations(
            model, random_images, tokenizer, processor, sae, cfg
        )

        # Compute statistics for the target latent
        positive_stats = compute_activation_statistics(positive_sae_codes, latent_id)
        random_stats = compute_activation_statistics(random_sae_codes, latent_id)

        # Store individual activations for global AUROC calculation
        all_positive_max_activations.extend(positive_stats[latent_id]["max_values"])
        all_positive_mean_activations.extend(positive_stats[latent_id]["mean_values"])
        all_random_max_activations.extend(random_stats[latent_id]["max_values"])
        all_random_mean_activations.extend(random_stats[latent_id]["mean_values"])

        results[latent_id] = {
            "explanation": explanation,
            "positive_mean_values": positive_stats[latent_id]["mean_values"],
            "positive_max_values": positive_stats[latent_id]["max_values"],
            "random_mean_values": random_stats[latent_id]["mean_values"],
            "random_max_values": random_stats[latent_id]["max_values"],
            "positive_stats_summary": {
                "mean_of_max": positive_stats[latent_id]["mean_of_max"],
                "mean_of_mean": positive_stats[latent_id]["mean_of_mean"],
                "std_of_max": positive_stats[latent_id]["std_of_max"],
                "std_of_mean": positive_stats[latent_id]["std_of_mean"],
                "sparsity": positive_stats[latent_id]["sparsity"],
            },
            "random_stats_summary": {
                "mean_of_max": random_stats[latent_id]["mean_of_max"],
                "mean_of_mean": random_stats[latent_id]["mean_of_mean"],
                "std_of_max": random_stats[latent_id]["std_of_max"],
                "std_of_mean": random_stats[latent_id]["std_of_mean"],
                "sparsity": random_stats[latent_id]["sparsity"],
            },
            "activation_difference_mean_of_max": positive_stats[latent_id]["mean_of_max"] - random_stats[latent_id]["mean_of_max"],
            "activation_difference_mean_of_mean": positive_stats[latent_id]["mean_of_mean"] - random_stats[latent_id]["mean_of_mean"],
        }

        # # Plot per-latent comparison if requested
        # if args.plot:
        #     plot_dir = os.path.join(results_dir, "plots", f"latent_{latent_id}")
        #     plot_activation_comparison(positive_stats, random_stats, latent_id, plot_dir)
        #     # Optionally plot per-latent distributions if needed
        #     # plot_activation_distributions(positive_stats[latent_id]["max_values"], random_stats[latent_id]["max_values"], f"Latent {latent_id} Max", plot_dir)
        #     # plot_activation_distributions(positive_stats[latent_id]["mean_values"], random_stats[latent_id]["mean_values"], f"Latent {latent_id} Mean", plot_dir)


    # --- Global AUROC Calculation ---
    print("\nComputing global AUROC...")
    global_max_auroc, global_max_fpr, global_max_tpr = compute_auroc(
        all_positive_max_activations, all_random_max_activations
    )
    global_mean_auroc, global_mean_fpr, global_mean_tpr = compute_auroc(
        all_positive_mean_activations, all_random_mean_activations
    )
    print(f"Global Max AUROC: {global_max_auroc:.4f}")
    print(f"Global Mean AUROC: {global_mean_auroc:.4f}")

    # --- Save Results ---
    # Save summary results (without individual activations)
    results_file = os.path.join(results_dir, "activation_analysis_summary.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved per-latent summary results to {results_file}")

# Compute overall summary statistics
    activation_differences_mean_of_max = [results[latent_id]["activation_difference_mean_of_max"] for latent_id in results]
    activation_differences_mean_of_mean = [results[latent_id]["activation_difference_mean_of_mean"] for latent_id in results]

    summary = {
        "mean_activation_difference_mean_of_max": np.mean(activation_differences_mean_of_max),
        "median_activation_difference_mean_of_max": np.median(activation_differences_mean_of_max),
        "mean_activation_difference_mean_of_mean": np.mean(activation_differences_mean_of_mean),
        "median_activation_difference_mean_of_mean": np.median(activation_differences_mean_of_mean),
        "num_latents_with_diff_mean_of_max_gt_0": sum(1 for diff in activation_differences_mean_of_max if diff > 0),
        "num_latents_with_diff_mean_of_mean_gt_0": sum(1 for diff in activation_differences_mean_of_mean if diff > 0),
        "global_max_auroc": global_max_auroc,
        "global_mean_auroc": global_mean_auroc,
        "total_latents_evaluated": len(results),
        "total_positive_images": len(all_positive_max_activations),
        "total_random_images": len(all_random_max_activations),
    }

    # Save summary
    summary_file = os.path.join(results_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nEvaluation Summary:")
    print(f"Mean Activation Difference (mean of max): {summary['mean_activation_difference_mean_of_max']:.4f}")
    print(f"Mean Activation Difference (mean of mean): {summary['mean_activation_difference_mean_of_mean']:.4f}")
    print(f"Global AUROC (max): {summary['global_max_auroc']:.4f}")
    print(f"Global AUROC (mean): {summary['global_mean_auroc']:.4f}")
    print(f"Total latents evaluated: {summary['total_latents_evaluated']}")
    print(f"Results saved to {results_dir}")

    # Plot global ROC curves and distributions if requested
    if args.plot:
        plot_dir = os.path.join(results_dir, "plots", "global")
        plot_roc_curve(global_max_fpr, global_max_tpr, global_max_auroc, "Global Max Activations", plot_dir)
        plot_roc_curve(global_mean_fpr, global_mean_tpr, global_mean_auroc, "Global Mean Activations", plot_dir)
        plot_activation_distributions(all_positive_max_activations, all_random_max_activations, "Global Max Activations", plot_dir)
        plot_activation_distributions(all_positive_mean_activations, all_random_mean_activations, "Global Mean Activations", plot_dir)
        print(f"Saved global plots to {plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated images based on dictionary learning explanations")

    # Required arguments
    parser.add_argument("--subject_model", type=str, required=True,
                        help="HuggingFace model ID of the subject model (e.g., 'google/gemma-3-4b-it')")
    parser.add_argument("--layer", type=str, required=True,
                        help="Which layer's features to evaluate ('mid' or 'later')")
    parser.add_argument("--explanation_mode", type=str, required=True,
                        help="Explanation mode (e.g., 'HF_easy2_masks_TOPK-5_MODEL-gemma-3-27b-it')")

    # Optional arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the model on ('cuda' or 'cpu')")
    parser.add_argument("--run_type", type=str, choices=["validation", "test"], default="test",
                        help="Split to evaluate ('validation' or 'test')")
    parser.add_argument("--max_latents_test", type=int, default=None,
                        help="Number of latents to process in test split (defaults to all)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots comparing positive and negative activations")

    args = parser.parse_args()
    main(args)


# Usage
'''
Usage:
python -m evaluation.image_gen_eval \
        --subject_model google/gemma-3-4b-it \
        --layer mid \
        --explanation_mode HF_easy2_masks_TOPK-5_MODEL-gemma-3-27b-it \
        --run_type test \
        --max_latents_test 1
        
'''

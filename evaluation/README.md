# Evaluation

This directory contains three evaluators for assessing SAE (Sparse Autoencoder) feature explanations in vision models.

---

## Overview

| Evaluator | File | Purpose | Key Metric |
|-----------|------|---------|-----------|
| Baseline Simulator | `baseline_simulator.py` | Compares SAE heatmaps with LangSAM segmentation masks | IoU, Precision, Recall |
| CLIP Simulator | `clip_simulator.py` | Measures CLIP alignment between images/heatmaps and concept text | CLIP similarity |
| Image Generation Eval | `image_gen_eval.py` | Checks if concept-generated images activate the expected features | AUROC |

---

## 1. Baseline Simulator

Compares SAE feature activation heatmaps against text-guided segmentation masks produced by LangSAM (Grounding DINO + SAM 2.1). Given a concept explanation for a feature, it segments relevant regions in the top-k images and computes spatial overlap with the heatmap.

**Models required:** SAM 2.1 Hiera Large, Grounding DINO

### Usage

```bash
python evaluation/baseline_simulator.py \
    --explainer_method_name <method_name> \
    --subject_model <hf_model_id> \
    --layer <mid|later> \
    --run_type <validation|test> \
    [--save_images]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--explainer_method_name` | *(required)* | Name of the explanation method to evaluate |
| `--subject_model` | `google/gemma-3-4b-it` | HuggingFace model ID of the subject model |
| `--layer` | `later` | Which layer's features to evaluate (`mid` or `later`) |
| `--run_type` | `validation` | Split to evaluate (`validation` or `test`) |
| `--save_images` | `False` | Save heatmap/segmentation overlay visualizations |

### Example

```bash
python evaluation/baseline_simulator.py \
    --explainer_method_name "steering_google_gemma-3-4b-it_sampling-False_blank_input__COEFF-100" \
    --layer mid \
    --run_type validation \
    --save_images
```

### Output

Results are saved per-latent as `combined_segmentation_results.json` containing IoU, precision, and recall scores. If `--save_images` is set, overlay visualizations are saved alongside.

---

## 2. CLIP Simulator

Evaluates feature explanations by computing CLIP (ViT-Large-Patch14) similarity between the top-k images (optionally masked by heatmap activations) and the text explanation for each feature.

**Models required:** `openai/clip-vit-large-patch14`

### Usage

```bash
python evaluation/clip_simulator.py \
    --explainer_method_name <method_name> \
    --subject_model <hf_model_id> \
    --layer <mid|later> \
    --run_type <validation|test> \
    --number_of_top_k_images <n> \
    [--mask_image]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--explainer_method_name` | *(required)* | Name of the explanation method to evaluate |
| `--subject_model` | `google/gemma-3-4b-it` | HuggingFace model ID of the subject model |
| `--layer` | `mid` | Which layer's features to evaluate (`mid` or `later`) |
| `--run_type` | `validation` | Split to evaluate (`validation` or `test`) |
| `--number_of_top_k_images` | `5` | Number of top-k images to use per latent |
| `--mask_image` | `True` | Mask images by heatmap activation before CLIP scoring |
| `--verbose` | `False` | Enable verbose logging output |

### Output

Per-latent CLIP similarity scores saved as `result.json` files in each latent's directory.

---

## 3. Image Generation Evaluator

Evaluates whether images generated from concept explanations actually activate the corresponding SAE features more strongly than random ImageNet images. Computes per-latent activation statistics and AUROC scores to measure how discriminable the generated images are.

### Prerequisites: Generate Images

Before running the evaluator, you must generate images using `image_generation.py`. This script uses Stable Diffusion to synthesize images from the concept explanations produced by your explainer method.

**Models required:** `stabilityai/stable-diffusion-3.5-medium` (default) or `stabilityai/stable-diffusion-3.5-large`

```bash
python -m evaluation.image_generation \
    --subject_model <hf_model_id> \
    --layer <mid|later> \
    --explainer_method_name <method_name> \
    --run_type <validation|test> \
    [--diffusion_model_id <hf_model_id>] \
    [--num_images <n>] \
    [--batch_size <n>] \
    [--steps <n>] \
    [--guidance_scale <float>] \
    [--seed <int>]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--subject_model` | `google/gemma-3-4b-it` | HuggingFace model ID of the subject model |
| `--layer` | `mid` | Which layer's features to evaluate (`mid` or `later`) |
| `--explainer_method_name` | *(required)* | Name of the explanation method |
| `--run_type` | `validation` | Split to generate images for (`validation` or `test`) |
| `--diffusion_model_id` | `stabilityai/stable-diffusion-3.5-medium` | HuggingFace model ID of the diffusion model |
| `--num_images` | `3` | Number of images to generate per latent |
| `--batch_size` | `16` | Number of latents to process per batch |
| `--steps` | `30` | Number of diffusion inference steps |
| `--guidance_scale` | `7.5` | Classifier-free guidance scale |
| `--seed` | `None` | Random seed for reproducibility |
| `--max_latents_test` | `None` | Number of latents to process in test split (defaults to all) |

Generated images are saved under each latent's subdirectory:
```
{DATA_DIR}/outputs/{sae_id}/{explainer_method_name}/latent_{id}/generated_images/
```

`image_gen_eval.py` reads existing explanations from the same per-latent structure:
```
{DATA_DIR}/outputs/{sae_id}/{explainer_method_name}/latent_{id}/explanations/explanation.json
```

**Example:**

```bash
python -m evaluation.image_generation \
    --subject_model google/gemma-3-4b-it \
    --layer mid \
    --explainer_method_name "steering_google_gemma-3-4b-it_sampling-False_blank_input__COEFF-100" \
    --run_type validation \
    --num_images 3
```

---

### Usage

```bash
python evaluation/image_gen_eval.py \
    --subject_model <hf_model_id> \
    --layer <mid|later> \
    --explanation_mode <explanation_mode> \
    [--run_type <validation|test>] \
    [--device <cuda|cpu>] \
    [--max_latents_test <n>] \
    [--plot]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--subject_model` | *(required)* | HuggingFace model ID of the subject model |
| `--layer` | *(required)* | Which layer's features to evaluate (`mid` or `later`) |
| `--explanation_mode` | *(required)* | Explanation mode name (e.g., `HF_easy2_masks_TOPK-5_MODEL-gemma-3-27b-it`) |
| `--run_type` | `test` | Split to evaluate (`validation` or `test`) |
| `--device` | `cuda` | Device to run inference on (`cuda` or `cpu`) |
| `--max_latents_test` | `None` | Number of latents to process in test split (defaults to all) |
| `--plot` | `False` | Generate ROC curve and activation distribution plots |

### Example

```bash
python evaluation/image_gen_eval.py \
    --subject_model google/gemma-3-4b-it \
    --layer mid \
    --explanation_mode "HF_easy2_masks_TOPK-5_MODEL-gemma-3-27b-it" \
    --run_type test \
    --max_latents_test 10 \
    --plot
```

### Output

A summary JSON with per-latent activation statistics (mean, max, sparsity) and global AUROC scores (`Global Max AUROC`, `Global Mean AUROC`). If `--plot` is set, ROC curves and activation distribution figures are saved.

---

## Shared Utilities

- **`utils.py`** — IoU calculation, image masking, JSON helpers

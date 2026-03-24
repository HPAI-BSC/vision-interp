from pathlib import Path
import json
import logging
import numpy as np
from typing import Dict, List, Optional
from PIL import Image
from PIL import ImageDraw, ImageFont

logger = logging.getLogger(__name__)

def create_output_path(base_dir: Path, subdirs: list[str]=None, filename: str=None) -> Path:
    """Create a full output path with subdirectories, ensuring they exist."""
    full_path = base_dir
    if subdirs:
        for subdir in subdirs:
            full_path /= subdir
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path / filename

def save_json_data(data: dict, output_path: Path) -> None:
    """Save dictionary as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved to {output_path}")

def numpy_to_pil(np_array: np.ndarray) -> Image.Image:
    """Convert NumPy array to PIL Image with normalization."""
    if np_array.dtype != np.uint8:
        np_array = ((np_array - np_array.min()) / 
                   (np_array.max() - np_array.min() + 1e-8) * 255).astype(np.uint8)
    if np_array.ndim == 3 and np_array.shape[0] in [1, 3, 4]:
        np_array = np_array.transpose(1, 2, 0)
    if np_array.shape[-1] == 1:
        np_array = np_array.squeeze(-1)
    return Image.fromarray(np_array)

def convert_numpy_to_pil(np_array: np.ndarray) -> Image.Image:
    """Backward-compatible alias for numpy_to_pil."""
    return numpy_to_pil(np_array)

def create_masked_image(image: np.ndarray, heatmap: np.ndarray) -> Image.Image:
    """Apply heatmap mask to image and return PIL Image."""
    img_height, img_width = image.shape[:2]
    mask = (heatmap > 0).astype(np.uint8) * 255
    mask_resized = Image.fromarray(mask).resize((img_width, img_height), Image.NEAREST)
    mask_array = np.array(mask_resized).astype(bool)
    
    masked_img = image.copy()
    masked_img[~mask_array] = 0
    return numpy_to_pil(masked_img)

def create_heatmap_overlay(image: np.ndarray, heatmap: np.ndarray) -> Image.Image:
    """Apply heatmap overlay to image and return PIL Image."""
    img_height, img_width = image.shape[:2]

    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((img_width, img_height), Image.NEAREST)
    heatmap_array = np.array(heatmap_resized) / 255.0

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
        logger.info(f"Converted grayscale image to RGB, new shape: {image.shape}")

    heatmap_color = np.zeros_like(image, dtype=np.float32)
    heatmap_color[..., 0] = heatmap_array * 255  # Red channel for heatmap
    img_heatmap = (image * 0.5 + heatmap_color * 0.5).clip(0, 255).astype(np.uint8)

    return numpy_to_pil(img_heatmap)


def calculate_iou(mask1, mask2):
    """
    Calculate precision, recall, and IoU between two binary masks.
    
    Parameters:
    mask1 (np.ndarray): "Ground truth" mask
    mask2 (np.ndarray): Prediction mask
    
    Returns:
    dict: Dictionary containing precision, recall, and IoU scores
    """
    # Ensure the masks are binary
    mask1_binary = mask1 > 0
    mask2_binary = mask2 > 0
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1_binary, mask2_binary).sum()
    union = np.logical_or(mask1_binary, mask2_binary).sum()
    
    # Calculate true positives, false positives, and false negatives
    true_positives = intersection
    false_positives = np.logical_and(np.logical_not(mask1_binary), mask2_binary).sum()
    false_negatives = np.logical_and(mask1_binary, np.logical_not(mask2_binary)).sum()
    
    # Calculate precision
    precision = 0.0
    if (true_positives + false_positives) > 0:
        precision = true_positives / (true_positives + false_positives)
    
    # Calculate recall
    recall = 0.0
    if (true_positives + false_negatives) > 0:
        recall = true_positives / (true_positives + false_negatives)
    
    # Calculate IoU
    iou = 0.0
    if union > 0:
        iou = intersection / union
    
    # # Calculate F1 score
    # f1_score = 0.0
    # if (precision + recall) > 0:
    #     f1_score = 2 * (precision * recall) / (precision + recall)
    
    return iou, precision, recall

def save_feature_visualization(
    feature: str,
    masked_images: List[Image.Image],
    explanation: List[str],
    output_dir: Path,
    images_per_row: int = 5,
    image_size: tuple = (224, 224)  # Added parameter for target size
) -> None:
    """Save a visualization for a feature with resized masked images and explanation text."""
    viz_dir = output_dir# / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Resize images to 224x224
    resized_images = [img.resize(image_size, Image.Resampling.LANCZOS) for img in masked_images]
    
    # Calculate dimensions using resized image size
    img_width, img_height = image_size
    num_images = len(resized_images)
    rows = (num_images + images_per_row - 1) // images_per_row
    text_padding = 20
    text_height = 100  # Adjust based on your needs
    
    # Create canvas
    total_width = min(num_images, images_per_row) * img_width
    total_height = (rows * img_height) + text_height + text_padding
    
    canvas = Image.new('RGB', (total_width, total_height), 'white')
    
    # Paste resized images
    for idx, img in enumerate(resized_images):
        row = idx // images_per_row
        col = idx % images_per_row
        canvas.paste(img, (col * img_width, row * img_height))
    
    # Add explanation text
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Wrap text if needed
    text_position = (10, rows * img_height + text_padding)
    max_text_width = total_width - 20
    wrapped_text = []
    #words = explanation.split()
    current_line = ""
    
    for words in explanation:
        for word in words.split():
            test_line = f"{current_line} {word}".strip()
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_text_width:
                current_line = test_line
            else:
                wrapped_text.append(current_line)
                current_line = word
        wrapped_text.append(current_line)
    
    # Draw the text
    y_offset = text_position[1]
    for line in wrapped_text:
        draw.text((text_position[0], y_offset), line, font=font, fill='black')
        y_offset += 20  # Line spacing
    
    # Save the visualization
    output_path = viz_dir / f"{feature}_visualization.png"
    canvas.save(output_path)
    # logger.info(f"Saved visualization for feature {feature} at {output_path}")
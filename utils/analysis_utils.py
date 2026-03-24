import os
from pathlib import Path
import json
import torch
import logging
logger = logging.getLogger(__name__)
import gc
import numpy as np
from PIL import Image
from functools import partial
from typing import Literal
from typing import List
# from transformers import BitsAndBytesConfig

# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
REPO_DIR = Path(__file__).parent.parent

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve


def compute_statistics(results_dict, results_dir=None, plot=False):
    """Compute statistics for the results."""
    positive_mean_values = []
    positive_max_values = []
    random_mean_values = []
    random_max_values = []

    positive_mean_values_per_latent = []
    negative_mean_values_per_latent = []
    for latent_id, results in results_dict.items():
        positive_mean_values.extend(results['positive_mean_values'])
        positive_max_values.extend(results['positive_max_values'])
        random_mean_values.extend(results['random_mean_values'])
        random_max_values.extend(results['random_max_values'])
        positive_mean_values_per_latent.append(results['positive_mean_values'])
        negative_mean_values_per_latent.append(results['random_mean_values'])
    
    # Convert lists to numpy arrays
    positive_mean_values = np.array(positive_mean_values)
    positive_max_values = np.array(positive_max_values)
    random_mean_values = np.array(random_mean_values)
    random_max_values = np.array(random_max_values)

    def compute_auroc(positive_values, negative_values, plot=False):
        """
        Compute AUROC for positive and negative values.
        """
        # Create labels (1 for positive samples, 0 for random/negative samples)
        positive_labels = np.ones(len(positive_values))
        random_labels = np.zeros(len(negative_values))            
        
        # Combine data and labels
        all_mean_values = np.concatenate([positive_values, negative_values])
        all_labels = np.concatenate([positive_labels, random_labels])
        
        # Calculate AUROC
        auroc_mean = roc_auc_score(all_labels, all_mean_values)
        logger.info(f"AUROC: {auroc_mean:.4f}")
        
        # Plot ROC curve if plotting is enabled
        if plot:
            fpr, tpr, _ = roc_curve(all_labels, all_mean_values)
            
            plt.figure(figsize=(4, 4))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {auroc_mean:.4f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            
            # Save the plot
            #roc_plot_path = os.path.join(results_dir, "roc_curve_mean_values.png")
            #plt.savefig(roc_plot_path)
            plt.show()
            plt.close()
            #logger.info(f"ROC curve saved to {roc_plot_path}")
                        
        return auroc_mean

    results_statistics = {}
    # Positive mean
    auroc_mean = compute_auroc(positive_mean_values, random_mean_values, plot=plot)
    positive_mean = np.mean(positive_mean_values)
    positive_median = np.median(positive_mean_values)
    positive_std = np.std(positive_mean_values)
    results_statistics['positive_mean_auroc'] = auroc_mean
    results_statistics['positive_mean_mean'] = positive_mean
    results_statistics['positive_mean_median'] = positive_median
    results_statistics['positive_mean_std'] = positive_std

    # Positive max
    auroc_mean = compute_auroc(positive_max_values, random_max_values, plot=plot)
    positive_mean = np.mean(positive_max_values)
    positive_median = np.median(positive_max_values)
    positive_std = np.std(positive_max_values)
    results_statistics['positive_max_auroc'] = auroc_mean
    results_statistics['positive_max_mean'] = positive_mean
    results_statistics['positive_max_median'] = positive_median
    results_statistics['positive_max_std'] = positive_std

    # Negative mean
    negative_mean = np.mean(random_mean_values)
    negative_median = np.median(random_mean_values)
    negative_std = np.std(random_mean_values)
    results_statistics['negative_mean_mean'] = negative_mean
    results_statistics['negative_mean_median'] = negative_median
    results_statistics['negative_mean_std'] = negative_std

    # Negative max
    negative_mean = np.mean(random_max_values)
    negative_median = np.median(random_max_values)
    negative_std = np.std(random_max_values)
    results_statistics['negative_max_mean'] = negative_mean
    results_statistics['negative_max_median'] = negative_median
    results_statistics['negative_max_std'] = negative_std

    results_statistics['positive_mean_values'] = positive_mean_values_per_latent
    results_statistics['negative_mean_values'] = negative_mean_values_per_latent
    
    if plot:
        plot_activation_distributions(
                positive_mean_values, 
                random_mean_values, 
                "Mean Activation Values", 
                results_dir
            )
    
    #auroc_max = compute_auroc(positive_max_values, random_max_values, plot=plot)

    # print(f"AUROC for mean activation values: {auroc_mean:.4f}")
    # print(f"AUROC for max activation values: {auroc_max:.4f}")
    return results_statistics

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
    plt.figure(figsize=(8, 5))

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

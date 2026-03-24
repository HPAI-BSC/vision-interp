import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
from pathlib import Path
from tqdm import tqdm
import os
from collections import defaultdict
from scipy.stats import mannwhitneyu
import itertools
import hashlib
from utils.analysis_utils import compute_statistics
from utils.utils import get_outputs_path
from config import get_top_k_images_path, N_LATENTS_VAL
from feature_loader import FeatureLoader

# Perform PCA on the embeddings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


metric_to_label = {
        'iou_score': 'IoU Score',
        'precision': 'Precision',
        'recall': 'Recall'
    }

def get_max_coeff_validation(explainer_method_name, subject_model_name="google/gemma-3-4b-it", layer="mid", agg_function="mean"):
    outputs_path_val = get_outputs_path(subject_model_name, layer, "validation")
    steering_path = os.path.join(outputs_path_val, explainer_method_name)
    steering_df_dict = read_segmentation_results(steering_path, iterate=True)
    max_iou_coeff, max_iou = get_coeff_max_iou(steering_df_dict, agg_function)
    return max_iou_coeff, max_iou

def get_paths_starting_with(base_path):
    """
    Get all directory paths that start with the given base_path.
    
    Args:
        base_path: The base path to search from
        
    Returns:
        List of Path objects for directories starting with base_path
    """
    # Convert to Path object if it's a string
    base_path = Path(base_path) if isinstance(base_path, str) else base_path
    
    # Get the parent directory
    parent_dir = base_path.parent
    
    # Get the prefix to match
    prefix = base_path.name
    
    # Find all directories in the parent that start with the prefix
    matching_paths = [
        parent_dir / d for d in os.listdir(parent_dir) 
        if os.path.isdir(os.path.join(parent_dir, d)) and d.startswith(prefix)
    ]
    
    return matching_paths

def read_segmentation_results(base_path, iterate=False, latent_ids=None):
    # Read segmentation results from all latent folders and extract metrics
    
    # Convert to Path object if it's a string
    base_path = Path(base_path)

    if iterate:
        base_paths_list = get_paths_starting_with(base_path)
    else:
        base_paths_list = [base_path]

    print(f'Considering {base_paths_list}')
    
    df_dict = {}
    for base_path in base_paths_list:
        results_data = []
        # Check if the directory exists
        if not base_path.exists():
            print(f"Directory not found: {base_path}")
            return pd.DataFrame()
        
        # List all latent directories
        latent_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("latent_")]
        if latent_ids is not None:
            latent_ids_set = set(latent_ids)
            latent_dirs = [d for d in latent_dirs if int(d.name.split("latent_")[-1]) in latent_ids_set]

        for latent_dir in latent_dirs:
            result_file = latent_dir / "segmentation_results" / "result.json"
            
            if result_file.exists():
                try:
                    with open(result_file, "r") as f:
                        data = json.load(f)
                    
                    latent_id = data.get("latent_id")
                    
                    # Extract metrics from each image result
                    for img_data in data.get("images", []):
                        results_data.append({
                            "latent_id": latent_id,
                            "image_id": img_data.get("image_id"),
                            "concept_name": img_data.get("concept_name"),
                            "iou_score": img_data.get("iou_score"),
                            "precision": img_data.get("precision"),
                            "recall": img_data.get("recall")
                        })
                except Exception as e:
                    print(f"Error reading {result_file}: {e}")
            # else:
            #     print(f"No result file found for {latent_dir}")
        
        # Create DataFrame from collected data
        df = pd.DataFrame(results_data)

        if iterate==False:
            return df
        else:
            coeff = base_path.name.split('_')[-1]
            df_dict[coeff] = df
    
    return df_dict

def read_clip_score_results(base_path, iterate=False, latent_ids=None):
    # Read clip score results from all latent folders and extract metrics
    
    # Convert to Path object if it's a string
    base_path = Path(base_path)

    if iterate:
        base_paths_list = get_paths_starting_with(base_path)
    else:
        base_paths_list = [base_path]
    
    df_dict = {}
    for base_path in base_paths_list:
        results_data = []
        # Check if the directory exists
        if not base_path.exists():
            print(f"Directory not found: {base_path}")
            return pd.DataFrame()
        
        # List all latent directories
        latent_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("latent_")]
        if latent_ids is not None:
            latent_ids_set = set(latent_ids)
            latent_dirs = [d for d in latent_dirs if int(d.name.split("latent_")[-1]) in latent_ids_set]

        for latent_dir in latent_dirs:
            result_file = latent_dir / "clip_scores" / "result.json"
            
            if result_file.exists():
                try:
                    with open(result_file, "r") as f:
                        data = json.load(f)
                    
                    latent_id = data.get("latent_id")

                    cos_sim_scores = data.get("cos_sim_score")
                    results_data.extend(cos_sim_scores)
                except Exception as e:
                    print(f"Error reading {result_file}: {e}")
            # else:
            #     print(f"No result file found for {latent_dir}")
        
        # Create DataFrame from collected data
        df = pd.DataFrame(results_data)

        if iterate==False:
            return df
        else:
            coeff = base_path.name.split('_')[-1]
            df_dict[coeff] = df
    
    return df_dict


all_colors_10 = list(mcolors.TABLEAU_COLORS.values())  # or use CSS4_COLORS for more options
all_colors_big = list(mcolors.CSS4_COLORS.values())

def distribution_comparison_plot_v1(df_dict: dict[str, pd.DataFrame], metric: str, agg_statistics: str = "mean", **kwargs):
    
    # Create a figure to compare IoU score distributions between steering and top-k explanations
    plt.figure(figsize=(8, 4))

    if len(df_dict) <= 10:
        all_colors = all_colors_10
    else:
        all_colors = all_colors_big

    for i, (df_key, df) in enumerate(df_dict.items()):
        # Plot both distributions
        sns.histplot(df[metric], bins=30, kde=True, alpha=0.6, label=df_key, color=all_colors[i])

        if agg_statistics == "mean":
            agg_value = df[metric].mean()
        elif agg_statistics == "median":
            agg_value = df[metric].median()
        else:
            raise ValueError(f"Invalid aggregation statistic: {agg_statistics}")

        # Add vertical lines for means
        plt.axvline(agg_value, linestyle='--', 
                    label=f'{df_key} {agg_statistics.capitalize()}: {agg_value:.3f}', color=all_colors[i])
        

    # Add labels and title
    plt.xlabel(metric_to_label[metric])
    if "coeff" in kwargs:
        plt.title(f'{metric_to_label[metric]} for {kwargs["coeff"]}')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Get the current axis
    ax = plt.gca()
    
    # Add light grid lines
    ax.grid(True, linestyle=(0, (5, 10)))
    ax.set_axisbelow(True)


def distribution_comparison_plot(df_dict: dict[str, pd.DataFrame], metric: str, agg_statistics: str = "mean", **kwargs):
    plt.figure(figsize=(8, 4))

    # Choose colors
    all_colors = all_colors_10 if len(df_dict) <= 10 else all_colors_big

    # Step 1: Compute global min and max
    all_values = pd.concat([df[metric] for df in df_dict.values()])
    min_val, max_val = all_values.min(), all_values.max()

    # Step 2: Create consistent bins
    bins = np.linspace(min_val, max_val, 31)  # 30 bins = 31 edges

    for i, (df_key, df) in enumerate(df_dict.items()):
        # Step 3: Use the same bins
        sns.histplot(df[metric], bins=bins, kde=True, alpha=0.6, label=df_key, color=all_colors[i])

        # Compute the mean or median
        if agg_statistics == "mean":
            agg_value = df[metric].mean()
        elif agg_statistics == "median":
            agg_value = df[metric].median()
        else:
            raise ValueError(f"Invalid aggregation statistic: {agg_statistics}")

        # Add vertical line for the stat
        plt.axvline(agg_value, linestyle='--', 
                    label=f'{df_key} {agg_statistics.capitalize()}: {agg_value:.3f}', color=all_colors[i])

    # Labels and formatting
    plt.xlabel(metric_to_label[metric])
    if "coeff" in kwargs:
        plt.title(f'{metric_to_label[metric]} for {kwargs["coeff"]}')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.grid(True, linestyle=(0, (5, 10)))
    ax.set_axisbelow(True)


def barplot_comparison(df_dict: dict[str, pd.DataFrame], metric: str, agg_statistics: str = "mean", **kwargs):
    """
    Create a bar plot comparing an aggregated statistic across multiple dataframes,
    with error bars showing standard deviation.
    
    Args:
        df_dict: Dictionary mapping names to dataframes
        metric: Column name to aggregate and compare
        agg_statistics: Statistic to compute ('mean' or 'median')
        **kwargs: Additional arguments like 'coeff' for title
    """
    metric_to_label = {
        'iou_score': 'IoU Score',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    
    # Create figure
    plt.figure(figsize=(8, 4))

    if len(df_dict) <= 10:
        all_colors = all_colors_10
    else:
        all_colors = all_colors_big
    
    # Compute aggregated values and standard deviations for each dataframe
    names = []
    values = []
    errors = []  # For storing standard deviations
    
    for name, df in df_dict.items():
        names.append(name)
        if agg_statistics == "mean":
            values.append(df[metric].mean())
            errors.append(df[metric].std())  # Standard deviation
        elif agg_statistics == "median":
            values.append(df[metric].median())
            errors.append(df[metric].std())  # Still using std for error bars with median
        else:
            raise ValueError(f"Invalid aggregation statistic: {agg_statistics}")
    
    # Create bar plot with wider bars (adjust width parameter)
    bar_width = 0.6  # Increased width (default is usually around 0.8)
    bars = plt.bar(names, values, yerr=errors, capsize=5, 
                  color=all_colors[:len(names)], width=bar_width,
                  error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    
    # Add labels and title
    plt.xlabel('Method')
    plt.ylabel(f'{agg_statistics.capitalize()} {metric_to_label.get(metric, metric)}')
    
    if "coeff" in kwargs:
        plt.title(f'{metric_to_label.get(metric, metric)} Comparison for {kwargs["coeff"]}')
    else:
        plt.title(f'{metric_to_label.get(metric, metric)} Comparison')
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    ax = plt.gca()
    ax.set_axisbelow(True)
    
    # Adjust y-axis to start from 0
    plt.ylim(bottom=0)

def boxplot_comparison(df_dict: dict[str, pd.DataFrame], metric: str, **kwargs):
    """
    Create a box plot comparing distributions across multiple dataframes.
    
    Args:
        df_dict: Dictionary mapping names to dataframes
        metric: Column name to compare distributions
        **kwargs: Additional arguments like 'coeff' for title
    """
    metric_to_label = {
        'iou_score': 'IoU Score',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    
    # Create figure
    plt.figure(figsize=(8, 4))

    if len(df_dict) <= 10:
        all_colors = all_colors_10
    else:
        all_colors = all_colors_big
    
    # Prepare data for boxplot
    data = []
    names = []
    
    for name, df in df_dict.items():
        # Extract the data for the specified metric
        data.append(df[metric].values)
        names.append(name)
    
    # Create box plot
    box = plt.boxplot(data, labels=names, patch_artist=True, widths=0.6)
    
    # Customize boxplot colors
    for i, patch in enumerate(box['boxes']):
        patch.set_facecolor(all_colors[i % len(all_colors)])

    # Set median lines to black
    for median in box['medians']:
        median.set(color='black', linewidth=1.5)
    
    # Add labels and title
    plt.xlabel('Method')
    plt.ylabel(f'{metric_to_label.get(metric, metric)}')
    
    if "coeff" in kwargs:
        plt.title(f'{metric_to_label.get(metric, metric)} Distribution for {kwargs["coeff"]}')
    else:
        plt.title(f'{metric_to_label.get(metric, metric)} Distribution')
    
    # Add grid
    plt.grid(True, alpha=0.3, axis='y')
    ax = plt.gca()
    ax.set_axisbelow(True)
    
    # Adjust y-axis to start from 0
    plt.ylim(bottom=0)

def read_image_gen_eval_results(base_path, latent_ids=None):
    activation_analysis_path = os.path.join(base_path, 'image_generation_evaluation_results', 'activation_analysis_summary.json')
    with open(activation_analysis_path, 'r') as f:
        activation_results = json.load(f)
    if latent_ids is not None:
        latent_ids_set = {str(lid) for lid in latent_ids}
        activation_results = {k: v for k, v in activation_results.items() if k in latent_ids_set}
    results = compute_statistics(activation_results, plot=False)
    return results

def get_coeff_max_iou(df_dict, agg_function: str = "mean"):
    """
    Get the coefficient with the highest mean IoU score.
    """
    coeffs = list(df_dict.keys())
    max_iou = 0
    for coeff in coeffs:
        if agg_function == "mean":
            agg_value = df_dict[coeff]['iou_score'].mean()
        elif agg_function == "median":
            agg_value = df_dict[coeff]['iou_score'].median()
        else:
            raise ValueError(f"Invalid aggregation function: {agg_function}")
        if agg_value > max_iou:
            max_iou = agg_value
            max_iou_coeff = coeff
    return max_iou_coeff, max_iou

def get_all_results(outputs_path, explainer_method_name, features_dir=None, run_type=None, max_latents_test=None):
    explanation_scores_path = os.path.join(outputs_path, explainer_method_name)

    # Compute which latent IDs to include, mirroring evaluator split selection
    latent_ids = None
    if features_dir is not None and run_type is not None:
        class _Cfg:
            number_of_top_k_images = None
            top_k_random_sample = False
            top_k_random_sample_seed = 0
        _Cfg.features_dir = features_dir
        all_features = FeatureLoader(_Cfg).list_features()
        if run_type == "validation":
            selected_features = all_features[:N_LATENTS_VAL]
        else:  # test
            end = N_LATENTS_VAL + max_latents_test if max_latents_test is not None else len(all_features)
            selected_features = all_features[N_LATENTS_VAL:end]
        latent_ids = [int(f.split("latent_")[-1]) for f in selected_features]

    try:
        segmentation_results_df = read_segmentation_results(explanation_scores_path, latent_ids=latent_ids)
        mean_iou = segmentation_results_df['iou_score'].mean()
    except:
        segmentation_results_df = None
        mean_iou = None
    try:
        results_image_gen = read_image_gen_eval_results(explanation_scores_path, latent_ids=latent_ids)
    except Exception as e:
        results_image_gen = None
        print(f"Error reading image gen eval results: {e}")
    try:
        results_clip_method = read_clip_score_results(explanation_scores_path, latent_ids=latent_ids)

    except Exception as e:
        results_clip_method = None
        print(f"Error reading clip scores results: {e}")
    
    result = {'segmentation_results_df': segmentation_results_df,
              'mean_iou': mean_iou,
              'gen_img_mean_values': results_image_gen['positive_mean_values'] if results_image_gen is not None else None,
              'gen_img_mean_mean': results_image_gen['positive_mean_mean'] if results_image_gen is not None else None,
              'gen_neg_img_mean_values': results_image_gen['negative_mean_values'] if results_image_gen is not None else None,
              'gen_img_mean_auroc': results_image_gen['positive_mean_auroc'] if results_image_gen is not None else None,
              'clip_values_df': results_clip_method,
              'clip_mean': results_clip_method.mean().item() if results_clip_method is not None else None}
    if all(v is None for v in result.values()):
        return None
    return result
            


def get_len_explanations(df_dict):
    # Check length of explanations
    len_dict = {}
    for explanation_type in df_dict.keys():
        explanations = df_dict[explanation_type]['explanation'].values
        len_dict[explanation_type] = 0
        counter = 0
        for exp in explanations:
            len_exp = len(exp.split(' '))
            len_dict[explanation_type] += len_exp
            counter += 1
        len_dict[explanation_type] = len_dict[explanation_type] / counter
    
    return len_dict


def plot_label_pie_charts_from_df(df, methods, label_type, title_prefix="", legend_title="Labels", figsize=(5, 5), ncol_legend=4):
    """
    Plots pie charts for label columns in a single DataFrame using consistent coloring and a shared legend.
    """
    # Step 1: Collect all labels across selected columns
    all_labels = sorted(set().union(*[
        df[f'{label_type}_{method}'].dropna().unique() for method in methods
    ]))
    color_map = dict(zip(all_labels, plt.cm.tab20.colors[:len(all_labels)]))

    # Step 2: Create subplots
    num_charts = len(methods)
    fig, axes = plt.subplots(1, num_charts, figsize=(figsize[0] * num_charts, figsize[1]), squeeze=False)
    axes = axes[0]

    for ax, method in zip(axes, methods):
        col_name = f'{label_type}_{method}'
        counts = df[col_name].value_counts()
        colors = [color_map[label] for label in counts.index]
        wedges, texts, autotexts = ax.pie(
            counts,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops=dict(edgecolor='black'),
            labels=None,
            pctdistance=0.85
        )
        for autotext in autotexts:
            autotext.set_fontsize(8)
        ax.set_title(f"{title_prefix}{method}", fontsize=12)

    # Step 3: Shared legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color_map[label], markersize=10, markeredgecolor='black')
               for label in all_labels]
    fig.legend(
        handles=handles,
        title=legend_title,
        loc='lower center',
        ncol=ncol_legend,
    )

    plt.tight_layout()
    plt.show()


# Apply clustering to identify meaningful groups
def cluster_embeddings(embeddings, n_clusters=None, method="kmeans", min_cluster_size=5):
    """
    Apply clustering to embeddings to identify meaningful groups.
    
    Args:
        embeddings (numpy.ndarray): Original high-dimensional embeddings
        n_clusters (int, optional): Number of clusters for KMeans (if None, attempts to estimate)
        method (str): Clustering method ('kmeans', 'dbscan', 'hdbscan', or 'agglomerative')
        min_cluster_size (int): Minimum number of samples per cluster (for HDBSCAN)
        
    Returns:
        numpy.ndarray: Cluster labels for each sample
    """
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    
    # Try to estimate a good number of clusters if not provided
    if n_clusters is None and method == "kmeans":
        # Simple estimation based on data size
        n_clusters = max(3, min(20, int(np.sqrt(embeddings.shape[0] / 2))))
        print(f"Estimated number of clusters: {n_clusters}")
    
    # Apply the selected clustering method
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(embeddings)
        
        # Evaluate clustering quality
        if len(np.unique(labels)) > 1:
            score = silhouette_score(embeddings, labels)
            print(f"Silhouette score: {score:.4f}")
            
    elif method == "dbscan":
        # DBSCAN automatically determines the number of clusters
        from sklearn.neighbors import NearestNeighbors
        
        # Try to estimate a good eps value based on nearest neighbors
        k = min(20, embeddings.shape[0] - 1)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        knee_point = np.sort(distances[:, k-1])[int(0.95 * len(distances))]
        
        model = DBSCAN(eps=knee_point, min_samples=min_cluster_size)
        labels = model.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"DBSCAN found {n_clusters} clusters with {np.sum(labels == -1)} noise points")
        
    elif method == "hdbscan":
        try:
            import hdbscan
            model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                   min_samples=1,
                                   metric='euclidean',
                                   prediction_data=True)
            labels = model.fit_predict(embeddings)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"HDBSCAN found {n_clusters} clusters with {np.sum(labels == -1)} noise points")
        except ImportError:
            print("HDBSCAN not installed. Please install with 'pip install hdbscan'")
            print("Falling back to KMeans...")
            model = KMeans(n_clusters=n_clusters or 5, random_state=42, n_init=10)
            labels = model.fit_predict(embeddings)
            
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters or 5, linkage='ward')
        labels = model.fit_predict(embeddings)
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")
            
    return labels

# Find representative examples for each cluster
def get_cluster_representatives(embeddings, cluster_labels, texts, n_examples=3):
    """
    Find the most representative examples for each cluster.
    
    Args:
        embeddings (numpy.ndarray): The original embeddings
        cluster_labels (numpy.ndarray): Cluster assignments
        texts (list): The corresponding text for each embedding
        n_examples (int): Number of representative examples to find per cluster
        
    Returns:
        dict: Mapping of cluster IDs to representative examples
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    cluster_representatives = {}
    
    # Find unique clusters
    unique_clusters = np.unique(cluster_labels)
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise points in DBSCAN/HDBSCAN
            continue
            
        # Get indices of samples in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        # Get the embeddings for this cluster
        cluster_embeddings = embeddings[cluster_indices]
        
        # Compute the centroid (mean) of the cluster
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Find the samples closest to the centroid
        similarities = cosine_similarity([centroid], cluster_embeddings)[0]
        closest_indices = np.argsort(-similarities)[:n_examples]
        
        # Store the representative examples
        representatives = [texts[cluster_indices[i]] for i in closest_indices]
        cluster_representatives[cluster_id] = representatives
    
    return cluster_representatives

def compute_pca(embeddings, n_components=2):
    """
    Compute PCA dimensionality reduction on embeddings.
    
    Args:
        embeddings (numpy.ndarray): The embeddings to reduce
        n_components (int): Number of components to keep
        
    Returns:
        tuple: (reduced embeddings, PCA object with model information)
    """
    
    # Initialize PCA
    pca = PCA(n_components=n_components)
    
    # Fit and transform the embeddings
    reduced_embeddings = pca.fit_transform(embeddings)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    return reduced_embeddings, pca

def compute_tsne(embeddings, n_components=2, perplexity=30, n_iter=1000, random_state=42):
    """
    Compute t-SNE dimensionality reduction on embeddings.
    
    Args:
        embeddings (numpy.ndarray): The embeddings to reduce
        n_components (int): Number of components to keep
        perplexity (float): The perplexity parameter for t-SNE
        n_iter (int): Number of iterations for optimization
        random_state (int): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: The reduced embeddings
    """
    
    # Initialize t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state
    )
    
    # Fit and transform the embeddings
    print(f"Computing t-SNE with perplexity={perplexity}, n_iter={n_iter}...")
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    return reduced_embeddings

def compute_umap(embeddings, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42):
    """
    Compute UMAP dimensionality reduction on embeddings.
    
    Args:
        embeddings (numpy.ndarray): The embeddings to reduce
        n_components (int): Number of components to keep
        n_neighbors (int): The size of local neighborhood (in terms of number of neighboring sample points)
        min_dist (float): Minimum distance between points in the low-dimensional space
        metric (str): The distance metric to use
        random_state (int): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: The reduced embeddings
    """
    import umap.umap_ as umap
    
    # Initialize UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    
    # Fit and transform the embeddings
    print(f"Computing UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    return reduced_embeddings


def plot_embeddings(reduced_embeddings, latent_ids=None, labels=None, hover_texts=None, method="t-SNE", colorscale="Viridis", 
                   title=None, point_size=5, opacity=0.7, width=1000, height=800):
    """
    Create an interactive scatter plot of reduced embeddings using Plotly.
    
    Args:
        reduced_embeddings (numpy.ndarray): The reduced embeddings (2D or 3D)
        labels (list or numpy.ndarray, optional): Labels or cluster assignments for coloring points
        hover_texts (list, optional): Texts to display when hovering over points
        method (str): The dimensionality reduction method used (for title)
        colorscale (str): Plotly colorscale to use
        title (str, optional): Custom title for the plot
        point_size (int): Size of the scatter points
        opacity (float): Opacity of the scatter points (0-1)
        width (int): Width of the figure in pixels
        height (int): Height of the figure in pixels
        
    Returns:
        plotly.graph_objects.Figure: The interactive plot
    """
    import plotly.graph_objects as go
    import plotly.express as px
    
    # Check dimensions of reduced embeddings
    if reduced_embeddings.shape[1] not in [2, 3]:
        raise ValueError("Reduced embeddings must be 2D or 3D (shape[1] must be 2 or 3)")
    
    # Set default title if not provided
    if title is None:
        title = f"{method} Visualization of Embeddings"
    
    # Create a DataFrame for easier plotting with hover data
    plot_df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'id': latent_ids  # Add latent ID (index number)
    })
    
    # Add labels if provided
    if labels is not None:
        # Convert labels to strings to ensure they're treated as categorical
        if isinstance(labels, (list, np.ndarray)):
            # Convert numeric labels to strings
            if np.issubdtype(np.array(labels).dtype, np.number):
                labels = [f"Cluster {label}" for label in labels]
            else:
                # Ensure all labels are strings
                labels = [str(label) for label in labels]
        plot_df['label'] = labels
    
    # Add hover texts if provided
    if hover_texts is not None:
        plot_df['explanation'] = hover_texts
    
    # Create the appropriate plot based on dimensions
    if reduced_embeddings.shape[1] == 2:
        if labels is not None:
            # Use color for labels
            fig = px.scatter(
                plot_df,
                x='x', 
                y='y',
                color='label',
                hover_data=['id','explanation'] if hover_texts is not None else None,
                color_continuous_scale=colorscale if isinstance(labels[0], (int, float, np.number)) else None,
                title=title,
                opacity=opacity,
                width=width,
                height=height
            )
        else:
            # No labels, use a single color
            if hover_texts is not None:
                fig = px.scatter(
                    plot_df,
                    x='x',
                    y='y',
                    hover_data=['id','explanation'],
                    opacity=opacity,
                    width=width,
                    height=height
                )
            else:
                fig = go.Figure(
                    data=[go.Scatter(
                        x=reduced_embeddings[:, 0],
                        y=reduced_embeddings[:, 1],
                        mode='markers',
                        marker=dict(
                            size=point_size,
                            opacity=opacity
                        )
                    )]
                )
            fig.update_layout(
                title=title,
                width=width,
                height=height
            )
    else:  # 3D case
        if labels is not None:
            plot_df['z'] = reduced_embeddings[:, 2]
            fig = px.scatter_3d(
                plot_df,
                x='x',
                y='y',
                z='z',
                color='label',
                hover_data=['id','explanation'] if hover_texts is not None else None,
                color_continuous_scale=colorscale if isinstance(labels[0], (int, float, np.number)) else None,
                opacity=opacity,
                width=width,
                height=height
            )
        else:
            if hover_texts is not None:
                plot_df['z'] = reduced_embeddings[:, 2]
                fig = px.scatter_3d(
                    plot_df,
                    x='x',
                    y='y',
                    z='z',
                    hover_data=['id','explanation'],
                    opacity=opacity,
                    width=width,
                    height=height
                )
            else:
                fig = go.Figure(
                    data=[go.Scatter3d(
                        x=reduced_embeddings[:, 0],
                        y=reduced_embeddings[:, 1],
                        z=reduced_embeddings[:, 2],
                        mode='markers',
                        marker=dict(
                            size=point_size,
                            opacity=opacity
                        )
                    )]
                )
            fig.update_layout(
                title=title,
                width=width,
                height=height
            )
    
    # Improve layout
    fig.update_layout(
        template="plotly_white",
        title={
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    return fig




def get_explanations(results_experiment_method):
    segmentation_results_df = results_experiment_method['segmentation_results_df']
    latent_idx_to_id = {}
    explanations = []
    for i, latent_id in enumerate(sorted(segmentation_results_df['latent_id'].unique())):
        latent_idx_to_id[i] = latent_id
        explanation = segmentation_results_df[segmentation_results_df['latent_id'] == latent_id]['concept_name'].iloc[0]
        # explanations are ordered
        explanations.append(explanation)
    return explanations, latent_idx_to_id


def get_explanation_labels(explanations_path, label_type='abstraction_level'):
    # Read the abstract level labels
    labels_path = os.path.join(explanations_path, f"labels_{label_type}.json")
    with open(labels_path, "r") as f:
        labels = json.load(f)
        
    labels_latent_ids = {}

    if label_type == 'abstraction_level':

        for latent_id, content in labels.items():
            label = content['label'][0]
            if 'low-level' in label:
                labels_latent_ids[latent_id] = 'low-level'
            elif 'high-level' in label:
                labels_latent_ids[latent_id] = 'high-level'
            else:
                labels_latent_ids[latent_id] = 'other'
        return labels_latent_ids
    
    elif label_type == 'background_object':
        for latent_id, content in labels.items():
            label = content['label'][0]
            if 'animal' in label:
                labels_latent_ids[latent_id] = 'animal'
            elif 'background' in label:
                labels_latent_ids[latent_id] = 'background'
            else:
                labels_latent_ids[latent_id] = 'other'
        return labels_latent_ids
    
from sklearn.metrics import roc_auc_score

def compute_auroc(df):
    act_values = df['synth_act_indiv_scores']
    neg_act_values = df['synth_neg_act_indiv_scores']
    pos_act_values = np.concatenate([x for x in act_values if isinstance(x, list)])
    neg_act_values = np.concatenate([x for x in neg_act_values if isinstance(x, list)])


    # Create labels (1 for positive samples, 0 for random/negative samples)
    positive_labels = np.ones(len(pos_act_values))
    negative_labels = np.zeros(len(neg_act_values))            

    # Combine data and labels
    all_mean_values = np.concatenate([pos_act_values, neg_act_values])
    all_labels = np.concatenate([positive_labels, negative_labels])

    auroc_mean = roc_auc_score(all_labels, all_mean_values)
    return auroc_mean


def load_full_results(subject_model_name, explainer_model_name, run_type, max_coeff_steering, max_coeff_steered_topk, overlay_type="masks", layer='mid', max_latents_test=None):
    
    # Default explainer method names
    steering_explainer_method_name = f"steering2_{explainer_model_name.replace('/', '_')}_sampling-False_blank_input_"
    top_k_explainer_method_name = f"HF_easy2_{overlay_type}_TOPK-5_MODEL-{explainer_model_name.split('/')[-1]}"
    steered_top_k_explainer_method_name = f"easy2_w_steering_{overlay_type}_{explainer_model_name.replace('/', '_')}_sampling-False_top_k_images_input_"

    # Get path where the features explanation-related files are stored
    outputs_path = get_outputs_path(subject_model_name, layer, run_type)
    results_experiment = {}
    features_dir = get_top_k_images_path(subject_model_name, layer)
    shared_kwargs = dict(features_dir=features_dir, run_type=run_type, max_latents_test=max_latents_test)

    # Raw steering
    steering_explainer_method_name_explainer_method_name = f"{steering_explainer_method_name}{max_coeff_steering}"
    print(f"Explainer method name: {steering_explainer_method_name_explainer_method_name}")
    results_experiment['steering'] = get_all_results(outputs_path, steering_explainer_method_name_explainer_method_name, **shared_kwargs)

    # Top-k
    print(f"Explainer method name: {top_k_explainer_method_name}")
    results_experiment['top_k'] = get_all_results(outputs_path, top_k_explainer_method_name, **shared_kwargs)


    # Steering-informed top-k
    steered_top_k_explainer_method_name = f"{steered_top_k_explainer_method_name}{max_coeff_steered_topk}"
    print(f"Explainer method name: {steered_top_k_explainer_method_name}")
    results_experiment['steered_top_k'] = get_all_results(outputs_path, steered_top_k_explainer_method_name, **shared_kwargs)
        
    # Get background/abstraction labels for all explainer methods
    explainer_methods_keys = ['steering', 'top_k', 'steered_top_k']
    explainer_methods_paths = [steering_explainer_method_name_explainer_method_name, top_k_explainer_method_name, steered_top_k_explainer_method_name]
    for label_type in ['abstraction_level', 'background_object']:
        for explainer_method, explainer_method_path in zip(explainer_methods_keys, explainer_methods_paths):
            labels_path = os.path.join(outputs_path, explainer_method_path, f"labels_{label_type}.json")
            if os.path.exists(labels_path):
                results_experiment[explainer_method][label_type] = get_explanation_labels(os.path.join(outputs_path, explainer_method_path),
                                                                                        label_type=label_type)
    
    return results_experiment


def group_results_in_df(results_experiment_method):
    """
    Group the results in a dataframe.
    """
    num_img_clip = 5
    segmentation_results_df = results_experiment_method['segmentation_results_df']
    # Explanations
    explanations, latent_idx_to_id = get_explanations(results_experiment_method)
    grouped_df = segmentation_results_df.groupby('latent_id')['iou_score'].mean().reset_index()
    grouped_df['explanation'] = explanations
    # Synth Act Scores
    total_rows = len(grouped_df)
    positive_values = results_experiment_method['gen_img_mean_values']
    n_valid = len(positive_values)
    padded_values = positive_values + [np.nan] * (total_rows - n_valid)
    grouped_df['synth_act_indiv_scores'] = padded_values

    negative_values = results_experiment_method['gen_neg_img_mean_values']
    n_valid = len(negative_values)
    padded_values = negative_values + [np.nan] * (total_rows - n_valid)
    grouped_df['synth_neg_act_indiv_scores'] = padded_values


    padded_array = np.full(total_rows, np.nan)
    mean_synth_img_act = np.array(results_experiment_method['gen_img_mean_values']).mean(-1)
    padded_array[:len(mean_synth_img_act)] = mean_synth_img_act
    grouped_df['synth_act_score'] = padded_array
    # CLIP Scores
    clip_scores = results_experiment_method['clip_values_df'].groupby(results_experiment_method['clip_values_df'].index // num_img_clip).mean().reset_index(drop=True)
    grouped_df['clip_score'] = clip_scores
    if 'abstraction_level' in results_experiment_method:
        grouped_df['abstraction_level'] = list(results_experiment_method['abstraction_level'].values())
    if 'background_object' in results_experiment_method:
        grouped_df['background_object'] = list(results_experiment_method['background_object'].values())


    return grouped_df, latent_idx_to_id


def merge_all_dfs(df_dict, method_a, method_b, method_c):

    # Merge method_a and method_b
    dual_df = df_dict[method_a].merge(
        df_dict[method_b], on='latent_id', suffixes=(f'_{method_a}', f'_{method_b}')
    )

    # Merge the result with method_c — we manually rename overlapping columns before merging to avoid _x/_y
    df_c = df_dict[method_c].copy()
    df_c = df_c.rename(columns={col: f"{col}_{method_c}" for col in df_c.columns if col != 'latent_id'})

    # Final merge
    dual_df = dual_df.merge(df_c, on='latent_id')

    return dual_df


def get_comparison_df(merged_df, method_a, method_b):
    metrics = ['synth_act_score', 'iou_score', 'clip_score']
    dual_df = merged_df.copy()

    for metric in metrics:
        dual_df[f'{metric}_diff'] = merged_df[f'{metric}_{method_a}'] - merged_df[f'{metric}_{method_b}']

    return dual_df


def perform_pairwise_tests(merged_df, metric, p_value_threshold=0.05):
    """
    Perform pairwise tests between all combinations (excluding self-comparisons)
    Returns a dictionary with the results of the tests.
    """
    methods_list = ['top_k', 'steered_top_k', 'steering']
    # Dictionary to store results
    results = {}
    # Perform pairwise tests between all combinations (excluding self-comparisons)
    for method_a, method_b in itertools.combinations(methods_list, 2):
        
        values_a = merged_df[f"{metric}_{method_a}"].dropna()
        values_b = merged_df[f"{metric}_{method_b}"].dropna()

        def test_significance_pair(values_a, values_b, method_a, method_b):
            # Alternative hypothesis: Underlying distrib. values_a > values_b
            stat, p_value = mannwhitneyu(values_a, values_b, alternative='greater')

            better_method = None
            significant = False
            if p_value < p_value_threshold:
                # We reject the null hypothesis, so (alternative hypothesis) values_a > values_b
                better_method = method_a
                significant = True
            return better_method, p_value, significant
        
        better_method, p_value, significant = test_significance_pair(values_a, values_b, method_a, method_b)
        
        results[f"{method_a}_vs_{method_b}"] = {
            'p_value': p_value,
            'significant': significant,
            'better_method': better_method
        }

        better_method, p_value, significant = test_significance_pair(values_b, values_a, method_b, method_a)
        
        results[f"{method_b}_vs_{method_a}"] = {
            'p_value': p_value,
            'significant': significant,
            'better_method': better_method
        }

    counter_method_dict = defaultdict(int)

    for result in results.items():
        if result[1]['significant'] == True:
            counter_method_dict[result[1]['better_method']] += 1
        else:
            counter_method_dict[result[1]['better_method']] += 1

    some_method_wins = False
    for method in counter_method_dict.keys():
        if method is not None:
            if counter_method_dict[method] >= 2:
                print(f"For {metric}, {method} is significantly better, {counter_method_dict[method]} times")
                some_method_wins = True
    if some_method_wins == False:
        print(f"For {metric}, no method wins")
    return counter_method_dict


def assign_best_explanations_by_rank(merged_df, metrics, random_tie_break=True, global_seed=12):
    """
    Assigns the best explanation per row by ranking multiple metrics across methods and selecting
    the method with the lowest average rank.

    Parameters:
        merged_df (pd.DataFrame): DataFrame with explanation and metric columns.
        metrics (list of str): Metrics to rank explanations by (e.g., ['iou_score', 'clip_score']).
        random_tie_break (bool): Whether to break ties randomly.

    Returns:
        pd.DataFrame: Modified DataFrame with best explanation, scores, and explanation type.
    """
    methods = ['steering', 'top_k', 'steered_top_k']
    tie_break_order = ['steered_top_k', 'top_k', 'steering']
    
    # Initialize output columns
    merged_df['best_explanation'] = ''
    merged_df['best_explanation_type'] = ''
    merged_df['best_iou_score'] = None
    merged_df['best_clip_score'] = None
    merged_df['best_synth_act_score'] = None

    for idx, row in merged_df.iterrows():
        rank_sums = {method: 0 for method in methods}
        valid_metric_counts = {method: 0 for method in methods}

        for metric in metrics:
            try:
                scores = {method: row[f"{metric}_{method}"] for method in methods}
            except KeyError:
                continue  # Skip if any column is missing

            if any(pd.isna(score) for score in scores.values()):
                continue  # Skip this metric for this row

            # Rank methods (1 = best)
            sorted_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (method, _) in enumerate(sorted_methods, 1):
                rank_sums[method] += rank
                valid_metric_counts[method] += 1

        if all(count == 0 for count in valid_metric_counts.values()):
            continue  # Skip row if no valid metrics

        # Compute average ranks
        avg_ranks = {
            method: (rank_sums[method] / valid_metric_counts[method]) if valid_metric_counts[method] > 0 else float('inf')
            for method in methods
        }

        # Get minimum rank value
        min_rank = min(avg_ranks.values())
        candidates = [m for m in methods if avg_ranks[m] == min_rank]

        assert len(candidates) > 0, f"No candidates found for row {idx}"

        if len(candidates) == 1:
            best_method = candidates[0]
            tie = False
        else:
            tie = True
            if random_tie_break:
                # Create a deterministic per-row seed using a hash of row index and global seed
                row_string = str(idx) + str(global_seed)
                row_hash = int(hashlib.md5(row_string.encode()).hexdigest(), 16) % (2**32)
                rng = np.random.default_rng(row_hash)
                best_method = rng.choice(candidates)
            else:
                # Resolve tie using fixed preference
                for preferred in tie_break_order:
                    if preferred in candidates:
                        best_method = preferred
                        break

        # Assign values
        merged_df.at[idx, 'best_explanation'] = row.get(f'explanation_{best_method}', '')
        merged_df.at[idx, 'best_explanation_type'] = best_method
        merged_df.at[idx, 'best_iou_score'] = row.get(f'iou_score_{best_method}', None)
        merged_df.at[idx, 'best_clip_score'] = row.get(f'clip_score_{best_method}', None)
        merged_df.at[idx, 'best_synth_act_score'] = row.get(f'synth_act_score_{best_method}', None)
        merged_df.at[idx, 'tie'] = tie

    # Enforce numeric dtypes
    merged_df['best_iou_score'] = pd.to_numeric(merged_df['best_iou_score'], errors='coerce')
    merged_df['best_clip_score'] = pd.to_numeric(merged_df['best_clip_score'], errors='coerce')
    merged_df['best_synth_act_score'] = pd.to_numeric(merged_df['best_synth_act_score'], errors='coerce')

    return merged_df

def compute_most_similar_explanations(embeddings_np_dict, include_steered_top_k= True, threshold=0.8):

    # Normalize each row to unit length (for cosine similarity)
    def normalize_rows(arr):
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)
    
    array1 = embeddings_np_dict['top_k']
    array1 = normalize_rows(array1)
    array2 = embeddings_np_dict['steering']
    array2 = normalize_rows(array2)
    # Compute cosine similarity per row across arrays
    sim12 = np.sum(array1 * array2, axis=1)

    if include_steered_top_k:
        array3 = embeddings_np_dict['steered_top_k']
        array3 = normalize_rows(array3)
        sim13 = np.sum(array1 * array3, axis=1)
        sim23 = np.sum(array2 * array3, axis=1)

        # Aggregate similarity (e.g., mean)
        mean_sim = (sim12 + sim13 + sim23) / 3.0
    
    else:
        mean_sim = sim12

    top_indices = np.where(mean_sim > threshold)[0]

    return top_indices

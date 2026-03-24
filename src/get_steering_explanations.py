# %%
import sys
import os
from pathlib import Path
from dotenv import dotenv_values
from dotenv import load_dotenv
load_dotenv()

REPO_DIR = os.environ.get('REPO_DIR')

# Check if REPO_DIR environment variable is set
if REPO_DIR is None:
    REPO_DIR = str(Path(__file__).resolve().parent.parent)
    os.environ['REPO_DIR'] = REPO_DIR
sys.path.append(REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, 'src'))

from datetime import datetime
import torch
from torch import Tensor
from typing import List, Tuple, Callable, Union
from dataclasses import dataclass
import argparse
import json
import re
from tqdm import tqdm
 
from utils.utils import load_model, resolve_attr
from dictionary_learning.utils import load_dictionary, load_dataset_from_yaml
from dictionary_learning.buffer import tokenized_batch
from datasets import load_from_disk
import demo_config
from demo_config import get_activation_dim
from PIL import Image
import numpy as np
from utils.hf_hook_utils import get_activation_addition_output_post_hook, get_activation_addition_output_post_hook_v2
from transformers import GenerationConfig

from utils.hf_hook_utils import add_hooks
from utils.hf_hook_utils import generate_completions
from utils.utils import save_json_data, create_output_path, load_vision_model
from utils.utils import get_paths
from utils.utils import list_features, get_or_create_attribution_images
from utils.hf_hook_utils import generate_completions_internvl
from src.steering_utils import read_top_k_images
from src.config import N_LATENTS_VAL

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
# Set logging level to ERROR to suppress WARNING messages
import logging
logging.getLogger().setLevel(logging.ERROR)

DATA_DIR = os.environ.get('DATA_DIR')
# TODO: change by reading from env vars
SAES_DIR = f"{DATA_DIR}/saes"
DATASET_NAME = "ILSVRC/imagenet-1k"
DATASET_SPLIT = "test"
TOP_K_IMAGES = 5
TEMPERATURES = {
    'google/gemma-3-4b-it': 0.7,
    'google/gemma-3-12b-it': 0.7,
    'google/gemma-3-27b-it': 0.7,
    'OpenGVLab/InternVL3-14B': 0.5
}

# Temporal!!
kwargs_top_k_images = {'top_k': TOP_K_IMAGES}


# List of strings to replace in the prompt completions
to_replace_list = ["The highlighted element in the image is a ",
                    "Here's a description of the highlighted element in the image:",
                    "Here's a description of the highlighted element across all the images:",
                    "Here's a description of the highlighted element across all images:",
                    "*",
                    "Summary:"
                    ]

def summarize_explanations(model, processor, data_batch, cfg) -> str:

    input_len = data_batch["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**data_batch, max_new_tokens=cfg.max_new_tokens, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)
    return [decoded]

def get_top_k_images_path(subject_model_name, layer, subset="test", dataset_size=50000):
    # Given a subject model name and a layer, return the path to the top k images
    top_k_images_path = get_paths(subject_model_name, layer, path_to_get="top_k_images_id")
    top_k_images_path = top_k_images_path.replace('test', subset)
    top_k_images_path = top_k_images_path.replace('50000', str(dataset_size))
    return top_k_images_path

def get_baseline_image_steering(prompt, steering_type, base_image_type, tokenizer, processor, cfg, batch_size):
    assert steering_type in ['raw', 'steering_aided_top_k', 'none']
    
    # Default prompt for the baseline inputs
    # Read image from the specified path
    # image_path = f"{DATA_DIR}/images/blank.jpg"
    # if os.path.exists(image_path):
    #     baseline_image = Image.open(image_path)
    # else:
    #     print(f"Warning: Image file not found at {image_path}")
    # INSERT_YOUR_CODE
    # Create a white image as the baseline_image (e.g., 224x224 RGB)
    baseline_image = Image.new("RGB", (224, 224), color="white")

    raw_inputs = [{'text': [prompt], 'image': [baseline_image]}]*batch_size
    data_batch = tokenized_batch(raw_inputs, tokenizer, cfg, processor)
    
    if base_image_type == 'random':
        data_batch['pixel_values'] = torch.randn_like(data_batch['pixel_values'])
    elif base_image_type == 'black':
        data_batch['pixel_values'] *= 0.0001
    # if base_image_type == 'none' return it as is
    return data_batch, None

def load_prompt(model_name, steering_type, base_image_type, mask_flag=True, heatmaps_flag=False):
    if 'paligemma' in model_name.lower():
        prompt = "Describe the image in great detail."
    else:
        if steering_type == 'raw':
            if base_image_type == 'blank' or base_image_type == 'none':
                prompt = 'You are given an image highlighting a visual or semantic element. This element may range from a low-level visual feature to a high-level abstract concept. Your task is to describe this element in a single, clear sentence. If the element is a high-level abstract concept, describe it as such; otherwise, describe its visual patterns. Favor a more general interpretation. Start the highlighted element description with \"The highlighted element in the image is a\"'
                # prompt = 'Describe the image.'

            elif base_image_type == 'black':
                prompt = "You are given a black image where a specific visual/semantic element has been highlighted. Your task is to provide a unique concise explanation of 3-5 words of the highlighted element. Do not get distracted by the black background. Answer directly with the explanation."
        elif steering_type == 'steering_aided_top_k' or steering_type == 'none':
            assert mask_flag or heatmaps_flag, "mask_flag or heatmaps_flag must be True"
            if mask_flag:
                prompt = "You are given set of images highlighting a visual or semantic element. The patches of the images not showing the element are masked out, giving the impression of a pixelated image. This element may range from a low-level visual feature to a high-level abstract concept. Your task is to describe this element in a single, clear sentence. If the element is a high-level abstract concept, describe it as such; otherwise, describe its visual patterns. Favor a more general interpretation. Provide a single description for the highlighted element appearing in all images, and please ignore the pixelated effect of the mask when describing the element. Start the highlighted element description with \"The highlighted element in the image is a\"."
            elif heatmaps_flag:
                prompt = "You are given set of images highlighting a visual or semantic element. The patches of the images showing the element are highlighted with a green heatmap. This element may range from a low-level visual feature to a high-level abstract concept. Your task is to describe this element in a single, clear sentence. If the element is a high-level abstract concept, describe it as such; otherwise, describe its visual patterns. Favor a more general interpretation. Provide a single description for the highlighted element appearing in all images, and please ignore the overlayed green heatmap when describing the element. Start the highlighted element description with \"The highlighted element in the image is a\"."
    return prompt

@dataclass
class ExperimentConfig:
    engine: str = 'HF'
    model_name: str = 'google/gemma-3-4b-it'
    lm_model_name: str = 'google/gemma-3-4b-it'
    dataset: str = "ILSVRC/imagenet-1k"
    dataset_type: str = 'raw'
    device: str = 'cuda:0'
    model_type: str = demo_config.LLM_CONFIG[model_name].model_type
    model_path: str = demo_config.LLM_CONFIG[model_name].model_path
    submodel: str = 'enc'
    site: str = 'res'
    io: str = 'out'
    activation_dim: int = get_activation_dim(model_name, submodel)
    dtype: torch.dtype = torch.bfloat16
    tokens_to_remove: list[int] = None
    remove_bos: bool = True
    ctx_len: int = 128
    get_full_model: bool = True
    sae_path: str = None
    coeff: float = None
    layer: int = None
    max_new_tokens: int = 100
    base_image_type: str = 'blank'
    n_generations: int = 5
    batch_size: int = 8
    save: bool = False
    temperature: float = 1
    sample: bool = False
    steering_position: str = 'input'
    run_type: str = 'validation'
    steering_type: str = 'raw'
    dataset_size: int = 50000
    intervention_type: str = 'addition'
    top_k_random_sample: bool = False
    type_mask: str = 'masks'
    direction_type: str = 'sae'
    max_latents_test: int = None

def main(cfg):
    device = cfg.device
    submodel = cfg.submodel
    save = cfg.save
    sae_path = cfg.sae_path
    coeff = cfg.coeff
    max_new_tokens = cfg.max_new_tokens
    base_image_type = cfg.base_image_type
    lm_model_name = cfg.lm_model_name
    batch_size = cfg.batch_size # number of features per batch
    steering_position = cfg.steering_position
    lm_model_path = demo_config.LLM_CONFIG[lm_model_name].model_path
    top_k_random_sample = cfg.top_k_random_sample
    print(f'Top k random sample: {top_k_random_sample}')
    print(f'Type mask: {cfg.type_mask}')
    if cfg.type_mask == 'masks':
        mask_flag = True
        heatmaps_flag = False
    elif cfg.type_mask == 'heatmaps':
        heatmaps_flag = True
        mask_flag = False
    print()
    if cfg.sample == False:
        cfg.temperature = 0.0001
        cfg.n_generations = 1
        print("No sampling, temperature set to 0.0 and n_generations set to 1")
    else:
        cfg.temperature = TEMPERATURES[cfg.lm_model_name]
        print(f"Sampling, temperature set to {cfg.temperature}")
    sample = cfg.sample

    if cfg.steering_type == 'raw':
        num_images = 1
    else:
        num_images = TOP_K_IMAGES

    print(f"num images: {num_images}")

    # Get the random number id from the sae path
    list_metadata = sae_path.split('/')[-2].split('_')
    rnd_num_id = list_metadata[-1]

    # Load SAE
    sae, sae_config = load_dictionary(sae_path, device)
    model_name = cfg.model_name # We do not read from SAE since it can be a different model
    model_type = demo_config.LLM_CONFIG[model_name].model_type
    
    # Load SAE config metadata
    dict_size = sae_config['trainer']['dict_size']
    #layer = sae_config['trainer']['layer']
    submodule_name = sae_config['trainer']['submodule_name'] # f"{submodel}_{site}_{io}_layer_{layer}"
    submodule_name_list = submodule_name.split('_')
    submodel = submodule_name_list[0].replace('-', '_')
    site = submodule_name_list[1]
    io = submodule_name_list[2]
    layer_num = int(submodule_name_list[4])
    activation_dim = get_activation_dim(model_name, submodel)
    dtype = demo_config.LLM_CONFIG[model_name].dtype

    
    # Set the config parameters
    cfg.model_type = model_type
    cfg.activation_dim = activation_dim
    cfg.submodel = submodel
    cfg.site = site
    cfg.io = io
    cfg.dtype = dtype

    sae_architecture = sae_config['trainer']['dict_class']
    sae.to(device)
    sae.to(dtype)

    model, tokenizer, processor = load_model(lm_model_path, cfg, device=device)
    # submodule = load_vision_model(model, cfg)
    if 'gemma-3' in model_name.lower() or 'paligemma' in model_name.lower():
        submodule = resolve_attr(model.vision_tower.vision_model, f"encoder.layers[{layer_num}]")
    elif 'internvl3' in model_name.lower():
        submodule = resolve_attr(model.vision_model, f"encoder.layers[{layer_num}]")


    def get_baseline_batch_inputs(prompt: str, batch_latent_ids: List[int], base_image_type: str, steering_type: str, **top_k_kwargs):
        """
        Creates a batch of baseline inputs for steering experiments.
        
        This function prepares a batch of identical inputs using either a blank image,
        random noise, or a black image as the baseline, depending on the specified type.
        
        Args:
            batch_size (int): Number of identical inputs to create in the batch
            base_image_type (str): Type of baseline image to use:
                - 'blank': White/empty image with values divided by themselves (results in NaN/1)
                - 'random': Random noise image
                - 'black': Near-black image (very small values)
                
        Returns:
            dict: Batch of tokenized inputs with the specified baseline images
        """

        batch_size = len(batch_latent_ids)

        if cfg.steering_type == 'raw':            
            # Return the baseline image
            return get_baseline_image_steering(prompt, steering_type, base_image_type, tokenizer, processor, cfg, batch_size)

        elif cfg.steering_type == 'steering_aided_top_k' or cfg.steering_type == 'none':
            # Return the top k images and heatmaps
            batch_top_k_images = [read_top_k_images(top_k_kwargs['ds'], latent_id, top_k_kwargs['top_k_images_dir'], k=top_k_kwargs['num_images'], top_k_random_sample=top_k_kwargs['top_k_random_sample'])['images'] for latent_id in batch_latent_ids[::top_k_kwargs['num_images']]]
            batch_top_k_heatmaps = [read_top_k_images(top_k_kwargs['ds'], latent_id, top_k_kwargs['top_k_images_dir'], k=top_k_kwargs['num_images'], top_k_random_sample=top_k_kwargs['top_k_random_sample'])['heatmaps'] for latent_id in batch_latent_ids[::top_k_kwargs['num_images']]]

            batch_top_masked_images = []
            positions_heatmaps = []
            for images_latent, heatmaps_latent in zip(batch_top_k_images, batch_top_k_heatmaps):
                image_arrays = []
                feature_heatmaps = []
                for image_data, heatmap_data in zip(images_latent, heatmaps_latent):
                    img_array = np.array(image_data) if isinstance(image_data, Image.Image) else image_data
                    image_arrays.append(img_array)
                    feature_heatmaps.append(heatmap_data)
                    positions_heatmap = np.where(heatmap_data.flatten() > 0)[0]
                    positions_heatmaps.append(positions_heatmap)
                    

                batch_top_images = get_or_create_attribution_images(image_arrays, feature_heatmaps, heatmaps_flag=top_k_kwargs['heatmaps_flag'], masks_flag=top_k_kwargs['mask_flag'])
                batch_top_masked_images.append(batch_top_images)

            raw_inputs = [{'text': [prompt], 'image': batch_top_images} for batch_top_images in batch_top_masked_images]
            data_batch = tokenized_batch(raw_inputs, tokenizer, cfg, processor)
            return data_batch, positions_heatmaps

    if cfg.steering_type != 'raw':
        ds = load_dataset_from_yaml(DATASET_NAME, split=DATASET_SPLIT)[0]
    else:
        ds = None

    clean_model_name = model_name.replace("/", "_")
    sae_id = f"{clean_model_name}_{sae_architecture}/{submodule_name}_{dict_size}_{rnd_num_id}"

    if 'gemma-3' in model_name.lower() or 'paligemma' in model_name.lower() or 'llama' in model_name.lower():
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=sample,
                                            top_p=0.95 if sample else None, top_k=64 if sample else None, 
                                            temperature=cfg.temperature,
                                            num_return_sequences=cfg.n_generations)
        print(f"Generation config: {generation_config.to_dict()}")
        generation_kwargs = {'output_scores': True,
                            'return_dict_in_generate': True,
                            'renormalize_logits': True,
                            'num_return_sequences': cfg.n_generations}
    elif 'internvl3' in model_name.lower():
        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=sample, temperature=cfg.temperature)
        generation_kwargs = {}

    prompt = load_prompt(model_name, cfg.steering_type, cfg.base_image_type, mask_flag=mask_flag, heatmaps_flag=heatmaps_flag)
    print(f"Prompt: {prompt}")

    if cfg.run_type == "validation":
        latents_range = list(range(0, N_LATENTS_VAL))
    elif cfg.run_type == "test":
        end = N_LATENTS_VAL + cfg.max_latents_test if cfg.max_latents_test is not None else dict_size
        latents_range = list(range(N_LATENTS_VAL, end))
    else:
        raise ValueError(f"Invalid run type: {cfg.run_type}")

    top_k_images_dir = get_top_k_images_path(cfg.model_name, cfg.layer, subset=DATASET_SPLIT, dataset_size=cfg.dataset_size)
    print(f'Top k images dir: {top_k_images_dir}')

    if steering_position == 'middle':
        context_length = demo_config.LLM_CONFIG[model_name].context_length
        steering_position_in_model = [(context_length//2) - int(context_length*0.2), (context_length//2) + int(context_length*0.2)]
    elif steering_position == 'input':
        steering_position_in_model = 'input'

    features = [feature for i, feature in enumerate(list_features(Path(top_k_images_dir))) if i in latents_range]

    # For permuted control: shift the full feature list before batching so the
    # derangement is global (avoids the batch-size-1 edge case where a per-batch
    # shift would map every latent to itself)
    if cfg.direction_type == 'permuted':
        n = len(features)
        shift = max(1, n // 2)
        effective_features = features[shift:] + features[:shift]
    else:
        effective_features = features

    for features_batch_idx in tqdm(range(0, len(features), batch_size), desc="Generating explanations"):
        batch_latent_ids = features[features_batch_idx:features_batch_idx+batch_size] if features_batch_idx+batch_size <= len(features) else features[features_batch_idx:len(features)]
        batch_latent_ids = [int(feature.split("latent_")[-1]) for feature in batch_latent_ids for _ in range(num_images)]

        effective_batch = effective_features[features_batch_idx:features_batch_idx+batch_size] if features_batch_idx+batch_size <= len(features) else effective_features[features_batch_idx:len(features)]
        effective_latent_ids = [int(feature.split("latent_")[-1]) for feature in effective_batch for _ in range(num_images)]


        direction = sae.decoder.weight[:, effective_latent_ids]

        # For random norm-matched control: replace direction with a random vector of the same norm
        if cfg.direction_type == 'random_norm_matched':
            direction_norms = direction.norm(dim=0, keepdim=True)  # [1, batch]
            random_dir = torch.randn_like(direction)
            random_dir = random_dir / random_dir.norm(dim=0, keepdim=True).clamp(min=1e-8)
            direction = random_dir * direction_norms
        # Repeat the direction tensor along the last dimension to match num_return_sequences
        # This ensures we have the same steering direction for each generated sequence
        num_sequences_per_prompt = cfg.n_generations# if cfg.model_name != 'OpenGVLab/InternVL3-14B' else 1
        direction_t = direction.t() # Transpose direction(s) to [batch_size, hidden_dim]
        # Repeat each row (corresponding to a latent) num_return_sequences times
        # This creates a tensor of shape [batch_size * num_return_sequences, hidden_dim]
        repeated_direction_t = torch.repeat_interleave(direction_t, num_sequences_per_prompt, dim=0)

        if cfg.steering_type == 'steering_aided_top_k' or cfg.steering_type == 'none':
            top_k_kwargs = {
                'ds': ds,
                'top_k_images_dir': top_k_images_dir,
                'num_images': num_images,
                'heatmaps_flag': heatmaps_flag,
                'mask_flag': mask_flag,
                'top_k_random_sample': top_k_random_sample
            }
            # We extract the positions of the "heatmaps" where the SAE latent is activated
            # Use effective_latent_ids so permuted control loads images for the permuted latent
            data_batch, positions_heatmaps = get_baseline_batch_inputs(prompt, effective_latent_ids, cfg.base_image_type, cfg.steering_type, **top_k_kwargs)
            if cfg.steering_type == 'steering_aided_top_k':
                # We need the hook to do steering
                steering_fwd_post_hooks = [(submodule, get_activation_addition_output_post_hook_v2(vector=repeated_direction_t, coeff=coeff, position=positions_heatmaps, intervention_type=cfg.intervention_type))]
            elif cfg.steering_type == 'none':
                # We do not need the hook to do steering
                steering_fwd_post_hooks = []
        elif cfg.steering_type == 'raw':
            data_batch, _ = get_baseline_batch_inputs(prompt, effective_latent_ids, cfg.base_image_type, cfg.steering_type)
            # We need the hook to do steering
            steering_fwd_post_hooks = [(submodule, get_activation_addition_output_post_hook_v2(vector=repeated_direction_t, coeff=coeff, position=steering_position_in_model, intervention_type=cfg.intervention_type))]
        
        if cfg.model_name == 'OpenGVLab/InternVL3-14B':
            questions = data_batch['questions'][0][0]
            pixel_values = data_batch['pixel_values']
            num_patches_list = data_batch['num_patches_list']
            pixel_values = torch.repeat_interleave(pixel_values, cfg.n_generations, dim=0)
            num_patches_list = [1] * cfg.n_generations
            questions = [questions] * cfg.n_generations
            steered_completions = generate_completions_internvl(model, tokenizer, pixel_values, questions, num_patches_list, fwd_pre_hooks=[], fwd_hooks=steering_fwd_post_hooks, generation_config=generation_config, **generation_kwargs)
            steered_log_probs = None
        else:
            steered_completions, steered_log_probs = generate_completions(model, tokenizer, data_batch, fwd_pre_hooks=[],
                                                                            fwd_hooks=steering_fwd_post_hooks,
                                                                            generation_config=generation_config, **generation_kwargs)
        
        # Reshape the completions to group them by prompt
        # Each prompt has num_return_sequences completions (as specified in generation_config)
        
        # Reshape the flat list into a list of sublists, where each sublist contains 
        # all completions for a single prompt
        
        for i, _latent_id in zip(range(0, len(steered_completions), num_sequences_per_prompt), batch_latent_ids[::num_images]):
            print(f"Latent id: {_latent_id}")
            prompt_completions = steered_completions[i:i+num_sequences_per_prompt]
            for to_replace in to_replace_list:
                prompt_completions = [completion.replace(to_replace, "").strip().capitalize() for completion in prompt_completions]

            
            log_probs = steered_log_probs[i:i+num_sequences_per_prompt] if steered_log_probs is not None else None
            
            print(f"Prompt completions: {prompt_completions}", end='\n')
            # print(f"Log probs: {log_probs}")

            if save:
                clean_model_name = model_name.replace("/", "_")
                if cfg.steering_type == 'none':
                    if mask_flag:
                        explanation_mode = f'HF_easy_masks_TOPK-{num_images}_MODEL-{lm_model_name.replace("/", "_")}' + f'_RNDTOPK-{cfg.top_k_random_sample}'
                    elif heatmaps_flag:
                        explanation_mode = f'HF_easy_heatmaps_TOPK-{num_images}_MODEL-{lm_model_name.replace("/", "_")}' + f'_RNDTOPK-{cfg.top_k_random_sample}'
                
                else:
                    fixed_steering_suffix = f'{lm_model_name.replace("/", "_")}_sampling-{cfg.sample}' + "_" + cfg.base_image_type + "_input_"
                    if cfg.steering_type == 'steering_aided_top_k':
                        if mask_flag:
                            explanation_mode = f"easy_w_steering_masks_" + fixed_steering_suffix + f"_RNDTOPK-{cfg.top_k_random_sample}" + f"_COEFF-{int(cfg.coeff)}"
                        elif heatmaps_flag:
                            explanation_mode = f"easy_w_steering_heatmaps_" + fixed_steering_suffix + f"_RNDTOPK-{cfg.top_k_random_sample}" + f"_COEFF-{int(cfg.coeff)}"
                    
                    elif cfg.steering_type == 'raw':
                        explanation_mode = f"steering_" + fixed_steering_suffix + f"_COEFF-{int(cfg.coeff)}"

                if cfg.direction_type != 'sae':
                    explanation_mode = explanation_mode + f"_DIRTYPE-{cfg.direction_type}"

                if cfg.dataset_size != 50000:
                    # Add the dataset size to the explanation mode
                    explanation_mode = explanation_mode + f"_SIZE-{cfg.dataset_size}"

                sae_id = f"{clean_model_name}_{sae_architecture}/{submodule_name}_{dict_size}_{rnd_num_id}"
                if cfg.run_type == "validation":
                    output_dir = f"{DATA_DIR}/outputs/validation"
                elif cfg.run_type == "test":
                    output_dir = f"{DATA_DIR}/outputs"
                else:
                    raise ValueError(f"Invalid run type: {cfg.run_type}")
                save_dir = os.path.join(output_dir, sae_id, f"{explanation_mode}", f"latent_{_latent_id}", 'explanations')
                print(f"Saving explanations to {save_dir}")
                logging.info(f"Saving explanations to {save_dir}")

                explanation = {
                        "feature_name": _latent_id,
                        "prompt": prompt,
                        "explanation": prompt_completions,
                        "raw_output": prompt_completions,
                        "prompt_log_probs": log_probs,
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "model": lm_model_name.replace("/", "_"),
                            "prompt_system": "",
                            "base_image_type": base_image_type,
                            "steering_coeff": str(int(coeff)),
                            "sampling": generation_config.to_dict() if type(generation_config) == GenerationConfig else generation_config,
                            "sae_id": sae_id,
                        }
                }

                # Create a filename for this latent
                latent_filename = create_output_path(Path(save_dir), filename=f"explanation.json")
                # Save the latent data to a JSON file
                save_json_data(explanation, Path(latent_filename))

    print('DONE!')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--layer", type=str, default="mid", choices=["early", "mid", "later"])
    parser.add_argument("--lm_model_name", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--submodel", type=str, default="enc")
    parser.add_argument("--coeff", type=float, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--steering_position", type=str, default="input")
    parser.add_argument("--steering_type", type=str, choices=["raw", "steering_aided_top_k", "none"], default="raw")
    parser.add_argument("--base_image_type", type=str, choices=["blank", "random", "black", "none"], default="blank")
    parser.add_argument("--intervention_type", type=str, choices=["addition", "replacement"], default="addition")
    parser.add_argument("--n_generations", type=int, default=1)
    parser.add_argument("--sample", action='store_true', default=False)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--run_type", type=str, choices=["validation", "test"], default="validation")
    parser.add_argument("--save", action='store_true', default=False)
    parser.add_argument("--dataset_size", type=int, default=50000)
    parser.add_argument("--top_k_random_sample", action='store_true', default=False)
    parser.add_argument("--type_mask", type=str, choices=["heatmaps", "masks"], default="masks")
    parser.add_argument("--direction_type", type=str, choices=["sae", "random_norm_matched", "permuted"], default="sae")
    parser.add_argument("--max_latents_test", type=int, default=None,
                        help="Number of latents to process for test split; defaults to all")

    args = parser.parse_args()


    assert args.steering_type in ['raw', 'steering_aided_top_k', 'none'], "Steering type must be raw, steering_aided_top_k or none"
    assert args.steering_position != 'middle', "Steering position must be input or top_k_images"

    sae_path = get_paths(args.model_name, args.layer)


    cfg = ExperimentConfig(
        model_name=args.model_name,
        lm_model_name=args.lm_model_name,
        model_path=demo_config.LLM_CONFIG[args.model_name].model_path,
        sae_path=sae_path,
        coeff=args.coeff,
        max_new_tokens=args.max_new_tokens,
        steering_position=args.steering_position,
        n_generations=args.n_generations,
        save=args.save,
        sample=args.sample,
        temperature=args.temperature,
        run_type=args.run_type,
        steering_type=args.steering_type,
        layer=args.layer,
        dataset_size=args.dataset_size,
        intervention_type=args.intervention_type,
        top_k_random_sample=args.top_k_random_sample,
        type_mask=args.type_mask,
        direction_type=args.direction_type,
        max_latents_test=args.max_latents_test
    )
    if args.steering_type == 'steering_aided_top_k':
        cfg.base_image_type = 'top_k_images'

    if cfg.lm_model_name == 'OpenGVLab/InternVL3-14B':
        cfg.batch_size = 1
        print(f'Setting batch size to 1 for {cfg.lm_model_name}')

    main(cfg)


'''
Usage:
python -m src.get_steering_explanations --model_name google/gemma-3-4b-it --lm_model_name google/gemma-3-4b-it --layer mid --n_generations 1 --coeff 100 --max_new_tokens 200 --steering_position input --run_type validation --save

python -m src.get_steering_explanations --model_name google/gemma-3-4b-it --lm_model_name google/gemma-3-4b-it --layer mid --n_generations 1 --coeff 200 --max_new_tokens 200 --steering_position input --run_type test

python -m src.get_steering_explanations --model_name OpenGVLab/InternVL3-14B --lm_model_name OpenGVLab/InternVL3-14B --layer mid --n_generations 1 --coeff 0 --max_new_tokens 200 --steering_position input --run_type validation --steering_type raw

python -m src.get_steering_explanations --model_name OpenGVLab/InternVL3-14B --lm_model_name OpenGVLab/InternVL3-14B --layer mid --n_generations 5 --coeff 100 --max_new_tokens 200 --steering_position input --run_type validation --steering_type raw --sample
'''
# %%

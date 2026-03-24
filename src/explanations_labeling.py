# %%
import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

REPO_DIR = os.environ.get('REPO_DIR')
sys.path.append(REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, 'src'))
    
from typing import List, Dict, Optional, Union, Tuple
import torch
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import argparse

from dictionary_learning.buffer import tokenized_batch
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
from utils.utils import get_outputs_path, model_types, read_segmentation_results, load_model
from dataclasses import dataclass

DATA_DIR = os.environ.get('DATA_DIR')


ABSTRACTION_LEVEL_PROMPT = '''You are given an explanation of a visual or semantic element. Your task is to assign a label to the explanation. The labels are:
- 'low-level visual feature'
- 'high-level visual feature'
- 'other'

Select the most appropriate label and paste it.

Explanation:
'''

BACKGROUND_OBJECT_PROMPT = '''You are given an explanation of a visual or semantic element. Your task is to assign a label to the explanation. The labels are:
- 'background': ground, sky, wall, landscape, etc.
- 'animal': dog, cat, bird, etc.
- 'other'

Select the most appropriate label (background, animal, other) and paste it.
'''

@dataclass
class Config:
    subject_model_name: str = None
    engine: str = 'HF'
    model_name: str = None
    explainer_method_name: str = None
    layer: int = None
    device: str = 'cuda:0'
    submodel: str = 'enc'
    site: str = 'res'
    io: str = 'out'
    model_type: str = None
    dtype: torch.dtype = torch.bfloat16
    get_full_model: bool = True
    outputs_path: str = None
    max_new_tokens: int = 50
    sample: bool = False
    temperature: float = 1
    save: bool = False
    verbose: bool = False
    run_type: str = None
    batch_size: int = 64

def get_labels(model, processor, data_batch, cfg: Config) -> str:

    completions = []

    with torch.inference_mode():
        generation_toks_batch = model.generate(**data_batch, max_new_tokens=cfg.max_new_tokens,
                                                do_sample=cfg.sample, temperature=cfg.temperature,
                                                )
        generation_toks_batch = generation_toks_batch[:, data_batch.input_ids.shape[-1]:]
    
    for generation in generation_toks_batch:
        completions.append(processor.decode(generation, skip_special_tokens=True).strip())

    return completions

# %%
def main(cfg, label_type: str):

    model, tokenizer, processor = load_model(cfg.model_name, 'HF', cfg, device=cfg.device)
    
    outputs_path = get_outputs_path(cfg.subject_model_name, cfg.layer, cfg.run_type)
    explanations_path = os.path.join(outputs_path, cfg.explainer_method_name)
    explanations_explainer_dict = read_segmentation_results(explanations_path)
    
    # We iterate over the latent directories sorted by the number of the latent
    explanations_explainer_dict = dict(sorted(explanations_explainer_dict.items()))

    print('NUMBER OF LATENTS', len(explanations_explainer_dict.keys()))

    features = list(explanations_explainer_dict.keys())

    results_dict = {}

    counter = 0
    for features_batch_idx in tqdm(range(0, len(features), cfg.batch_size), desc="Generating labels"):
        latents_ids = features[features_batch_idx:features_batch_idx+cfg.batch_size] if features_batch_idx+cfg.batch_size < len(features) else features[features_batch_idx:len(features)]
        #print('LATENT DIR', latent_dir)
        explanations = []
        prompts = []
        for latent in latents_ids:
            explanation = explanations_explainer_dict[latent]
            explanations.append(explanation)
            if label_type == 'abstraction_level':
                prompts.append(f"{ABSTRACTION_LEVEL_PROMPT}\n{explanation}")
            elif label_type == 'background_object':
                prompts.append(f"{BACKGROUND_OBJECT_PROMPT}\n{explanation}")
            else:
                raise ValueError(f"Label type {label_type} not supported")

        raw_inputs = [{'text': [prompt]} for prompt in prompts]

        data_batch = tokenized_batch(raw_inputs, tokenizer, processor, cfg)
        if cfg.verbose == True:
            logger.info("data_batch", data_batch)

        labels = get_labels(model, processor, data_batch, cfg)

        for label, prompt, explanation, latent in zip(labels, prompts, explanations, latents_ids):
            results_latent = {
                "prompt": prompt,
                "original_explanation": explanation,
                "label": [label],
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "sampling": cfg.sample,                        
                }
            }
            # print(f"Latent: {latent}")
            # print(label)


            results_dict[latent] = results_latent
            if cfg.verbose:
                logger.info(f"Latent: {latent}")
                logger.info(explanation)
        counter += 1
        

    if cfg.save:
        save_dir_explanation = f"{explanations_path}/labels_{label_type}.json"
        print(save_dir_explanation)
        with open(save_dir_explanation, "w") as f:
            json.dump(results_dict, f, indent=2)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_model_name", type=str, default="google/gemma-3-4b-it") # Subject model
    parser.add_argument("--layer", type=str, default="mid", choices=["early", "mid", "later"])
    parser.add_argument("--refiner_model_name", type=str, default="google/gemma-3-27b-it") # Explainer model
    parser.add_argument("--explainer_method_name", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--sample", action='store_true', default=False)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--run_type", type=str, default="test")
    parser.add_argument("--save", action='store_true', default=False)
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--label_type", type=str, default="abstraction_level")

    args = parser.parse_args()

    print('args.verbose', args.verbose)

    outputs_path = get_outputs_path(args.subject_model_name, args.layer, args.run_type)
    
    cfg = Config(
        subject_model_name=args.subject_model_name,
        model_name=args.refiner_model_name,
        layer=args.layer,
        explainer_method_name=args.explainer_method_name,
        max_new_tokens=args.max_new_tokens,
        save=args.save,
        sample=args.sample,
        temperature=args.temperature,
        outputs_path=outputs_path,
        model_type = model_types[args.refiner_model_name],
        verbose=args.verbose,
        run_type=args.run_type
    )
    #print(cfg)
    main(cfg, args.label_type)

# Usage
# python -m src.explanations_labeling --explainer_method_name HF_easy2_heatmaps_TOPK-5_MODEL-gemma-3-27b-it_SIZE-1000 --run_type test
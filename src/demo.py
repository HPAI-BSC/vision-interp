# %%
import sys
import os
from dotenv import load_dotenv
load_dotenv()
REPO_DIR = os.environ.get('REPO_DIR')
# Check if REPO_DIR environment variable is set
if REPO_DIR is None:
    print("Error: REPO_DIR environment variable is not set.")
    sys.exit(1)
sys.path.append(REPO_DIR)
sys.path.append(os.path.join(REPO_DIR, 'src'))

import torch
from functools import partial
import argparse
import itertools
from tqdm import tqdm
from collections import defaultdict
import os
import random
import torch.multiprocessing as mp
import time
import uuid
from utils.utils import load_model

import demo_config
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.training import trainSAE
import dictionary_learning

from evaluate_sae import eval_saes
from demo_config import get_activation_dim, get_context_length
from dataclasses import dataclass

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

import logging
logging.getLogger().setLevel(logging.ERROR)

LOG_STEPS = 100 # Log the training on wandb or print to console every log_steps

@dataclass
class TrainingConfig:
    model_name: str = 'google/paligemma2-3b-mix-224'
    layer: int = None
    dataset: str = "ILSVRC/imagenet-1k"
    device: str = 'cuda:0'
    model_type: str = None
    model_path: str = None
    submodel: str = 'enc'
    io: str = 'out'
    activation_dim: int = None
    dtype: torch.dtype = torch.bfloat16
    tokens_to_remove: list[int] = None
    remove_bos: bool = True
    ratio_of_training_data: float = 0.1
    get_full_model: bool = False
    context_length: int = None
    model_img_size: int = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.environ.get('DATA_DIR'), 'saes'), help="where to store SAEs checkpoints")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb logging")
    parser.add_argument("--save_checkpoints", action="store_true", help="save checkpoints at different stages of training")
    parser.add_argument(
        "--layers", type=int, nargs="+", required=True, help="layers to train SAE on"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="which language model to use",
    )
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="+",
        choices=[e.value for e in demo_config.TrainerType],
        required=True,
        help="which SAE architectures to train",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device to train on")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="txt360",
        help="dataset to use for training"
    )
    parser.add_argument("--test_set", type=str, default=None, help="test set")
    parser.add_argument("--submodel", type=str, default='enc', choices=['enc', 'dec'], help="submodel to use (dec, enc)")
    parser.add_argument("--io", type=str, default='out', choices=['in', 'out'], help="io to use (in, out)")
    parser.add_argument("--ratio_of_training_data", type=float, default=1, help="ratio of training data to use (default: 1)")
    args = parser.parse_args()
    return args


def run_sae_training(
    layer: int,
    save_dir: str,
    architectures: list,
    random_seeds: list[int],
    dictionary_widths: list[int],
    learning_rates: list[float],
    use_wandb: bool = False,
    save_checkpoints: bool = False,
    buffer_tokens: int = 1_000_000,
    cfg: TrainingConfig = None
):
    random.seed(demo_config.random_seeds[0])
    torch.manual_seed(demo_config.random_seeds[0])

    model_name = cfg.model_name
    dataset = cfg.dataset
    io = cfg.io
    submodel = cfg.submodel
    device = cfg.device
    dtype = cfg.dtype

    if cfg.model_type == 'lm':
        assert submodel == 'dec', "submodel must be dec for language models"


    submodule_name = f"{submodel.replace('_', '-')}_res_{io}_layer_{layer}"

    # model and data parameters
    sae_batch_size = demo_config.LLM_CONFIG[model_name].sae_batch_size

    print('save_dir', save_dir)

    # Model-specific parameters
    context_length = cfg.context_length # max number of tokens in context (i.e. max tokens per datapoint)
    llm_batch_size = demo_config.LLM_CONFIG[model_name].llm_batch_size # number of datapoints to process at once by the LLM
    n_ctxs = buffer_tokens // context_length # number of contexts in the buffer
    print(f"n_ctxs: {n_ctxs}, buffer_size_in_tokens: {buffer_tokens}")

    # Load model, tokenizer and submodule
    model, tokenizer, processor = load_model(model_name, cfg, dtype, device)

    # Get generator from HF dataset (columns resolved from YAML)
    generator, len_dataset = hf_dataset_to_generator(dataset, ratio_of_training_data=cfg.ratio_of_training_data)
    # len_dataset is the number of datapoints in the dataset
    num_tokens = len_dataset*context_length # Number of total tokens in the dataset
    steps = num_tokens // sae_batch_size # Total number of SAE batches to train
        
    print(f"LEN DATASET: {len_dataset}")
    print(f"NUM TOKENS: {num_tokens}")

    activation_buffer = ActivationBuffer(
        generator,
        model,
        n_ctxs=n_ctxs,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        tokenizer=tokenizer,
        processor=processor,
        max_activation_norm_multiple=demo_config.max_activation_norm_multiple,
        training=True,
        cfg=cfg
    )

    if save_checkpoints:
        desired_checkpoints = [0.1, 0.25, 0.5, 0.75, 1]
        desired_checkpoints.sort()
        print(f"desired_checkpoints: {desired_checkpoints}")

        save_steps = [int(steps * step) for step in desired_checkpoints]
        save_steps.sort()
        print(f"save_steps: {save_steps}")
    else:
        save_steps = None

    trainer_configs = demo_config.get_trainer_configs(
        architectures,
        learning_rates,
        random_seeds,
        dictionary_widths,
        layer,
        submodule_name,
        steps,
        num_tokens,
        cfg=cfg
    )

    #print(f"trainer_configs: {trainer_configs}")

    print(f"len trainer configs: {len(trainer_configs)}")
    assert len(trainer_configs) == 1, "Only one trainer config is supported"
    trainer_config = trainer_configs[0]
    sae_architecture = architectures[0]
    # Generate a random 8-digit number
    rnd_num_id = str(int(uuid.uuid4()))[:8]
    #del trainer_config["trainer"]
    dict_size = trainer_config['dict_size']
    layer = trainer_config['layer']
    if 'k' in trainer_config:
        k = trainer_config['k']
    else:
        k = trainer_config['target_l0']
    submodule_name = trainer_config['submodule_name']


    suffix = model_name.replace("/", "_")
    save_dir = f"{save_dir}/{suffix}"
    save_dir = f"{save_dir}/{submodule_name}_{sae_architecture}_{dict_size}_{k}_{cfg.ratio_of_training_data}_{rnd_num_id}"

    trainSAE(
        data=activation_buffer,
        trainer_configs=trainer_configs,
        use_wandb=use_wandb,
        steps=steps,
        save_steps=save_steps,
        save_dir=save_dir,
        log_steps=LOG_STEPS,
        wandb_project=demo_config.wandb_project,
        normalize_activations=True,
        verbose=False,
        autocast_dtype=dtype, #torch.bfloat16
    )
    del model, tokenizer, processor
    return save_dir


if __name__ == "__main__":
    args = get_args()

    # Load config and override with args
    cfg = TrainingConfig(
        model_name=args.model_name,
        model_path=demo_config.LLM_CONFIG[args.model_name].model_path,
        dataset=args.dataset,
        device=args.device,
        submodel=args.submodel,
        io=args.io,
        model_type=demo_config.LLM_CONFIG[args.model_name].model_type,
        activation_dim=get_activation_dim(args.model_name, args.submodel),
        dtype=demo_config.LLM_CONFIG[args.model_name].dtype,
        ratio_of_training_data=args.ratio_of_training_data,
        context_length=get_context_length(args.model_name, args.submodel),
        model_img_size=demo_config.LLM_CONFIG[args.model_name].model_img_size,
    )

    os.environ["WANDB_DIR"] = args.save_dir
    # This prevents random CUDA out of memory errors
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # For wandb to work with multiprocessing
    mp.set_start_method("spawn", force=True)

    start_time = time.time()

    # Gemma-3 models have a lot of context length so we need a bigger buffer not to waste datapoints
    buffer_tokens = demo_config.buffer_tokens if 'gemma-3' not in args.model_name.lower() else 1_500_000

    for layer in args.layers:
        cfg.layer = layer
        complete_save_dir = run_sae_training(
            layer=layer,
            save_dir=args.save_dir,
            architectures=args.architectures,
            random_seeds=demo_config.random_seeds,
            dictionary_widths=demo_config.dictionary_widths,
            learning_rates=demo_config.learning_rates,
            use_wandb=args.use_wandb,
            save_checkpoints=args.save_checkpoints,
            buffer_tokens=buffer_tokens,
            cfg=cfg
        )

    ae_paths = dictionary_learning.utils.get_nested_folders(complete_save_dir)
    print(f"ae_paths: {ae_paths}")

    if args.test_set is None:
        print("No test set provided, evaluating on default eval set")

    eval_saes(
        demo_config.eval_dataset if args.test_set is None else args.test_set,
        ae_paths,
        n_inputs=2000,
        overwrite_prev_results=True,
        save_results=True,
        device=cfg.device,
    )

    print(f"Total time: {time.time() - start_time}")

# Usage example:
# python src/demo.py --model_name openai/clip-vit-large-patch14 --layers 8 --architectures top_k --dataset Bingsu/Cat_and_Dog --test_set Bingsu/Cat_and_Dog --ratio_of_training_data 1
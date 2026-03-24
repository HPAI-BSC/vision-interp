import torch as t
# HF models
from transformers import AutoModelForCausalLM, ViTForImageClassification
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
import gc
from tqdm import tqdm
from functools import partial
from .config import DEBUG
import copy
import time
from PIL import Image
from src.processing import tokenized_batch
from .hook_managers import HookManager


def remove_positions(tokenizer, input_ids, hidden_states, tokens_to_remove=None, remove_bos=True):
    """Remove specified tokens and padding tokens from the activations.
    
    Args:
        input_ids: The input token IDs
        hidden_states: The hidden states to filter
        tokens_to_remove: List of token IDs to remove (defaults to None)
    
    Returns:
        Filtered hidden states with specified tokens removed
    """
    # Always remove padding tokens
    mask_to_apply = input_ids != tokenizer.pad_token_id
    
    if remove_bos:
        # Remove BOS token if specified
        bos_token_mask = input_ids != tokenizer.bos_token_id
        mask_to_apply = bos_token_mask & mask_to_apply
    
    # Remove additional tokens if specified
    if tokens_to_remove is not None:
        for token_id in tokens_to_remove:
            token_mask = input_ids != token_id
            mask_to_apply = token_mask & mask_to_apply
    
    return hidden_states[mask_to_apply]

def hf_forward(model, data_batch, tokenizer, cfg, remove_high_norm=None, training=True):
    """Forward pass using HuggingFace transformers to extract activations from the model.
    
    Args:
        model: LanguageModel instance
        data_batch: Batch of data to forward pass
        tokenizer: Tokenizer instance
        cfg: Config
        use_hooks: Whether to use hooks to extract activations
    Returns:
        hidden_states: Tensor of activations extracted from the model
    """
    if hasattr(model, 'submodule'):
        use_hooks = True
    else:
        use_hooks = False

    model_kwargs = {}
    if use_hooks:
        # Submodule is available, we use it with hooks. Instead of using output_hidden_states (not available for all models).
        hook_manager = HookManager()
        hook_manager.attach_and_verify_hook(model.submodule, io=cfg.io)
    
    else:
        model_kwargs["output_hidden_states"] = True

    if any(x in cfg.model_name.lower() for x in ('qwen2-vl', 'qwen2.5-vl', 'mimo-vl', 'aloe-vision-7b')) and cfg.submodel == 'enc':
        with t.no_grad():
            output = model(data_batch['pixel_values'], grid_thw=data_batch['image_grid_thw'], **model_kwargs)
    elif 'internvl' in cfg.model_name.lower():
        with t.no_grad():
            output = model(data_batch['pixel_values'], **model_kwargs)
    else:
        with t.no_grad():
            output = model(**data_batch, **model_kwargs)
    
    if not use_hooks:
        if cfg.io == 'out':
            # First hidden state (idx 0) is the (output) of the embedding layer
            layer = cfg.layer+1
        else:
            layer = cfg.layer
        
        hidden_states = output['hidden_states'][layer]
    
    else:
        hidden_states = t.cat(hook_manager.hooks_saved, dim=0)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # Clear previous hook data
        hook_manager.clear_saved_data()
        hook_manager.remove_hooks()
        
    # When working with vision models, we can't access ids so we remove the CLS via indexing
    if cfg.model_type == 'vlm' and cfg.submodel == 'enc' and 'internvl' in cfg.model_name.lower():
        # We remove the CLS token
        # TODO: move this to cfg.tokens_to_remove
        hidden_states = hidden_states[:,1:,:]
    if cfg.model_type == 'vision':
        # By default we remove the CLS token in vision models
        if 'with-registers' in cfg.model_name.lower() and not training:
            # During inference (not training), we remove the CLS token and the 4 register tokens to be able to get
            # feature activation heatmaps across image patches
            hidden_states = hidden_states[:,1:-4,:]
        else:
            hidden_states = hidden_states[:,1:,:]

    if training:
        if (cfg.model_type == 'vlm' and cfg.submodel == 'dec') or cfg.model_type == 'lm':
            input_ids = data_batch['input_ids']
            # If we are training, we remove the tokens that can mess up the training (e.g. BOS token that is typically used as an attention sink)
            hidden_states = remove_positions(tokenizer, input_ids, hidden_states, cfg.tokens_to_remove, cfg.remove_bos)

    else:
        # Reshape to [batch_size*sequence_length, hidden_size]
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

    if training and remove_high_norm is not None:
        # some models (like Qwen) have random high norm activation sinks which reduce training effectiveness
        norms_BL = hidden_states.norm(dim=-1)
        median_norm = norms_BL.median()
        norm_mask = norms_BL > median_norm * remove_high_norm
        if norm_mask.sum() > 0:
            print(f"Removed {norm_mask.sum()} high norm activations")
            print(f"Median norm: {median_norm}, remove_high_norm: {median_norm * remove_high_norm}")
        hidden_states = hidden_states[~norm_mask]

    return hidden_states


class ActivationBuffer:
    """
    Implements a buffer of activations. The buffer stores activations from a model,
    yields them in batches, and refreshes them when the buffer is less than half full.
    """
    def __init__(self, 
                 data, # generator which yields text data
                 model,# LanguageModel from which to extract activations
                 n_ctxs=3e4, # approximate number of contexts to store in the buffer
                 ctx_len=128, # length of each context
                 refresh_batch_size=512, # size of batches in which to process the data when adding to buffer
                 out_batch_size=8192, # size of batches in which to yield activations
                 tokenizer=None,
                 processor=None,
                 max_activation_norm_multiple=None,
                 training=True,
                 cfg=None
                 ):

        # data vars
        self.data = data
        self.model = model
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len

        # tokenizer / processor
        self.tokenizer = tokenizer
        self.processor = processor

        # cfg vars
        self.cfg = cfg
        self.d_submodule = cfg.activation_dim
        self.io = cfg.io
        self.remove_bos = cfg.remove_bos
        self.dtype = cfg.dtype
        self.model_type = cfg.model_type
        self.tokens_to_remove = cfg.tokens_to_remove
        self.submodel = cfg.submodel
        self.device = cfg.device
        self.remove_high_norm = max_activation_norm_multiple

        # buffer vars
        self.activation_buffer_size = n_ctxs * ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.training = training

        if self.io not in ['in', 'out']:
            raise ValueError("io must be either 'in' or 'out'")

        self.activations = t.empty(0, self.d_submodule, device=self.device, dtype=self.dtype)
        self.read = t.zeros(0).bool()

    def __iter__(self):
        return self

    def __next__(self, deterministic=False):
        """
        Return a batch of activations
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.activation_buffer_size // 2:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            if deterministic == False:
                idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:self.out_batch_size]]
            else:
                idxs = unreads[:self.out_batch_size]
            self.read[idxs] = True
            return self.activations[idxs]
        
    def input_batch(self, batch_size=None):

        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            # return list of texts
            return [
                next(self.data) for _ in range(batch_size)
            ]

        except StopIteration:
            raise StopIteration("End of data stream reached")
    
        
    def refresh(self):
        gc.collect()
        t.cuda.empty_cache()
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        new_activations = t.empty(self.activation_buffer_size, self.d_submodule, device=self.device, dtype=self.dtype)

        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations

        # Optional progress bar when filling buffer. At larger models / buffer sizes (e.g. gemma-2-2b, 1M tokens on a 4090) this can take a couple minutes.
        # pbar = tqdm(total=self.activation_buffer_size, initial=current_idx, desc="Refreshing activations")

        while current_idx < self.activation_buffer_size:
            # Clear previous data before processing
            with t.no_grad():
                # Get input batch
                input_batch = self.input_batch()
                data_batch = tokenized_batch(input_batch, self.tokenizer, self.cfg, self.processor)

                hidden_states = hf_forward(self.model, data_batch, self.tokenizer, self.cfg,
                                           remove_high_norm=self.remove_high_norm, training=self.training)

            remaining_space = self.activation_buffer_size - current_idx
            assert remaining_space > 0
            hidden_states = hidden_states[:remaining_space]
            self.activations[current_idx : current_idx + len(hidden_states)] = hidden_states.to(
                self.device
            )
            current_idx += len(hidden_states)

        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    @property
    def config(self):
        return {
            'n_ctxs' : self.n_ctxs,
            'ctx_len' : self.ctx_len,
            'refresh_batch_size' : self.refresh_batch_size,
            'out_batch_size' : self.out_batch_size,
            'device' : self.device
        }

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()
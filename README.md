# Language Models Can Explain Visual Features via Steering [CVPR 2026]

<div align="center">
<!--   <img src="https://raw.githubusercontent.com/HPAI-BSC/vision-interp/TODO.png" width="400" alt="HPAI"/> -->
</div>
<div align="center" style="line-height: 1;">
  <a href="https://hpai.bsc.es/" target="_blank" style="margin: 1px;">
    <img alt="Web" src="https://img.shields.io/badge/Website-HPAI-8A2BE2" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/HPAI-BSC" target="_blank" style="margin: 1px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HPAI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/HPAI-BSC" target="_blank" style="margin: 1px;">
    <img alt="GitHub" src="https://img.shields.io/badge/GitHub-HPAI-%23121011.svg?logo=github&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
  <a href="https://www.linkedin.com/company/hpai" target="_blank" style="margin: 1px;">
    <img alt="Linkedin" src="https://img.shields.io/badge/Linkedin-HPAI-blue" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://bsky.app/profile/hpai.bsky.social" target="_blank" style="margin: 1px;">
    <img alt="BlueSky" src="https://img.shields.io/badge/Bluesky-HPAI-0285FF?logo=bluesky&logoColor=fff" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://linktr.ee/hpai_bsc" target="_blank" style="margin: 1px;">
    <img alt="LinkTree" src="https://img.shields.io/badge/Linktree-HPAI-43E55E?style=flat&logo=linktree&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
<!--   <a href="TODO" target="_blank" style="margin: 1px;">
    <img alt="Arxiv" src="https://img.shields.io/badge/arXiv-TODO.svg" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="LICENSE" style="margin: 1px;">
    <img alt="License" src="https://img.shields.io/github/license/HPAI-BSC/vision-interp" style="display: inline-block; vertical-align: middle;"/>
  </a> -->
</div>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2603.22593" target="_blank" style="margin: 1px;">
    <img alt="Arxiv" src="https://img.shields.io/badge/arXiv-2603.22593-b31b1b.svg" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>


## Getting Started
Navigate to the path where you would like to clone this repository (will refer to `REPO_DIR`) and run:

```bash
git clone https://github.com/HPAI-BSC/HF-SAE.git
cd HF-SAE
```

Run the following to get set up using `uv`:

```bash
# Install Python 3.10 and create env
uv python install 3.10
uv venv --python 3.10

# Sync dependencies
uv sync

# Clone and add lang-segment-anything (required for baseline_simulator.py)
git clone https://github.com/luca-medeiros/lang-segment-anything.git
uv add ./lang-segment-anything
```


For downloading models/datasets from Hugging Face set the environment variables to allow internet access and identify yourself:
```bash
huggingface-cli login
```

Set the environment variables in `.env` file.
```bash
REPO_DIR=... # Your repository directory
DATA_DIR=... # Your data directory, where artifacts will be saved
HF_HOME=... # Optional, custom Hugging Face cache directory, default is `~/.cache/huggingface`
```

## Config files
The config files are located in `config` directory. If you want to add a new dataset or new model, you need to add it to `dataset_config.yaml` and `llm_config.yaml` files:

`llm_config.yaml` contains the configuration for the LLMs.
`dataset_config.yaml` contains the configuration for the datasets.
`saes.yaml` contains the paths to the SAEs and top-k images and outputs for each model and layer.

## Pretrained SAEs

Pretrained SAEs are available on Hugging Face and are downloaded automatically by the code when needed:

- **Gemma-3-4B-IT**: [javifer/google_gemma-3-4b-it-saes](https://huggingface.co/javifer/google_gemma-3-4b-it-saes)
- **InternVL3-14B**: [javifer/OpenGVLab_InternVL3-14B-saes](https://huggingface.co/javifer/OpenGVLab_InternVL3-14B-saes)

## SAE Training

For training SAEs, run`src/demo.py`, which launches training and evaluation of an SAE(s). Some training parameters are passed as arguments to `src/demo.py`, and the rest are set in the config file `src/demo_config.py`:

```bash
python src/demo.py \
  --model_name google/gemma-3-4b-it \
  --layers 16 \
  --architectures top_k \
  --dataset ILSVRC/imagenet-1k \
  --test_set ILSVRC/imagenet-1k \
  --ratio_of_training_data 0.5 \
  --submodel enc
```

## Top Activating Visualizations
To compute top activating visualizations, we use `src/get_max_activating_vision.py` script.

```bash
python src/get_max_activating_vision.py \
  --top_k 10 \
  --ids_selection top_k \
  --n_images 128 \
  --sae_path [PATH_TO_SAE]
```

To visualize the top activating visualizations, use `notebooks/max_activating_viz_vision_read.ipynb` notebook.

## Evaluation

Three evaluators are available in the `evaluation/` directory to assess SAE feature explanations. See [`evaluation/README.md`](evaluation/README.md) for full usage details.

- **Baseline Simulator** (`evaluation/baseline_simulator.py`): Compares SAE activation heatmaps with LangSAM text-guided segmentation masks (IoU, precision, recall).
- **CLIP Simulator** (`evaluation/clip_simulator.py`): Measures CLIP similarity between top-k images and their concept explanations.
- **Image Generation Evaluator** (`evaluation/image_gen_eval.py`): Checks whether images generated from concept explanations activate the expected SAE features more than random images (AUROC).

## Data Format

Based on the model type, the input is processed and tokenized differently, this is handled by `tokenized_batch` function in `src/processing.py`.

If we are working with a vision-language model and use both text and images, the input format is:
```python 
input = [
    {'image': [image]},
    {'text': ['Text...(e.g. Describe this image in detail.)']}
]
data_batch = tokenized_batch(input, processor.tokenizer, processor, cfg)
```
If we are working with a vision model, the input format is:
```python 
input = [{'image': [image]}]
data_batch = tokenized_batch(input, processor.tokenizer, processor, cfg)
```

If we are working with a text model, the input format is:
```python 
input = [{'text': ['Text...']}]
data_batch = tokenized_batch(input, processor.tokenizer, processor, cfg)
```
# Score Entropy Discrete Diffusion
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repo contains a PyTorch implementation for the paper [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution
](https://arxiv.org/abs/2310.16834) by [Aaron Lou](https://aaronlou.com), [Chenlin Meng](https://cs.stanford.edu/~chenlin/) and [Stefano Ermon](https://cs.stanford.edu/~ermon/).

![cover](assets/main.gif)

## Design Choices

This codebase is built modularly to promote future research (as opposed to a more compact framework, which would be better for applications). The primary files are 

1. ```noise_lib.py```: the noise schedule
2. ```graph_lib```: the forward diffusion process
3. ```sampling.py```: the sampling strategies
4. ```model/```: the model architecture

## Installation

Simply run

```
conda env create -f environment.yml
```

which will create a ```sedd``` environment with packages installed. Note that this installs with CUDA 11.8, and different CUDA versions must be installed manually. The biggest factor is making sure that the ```torch``` and ```flash-attn``` packages use the same CUDA version (more found [here](https://github.com/Dao-AILab/flash-attention)).

### Windows

On Windows, create the portable environment instead (NVIDIA GPU/CUDA 12.8):

```
conda env create -f environment-windows.yml
conda activate sedd
```

`flash-attn` has no native Windows wheel, so the code automatically falls back to
PyTorch scaled-dot-product attention. Sampling selects CUDA when available and CPU
otherwise; it can also be forced with `--device cpu` or `--device cuda`. CPU sampling
of the pretrained models is supported but is expected to be very slow.

## Working with Pretrained Models

### Download Models

Our pretrained models are hosted on huggingface ([small](https://huggingface.co/louaaron/sedd-small), [medium](https://huggingface.co/louaaron/sedd-medium)). However, models can also be loaded in locally (say after training). All functionality is found in ```load_model.py```.

```
# load in a pretrained model
pretrained_small_model, graph, noise = load_model("louaaron/sedd-small")
pretrained_medium_model, graph, noise = load_model("louaaron/sedd-medium")
# load in a local experiment
local_model, graph, noise = load_model("exp_local/experiment)
```

This loading gives the model, as well as the graph and noise (which are used for the loss/sampling setup).

### Run Sampling

We can run sampling using a command 

```
python run_sample.py --model_path MODEL_PATH --steps STEPS
```

We can also sample conditionally using

```
python run_sample_cond.py --model_path MODEL_PATH --step STEPS --prefix PREFIX --suffix SUFFIX
```

## Training New Models

### Run Training

We provide training code, which can be run with the command
```
python run_train.py
```
This creates a new directory `direc=exp_local/DATE/TIME` with the following structure (compatible with running sampling experiments locally)
```
├── direc
│   ├── .hydra
│   │   ├── config.yaml
│   │   ├── ...
│   ├── checkpoints
│   │   ├── checkpoint_*.pth
│   ├── checkpoints-meta
│   │   ├── checkpoint.pth
│   ├── samples
│   │   ├── iter_*
│   │   │   ├── sample_*.txt
│   ├── logs
```
Here, `checkpoints-meta` is used for reloading the run following interruptions, `samples` contains generated images as the run progresses, and `logs` contains the run output. Arguments can be added with `ARG_NAME=ARG_VALUE`, with important ones being:
```
ngpus                     the number of gpus to use in training (using pytorch DDP)
training.accum            number of accumulation steps, set to 1 for small and 2 for medium (assuming an 8x80GB node)
noise.type                one of geometric, loglinear 
graph.type                one of uniform, absorb
model                     one of small, medium
model.scale_by_sigma      set to False if graph.type=uniform (not yet configured)
```
Some example commands include
```
# training hyperparameters for SEDD absorb
python train.py noise_lib=loglinear graph.type=absorb model=medium training.accum=2
# training hyperparameters for SEDD uniform
python train.py noise_lib=geometric graph.type=uniform model=small model.scale_by_sigma=False
```

### Training with the s1K reasoning dataset

The loader supports both `s1k` (`simplescaling/s1K`) and the recommended newer
`s1k-1.1` (`simplescaling/s1K-1.1`). The latter contains the same curated set of
1,000 reasoning questions with newer R1/DeepSeek reasoning traces. It is downloaded
and cached under `data/` automatically.

```
python train.py data.train=s1k-1.1
```

The Windows config uses an effective batch size of 32 with 8 accumulation steps
(micro-batch 4) and disables memory-heavy snapshot sampling/perplexity evaluation.
If memory is still tight, use `training.batch_size=16 training.accum=8
eval.batch_size=16`, which lowers the micro-batch to 2.

Each structured record is converted to `Question / Reasoning / Answer` text before
GPT-2 tokenization. Since SEDD is an unconditional diffusion language model, this
trains on the complete sequence rather than applying supervised fine-tuning loss
only to the answer. Keep `data.valid=wikitext103`: both s1K variants contain only a
training split.

### Supervised fine-tuning from pretrained SEDD

Use the separate conditional SFT entry point to load `louaaron/sedd-small`, keep
Question tokens clean, and apply forward diffusion plus Score Entropy loss only
to Reasoning/Answer tokens:

```
python finetune_s1.py --dataset s1k-1.1 --steps 10000
```

The script makes a deterministic 800/100/100 train/validation/test split. It
writes `checkpoint.pt`, a Hugging Face-compatible final model, `splits.json`, and
test metrics under `exp_local/s1k-1.1-sft`. Resume with:

```
python finetune_s1.py --resume exp_local/s1k-1.1-sft/checkpoint.pt
```

To conditionally sample held-out questions after training and compute a simple
normalized exact-match diagnostic, add `--sample-count 10 --sample-steps 128`.

## Other Features

### SLURM compatibility

To train on slurm, simply run 
```
python train.py -m args
```

## Citation
```
@article{lou2024discrete,
  title={Discrete diffusion modeling by estimating the ratios of the data distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.16834},
  year={2024}
}
```
## Acknowledgements

This repository builds heavily off of [score sde](https://github.com/yang-song/score_sde_pytorch), [plaid](https://github.com/igul222/plaid), and [DiT](https://github.com/facebookresearch/DiT).

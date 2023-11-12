# LoRA inspector

<!--toc:start-->

- [LoRA inspector](#lora-inspector)
  - [Install](#install)
  - [Usage](#usage)
    - [Inspect](#inspect)
    - [Save meta](#save-meta)
    - [Average weights](#average-weights)
    - [Tag frequency](#tag-frequency)
    - [Dataset](#dataset)
    - [Definition](#definition)
  - [Update metadata](#update-metadata)
    - [Usage](#usage)
  - [Changelog](#changelog)
  - [Development](#development)
  - [Future](#future)
  - [Reference](#reference)
  <!--toc:end-->

![lora-inspector](https://user-images.githubusercontent.com/15027/230981999-1af9ec4e-4c05-40bc-a10a-b825c73b1013.png)

Inspect LoRA files for meta info (and hopefully quantitative analysis of the
LoRA weights).

- view training parameters
- extract metadata to be stored (we can store it in JSON currently)
- only `safetensors` are supported (want to support all LoRA files)
- only metadata from kohya-ss LoRA (want to parse all metadata in LoRA files)

---

_NOTE_ this is a work in progress and not meant for production use. _NOTE_

---

## Install

Clone this repo or download the python script file.

Requires dependencies:

```
torch
safetensors
tqdm
```

Can install them one of the following:

- Add this script to your training directory and use the virtual environment
  (`venv`). **RECOMMENDED**
- Make/use with a venv/conda
- `pip install safetensors tqdm` (See
  [Get started](https://pytorch.org/get-started/locally/) for instructions on
  how to install PyTorch)

## Usage

### Inspect

```bash
$ python lora-inspector.py --help
usage: lora-inspector.py [-h] [-s] [-w] [-t] [-d] lora_file_or_dir

positional arguments:
  lora_file_or_dir  Directory containing the lora files

options:
  -h, --help        show this help message and exit
  -s, --save_meta   Should we save the metadata to a file?
  -w, --weights     Show the average magnitude and strength of the weights
  -t, --tags        Show the most common tags in the training set
  -d, --dataset     Show the dataset metadata including directory names and number of images
```

You can add a directory or file:

```bash
$ python lora-inspector.py /mnt/900/training/sets/landscape-2023-11-06-200718-e4d7120b -w
/mnt/900/training/sets/landscape-2023-11-06-200718-e4d7120b/landscape-2023-11-06-200718-e4d7120b-000015.safetensors
Date: 2023-11-06T20:16:34 Title: landscape
License: CreativeML Open RAIL-M Author: rockerBOO
Description: High quality landscape photos
Resolution: 512x512 Architecture: stable-diffusion-v1/lora
Network Dim/Rank: 16.0 Alpha: 8.0 Dropout: 0.3 dtype: torch.float32
Module: networks.lora : {'block_dims': '4,4,4,4,4,4,4,4,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8', 'block_alphas': '16,16,16,16,16,16,16,16,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32', 'block_dropout': '0.01, 0.010620912260804992, 0.01248099020159499, 0.015572268683063176, 0.01988151037617019, 0.02539026244641935, 0.032074935571726845, 0.03990690495552037, 0.04885263290251277, 0.058873812432261884, 0.0699275313155418, 0.08196645583109653, 0.09493903345590124, 0.10878971362098, 0.12345918558747097, 0.13888463242431537, 0.155, 0.17173627983648962, 0.18902180461412393, 0.20678255506208312, 0.22494247692026895, 0.2434238066153228, 0.26214740425618505, 0.2810330925232585', 'dropout': 0.3}
Learning Rate (LR): 2e-06 UNet LR: 1.0 TE LR: 1.0
Optimizer: prodigyopt.prodigy.Prodigy(weight_decay=0.1,betas=(0.9, 0.9999),d_coef=1.5,use_bias_correction=True)
Scheduler: cosine  Warmup steps: 0
Epoch: 15 Batches per epoch: 57 Gradient accumulation steps: 24
Train images: 57 Regularization images: 0
Noise offset: 0.05 Adaptive noise scale: 0.01 IP noise gamma: 0.1  Multires noise discount: 0.3
Min SNR gamma: 5.0 Zero terminal SNR: True Debiased Estimation: True
UNet weight average magnitude: 0.7865518983141094
UNet weight average strength: 0.00995593195090544
No Text Encoder found in this LoRA
----------------------
/mnt/900/training/sets/landscape-2023-11-06-200718-e4d7120b/landscape-2023-11-06-200718-e4d7120b.safetensors
Date: 2023-11-06T20:27:12 Title: landscape
License: CreativeML Open RAIL-M Author: rockerBOO
Description: High quality landscape photos
Resolution: 512x512 Architecture: stable-diffusion-v1/lora
Network Dim/Rank: 16.0 Alpha: 8.0 Dropout: 0.3 dtype: torch.float32
Module: networks.lora : {'block_dims': '4,4,4,4,4,4,4,4,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8', 'block_alphas': '16,16,16,16,16,16,16,16,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32', 'block_dropout': '0.01, 0.010620912260804992, 0.01248099020159499, 0.015572268683063176, 0.01988151037617019, 0.02539026244641935, 0.032074935571726845, 0.03990690495552037, 0.04885263290251277, 0.058873812432261884, 0.0699275313155418, 0.08196645583109653, 0.09493903345590124, 0.10878971362098, 0.12345918558747097, 0.13888463242431537, 0.155, 0.17173627983648962, 0.18902180461412393, 0.20678255506208312, 0.22494247692026895, 0.2434238066153228, 0.26214740425618505, 0.2810330925232585', 'dropout': 0.3}
Learning Rate (LR): 2e-06 UNet LR: 1.0 TE LR: 1.0
Optimizer: prodigyopt.prodigy.Prodigy(weight_decay=0.1,betas=(0.9, 0.9999),d_coef=1.5,use_bias_correction=True)
Scheduler: cosine  Warmup steps: 0
Epoch: 30 Batches per epoch: 57 Gradient accumulation steps: 24
Train images: 57 Regularization images: 0
Noise offset: 0.05 Adaptive noise scale: 0.01 IP noise gamma: 0.1  Multires noise discount: 0.3
Min SNR gamma: 5.0 Zero terminal SNR: True Debiased Estimation: True
UNet weight average magnitude: 0.8033398082829257
UNet weight average strength: 0.010114916750103732
No Text Encoder found in this LoRA
----------------------
```

```bash
$ python lora-inspector.py /mnt/900/lora/testing/landscape-2023-11-06-200718-e4d7120b.safetensors
/mnt/900/lora/testing/landscape-2023-11-06-200718-e4d7120b.safetensors
Date: 2023-11-06T20:27:12 Title: landscape
License: CreativeML Open RAIL-M Author: rockerBOO
Description: High quality landscape photos
Resolution: 512x512 Architecture: stable-diffusion-v1/lora
Network Dim/Rank: 16.0 Alpha: 8.0 Dropout: 0.3 dtype: torch.float32
Module: networks.lora : {'block_dims': '4,4,4,4,4,4,4,4,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8', 'block_alphas': '16,16,16,16,16,16,16,16,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32', 'block_dropout': '0.01, 0.010620912260804992, 0.01248099020159499, 0.015572268683063176, 0.01988151037617019, 0.02539026244641935, 0.032074935571726845, 0.03990690495552037, 0.04885263290251277, 0.058873812432261884, 0.0699275313155418, 0.08196645583109653, 0.09493903345590124, 0.10878971362098, 0.12345918558747097, 0.13888463242431537, 0.155, 0.17173627983648962, 0.18902180461412393, 0.20678255506208312, 0.22494247692026895, 0.2434238066153228, 0.26214740425618505, 0.2810330925232585', 'dropout': 0.3}
Learning Rate (LR): 2e-06 UNet LR: 1.0 TE LR: 1.0
Optimizer: prodigyopt.prodigy.Prodigy(weight_decay=0.1,betas=(0.9, 0.9999),d_coef=1.5,use_bias_correction=True)
Scheduler: cosine  Warmup steps: 0
Epoch: 30 Batches per epoch: 57 Gradient accumulation steps: 24
Train images: 57 Regularization images: 0
Noise offset: 0.05 Adaptive noise scale: 0.01 IP noise gamma: 0.1  Multires noise discount: 0.3
Min SNR gamma: 5.0 Zero terminal SNR: True Debiased Estimation: True
UNet weight average magnitude: 0.8033398082829257
UNet weight average strength: 0.010114916750103732
No Text Encoder found in this LoRA
----------------------
```

### Save meta

We also have support for saving the meta that is extracted and converted from
strings. We can then save those to a JSON file. These will save the metadata
into `meta/alorafile.safetensors-{session_id}.json` in the current working
directory.

```bash
$ python lora-inspector.py ~/loras/alorafile.safetensors --save_meta
```

```bash
$ python lora-inspector.py /mnt/900/training/cyberpunk-anime-21-min-snr/unet-1.15-te-1.15-noise-0.1-steps--linear-DAdaptation-networks.lora/last.safetensors --save_meta
/mnt/900/training/cyberpunk-anime-21-min-snr/unet-1.15-te-1.15-noise-0.1-steps--linear-DAdaptation-networks.lora/last.safetensors
train images: 1005 regularization images: 32000
learning rate: 1.15 unet: 1.15 text encoder: 1.15
epoch: 1 batches: 2025
optimizer: dadaptation.dadapt_adam.DAdaptAdam lr scheduler: linear
network dim/rank: 8.0 alpha: 4.0 module: networks.lora
----------------------
```

### Average weights

Find the average magnitude and average strength of your weights. Compare these
with other LoRAs to see how powerful or not so powerful your weights are. _NOTE_
Weights shown are not conclusive to a good value. They are an initial example.

```bash
$ python lora-inspector.py /mnt/900/lora/studioGhibliStyle_offset.safetensors -w
UNet weight average magnitude: 4.299801171795097
UNet weight average strength: 0.01127891692482733
Text Encoder weight average magnitude: 3.128134997225176
Text Encoder weight average strength: 0.00769676965767913
```

### Tag frequency

Shows the frequency of a tag (words separated by commas). Trigger words are
generally the most frequent, as they would use that word across the whole
training dataset.

```
$ python lora-inspector.py -t /mnt/900/lora/booscapes.safetensors
...
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
Tags
-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
4k photo”                         23
spectacular mountains             17
award winning nature photo        16
ryan dyar                         14
image credit nasa nat geo         11
sunset in a valley                11
garden                            10
british columbia                  10
dramatic autumn landscape         10
autumn mountains                  10
an amazing landscape image        10
austria                           9
nature scenery                    9
pristine water                    9
boreal forest                     9
scenic view of river              9
alpes                             9
mythical floral hills             8
misty environment                 8
a photo of a lake on a sunny day  8
majestic beautiful world          8
breathtaking stars                8
lush valley                       7
dramatic scenery                  7
solar storm                       7
siberia                           7
cosmic skies                      7
dolomites                         7
oregon                            6
landscape photography 4k          6
very long spires                  6
beautiful forests and trees       6
wildscapes                        6
mountain behind meadow            6
colorful wildflowers              6
photo of green river              6
beautiful night sky               6
switzerland                       6
natural dynamic range color       6
middle earth                      6
jessica rossier color scheme      6
arizona                           6
enchanting and otherworldly       6
```

### Dataset

A pretty basic view of the dataset with the directories and number of images.

```
$ python lora-inspector.py -d /mnt/900/lora/booscapes.safetensors
Dataset dirs: 2
    [source] 50 images
    [p7] 4 images
```

### Definition

- epoch: an epoch is seeing the entire dataset once
- Batches per epoch: how many batches per each epoch (does not include gradient
  accumulation steps)
- Gradient accumulation steps: gradient accumulation steps
- Train images: number of training images you have
- Regularization images: number of regularization images
- Scheduler: the learning rate scheduler (cosine, cosine_with_restart, linear,
  constant, …)
- Optimizer: the optimizer (Adam, Prodigy, DAdaptation, Lion, …)
- Network dim/rank: the rank of the LoRA network
- Alpha: the alpha to the rank of the LoRA network
- Module: the python module that created the network
- Noise offset: noise offset option
- Adaptive noise scale: adaptive noise scale
- IP noise gamma: Input Perturbation noise gamma
  [Input Perturbation Reduces Exposure Bias in Diffusion Models](https://arxiv.org/abs/2301.11706)

  - > …we propose a very simple but effective training regularization,
    > consisting in perturbing the ground truth samples to simulate the
    > inference time prediction errors.

- multires noise discount: multires noise discount (See
  [Multi-Resolution Noise for Diffusion Model Training](https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2))
- multires noise scale: multires noise scale

- average magnitude: square each weight, add them up, get the square root
- average strength: abs each weight, add them up, get average
- debiased estimation loss:
  [Debias the Training of Diffusion Models](https://arxiv.org/abs/2310.08442)

## Update metadata

Simple script to update your metadata values. Helpful for changing
`ss_output_name` for applications that use this value to set a good name for it.

To see your current metadata values, save the metadata using
`lora-inspector.py --save_meta ...` and inspect the JSON file.

```
$ python update_metadata.py --help
usage: update_metadata.py [-h] [--key KEY] [--value VALUE] safetensors_file

positional arguments:
  safetensors_file

options:
  -h, --help        show this help message and exit
  --key KEY         Key to change in the metadata
  --value VALUE     Value to set to the metadata
```

### Usage

```
$ python update_metadata.py /mnt/900/lora/testing/armored-core-2023-08-02-173642-ddb4785e.safetensors --key ss_output_name --value mechBOO_v2
Updated ss_output_name with mechBOO_v2
Saved to /mnt/900/lora/testing/armored-core-2023-08-02-173642-ddb4785e.safetensors
```

## Changelog

- 2023-11-11 — Add debiased estimation loss, dtype (precision)
- 2023-10-27 — Add IP noise gamma
- 2023-08-27 — Add max_grad_norm, scale weight norms, gradient accumulation
  steps, dropout, and datasets
- 2023-08-08 — Add simple metadata updater script
- 2023-07-31 — Add SDXL support
- 2023-07-17 — Add network dropout, scale weight norms, adaptive noise scale,
  and steps
- 2023-07-06 — Add Tag Frequency
- 2023-04-12 — Add gradient norm, gradient checkpoint metadata
- 2023-04-03 — Add clip_skip, segment off LoCon/conv layers in average weights
- 2023-04-03 — Add noise_offset, min_snr_gamma (when added to kohya-ss), and
  network_args (for LoCon values)
- 2023-04-02 — Add `--weights` which allows you to see the average magnitude and
  strength of your LoRA UNet and Text Encoder weights.

## Development

Formatted using [`black`](https://github.com/psf/black).

## Future

What else do you want to see? Make an issue or a PR.

Use cases/ideas that this can expand into:

- Extract metadata from LoRA files to be used elsewhere
- Put the metadata into a database or search engine to find specific trainings
- Find possible issues with the training due to the metadata
- Compare LoRA files together

## Reference

- https://github.com/Zyin055/Inspect-Embedding-Training
- https://github.com/kohya-ss/sd-scripts

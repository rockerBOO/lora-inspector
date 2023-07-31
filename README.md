# LoRA inspector

<!--toc:start-->

- [LoRA inspector](#lora-inspector)
  - [Install](#install)
  - [Usage](#usage)
    - [Inspect](#inspect)
    - [Save meta](#save-meta)
    - [Average weights](#average-weights)
    - [Tag Frequency](#tag-frequency)
    - [Definition](#definition)
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

- `pip install torch safetensors tqdm`,
- make/use with a venv/conda
- add this script to your training directory (to access the dependencies).

## Usage

### Inspect

```bash
$ python lora-inspector.py --help
usage: lora-inspector.py [-h] [-s] [-w] [-t] lora_file_or_dir

positional arguments:
  lora_file_or_dir  Directory containing the lora files

options:
  -h, --help        show this help message and exit
  -s, --save_meta   Should we save the metadata to a file?
  -w, --weights     Show the average magnitude and strength of the weights
  -t, --tags        Show the most common tags in the training set
```

You can add a directory or file:

```bash
$ python lora-inspector.py /mnt/900/training/sets/cyberpunk-anime-24-2023-04-03-13:23:36-e94c9dd8 -w
UNet weight average magnitude: 1.974654662580427
UNet weight average strength: 0.015630576804489364
Text Encoder weight average magnitude: 0.9055601234903398
Text Encoder weight average strength: 0.009029422735480932
/mnt/900/training/sets/cyberpunk-anime-24-2023-04-03-13:23:36-e94c9dd8/last.safetensors
train images: 201 regularization images: 1600
learning rate: 0.01 unet: 0.001 text encoder: 0.0001 scheduler: cosine
epoch: 2 batches: 402 optimizer: torch.optim.adamw.AdamW
network dim/rank: 8.0 alpha: 4.0 module: networks.lora {'conv_dim': '32', 'conv_alpha': '0.3'}
noise_offset: 0.1 min_snr_gamma: 1.0
----------------------
UNet weight average magnitude: 1.9316962578548313
UNet weight average strength: 0.015348419610733969
Text Encoder weight average magnitude: 0.9034643649275405
Text Encoder weight average strength: 0.009012302642521
/mnt/900/training/sets/cyberpunk-anime-24-2023-04-03-13:23:36-e94c9dd8/epoch-000001.safetensors
train images: 201 regularization images: 1600
learning rate: 0.01 unet: 0.001 text encoder: 0.0001 scheduler: cosine
epoch: 1 batches: 402 optimizer: torch.optim.adamw.AdamW
network dim/rank: 8.0 alpha: 4.0 module: networks.lora {'conv_dim': '32', 'conv_alpha': '0.3'}
noise_offset: 0.1 min_snr_gamma: 1.0
```

```bash
$ python lora-inspector.py /mnt/900/training/cyberpunk-anime-21-min-snr/unet-1.15-te-1.15-noise-0.1-steps--linear-DAdaptation-networks.lora/last.safetensors
/mnt/900/training/cyberpunk-anime-21-min-snr/unet-1.15-te-1.15-noise-0.1-steps--linear-DAdaptation-networks.lora/last.safetensors
train images: 1005 regularization images: 32000
learning rate: 1.15 unet: 1.15 text encoder: 1.15
epoch: 1 batches: 2025
optimizer: dadaptation.dadapt_adam.DAdaptAdam lr scheduler: linear
network dim/rank: 8.0 alpha: 4.0 module: networks.lora
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

### Tag Frequency

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

### Definition

- epoch: an epoch is seeing the entire dataset once
- batches: how many batches per each epoch (does not include gradient
  accumulation steps)
- train images: number of training images you have
- regularization images: number of regularization images
- scheduler: the learning rate scheduler.
- optimizer: the optimizer
- network dim/rank: the rank of the LoRA network
- alpha: the alpha to the rank of the LoRA network
- module: which python module was used to to create the network (includes module
  arguments)
- noise offset: noise offset option
- adaptive noise scale: adapative noise scale
- multires noise discount: multires noise discount
- multires noise scale: multires noise scale

- average magnitude: square each weight, add them up, get the square root
- average strength: abs each weight, add them up, get average

## Changelog

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

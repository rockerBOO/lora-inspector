# LoRA inspector

<!--toc:start-->
- [LoRA inspector](#lora-inspector)
  - [Install](#install)
  - [Usage](#usage)
    - [Inspect](#inspect)
    - [Save meta](#save-meta)
    - [Average weights](#average-weights)
  - [Changelog](#changelog)
  - [Development](#development)
  - [Future](#future)
  - [Reference](#reference)
<!--toc:end-->

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
$ python lora-inspector.py -h
usage: lora-inspector.py [-h] [-s] [-w] lora_file_or_dir

positional arguments:
  lora_file_or_dir  Directory containing the lora files

options:
  -h, --help        show this help message and exit
  -s, --save_meta   Should we save the metadata to a file?
  -w, --weights     Find the average weights
```

You can add a directory or file:

```bash
$ python lora-inspector.py /mnt/900/training/cyberpunk-anime-21-min-snr
0it [00:00, ?it/s]/mnt/900/training/cyberpunk-anime-21-min-snr/unet-1e-4-te-5e-5-noise-0.1-steps--cosine-Lion-networks.lora/last.safetensors
train images: 1005 regularization images: 32000
learning rate: 0.001 unet: 0.0001 text encoder: 5e-05
epoch: 1 batches: 2025
optimizer: lion_pytorch.lion_pytorch.Lion lr scheduler: cosine
network dim/rank: 8.0 alpha: 4.0 module: networks.lora
----------------------
1it [00:00,  2.04it/s]/mnt/900/training/cyberpunk-anime-21-min-snr/unet-1-te-1-noise-0.1-steps--linear-AdaFactor-networks.lora/last.safetensors
train images: 1005 regularization images: 32000
learning rate: None unet: None text encoder: None
epoch: 1 batches: 2025
optimizer: transformers.optimization.Adafactor(relative_step=True) lr scheduler: adafactor:1.0
network dim/rank: 8.0 alpha: 4.0 module: networks.lora
----------------------
/mnt/900/training/cyberpunk-anime-21-min-snr/unet-1e-4-te-5e-5-noise-0.1-steps--linear-AdamW-networks.lora/last.safetensors
train images: 1005 regularization images: 32000
learning rate: 0.0001 unet: 0.0001 text encoder: 5e-05
epoch: 1 batches: 2025
optimizer: torch.optim.adamw.AdamW lr scheduler: linear
network dim/rank: 8.0 alpha: 4.0 module: networks.lora
----------------------
/mnt/900/training/cyberpunk-anime-21-min-snr/unet-1e-4-te-5e-5-noise-0.1-steps-epoch--2-cosine-AdamW-networks.lora/last.safetensors
train images: 1005 regularization images: 32000
learning rate: 0.0001 unet: 0.0001 text encoder: 5e-05
epoch: 2 batches: 2025
optimizer: torch.optim.adamw.AdamW lr scheduler: cosine
network dim/rank: 8.0 alpha: 4.0 module: networks.lora
----------------------
/mnt/900/training/cyberpunk-anime-21-min-snr/unet-1e-4-te-5e-5-noise-0.1-steps-epoch--2-cosine-AdamW-networks.lora/epoch-000001.safetensors
train images: 1005 regularization images: 32000
learning rate: 0.0001 unet: 0.0001 text encoder: 5e-05
epoch: 1 batches: 2025
optimizer: torch.optim.adamw.AdamW lr scheduler: cosine
network dim/rank: 8.0 alpha: 4.0 module: networks.lora
----------------------
/mnt/900/training/cyberpunk-anime-21-min-snr/unet-1.15-te-1.15-noise-0.1-steps--linear-DAdaptation-networks.lora/last.safetensors
train images: 1005 regularization images: 32000
learning rate: 1.15 unet: 1.15 text encoder: 1.15
epoch: 1 batches: 2025
optimizer: dadaptation.dadapt_adam.DAdaptAdam lr scheduler: linear
network dim/rank: 8.0 alpha: 4.0 module: networks.lora
----------------------
/mnt/900/training/cyberpunk-anime-21-min-snr/unet-1.0-te-1.0-noise-0.1-steps--linear-DAdaptation-networks.lora/last.safetensors
train images: 1005 regularization images: 32000
learning rate: 1.0 unet: 1.0 text encoder: 1.0
epoch: 1 batches: 2025
optimizer: dadaptation.dadapt_adam.DAdaptAdam lr scheduler: linear
network dim/rank: 8.0 alpha: 4.0 module: networks.lora
----------------------
7it [00:00, 14.12it/s]
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
into `meta/alorafile.safetensors-{session_id}.json` in the meta directory.

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

## Changelog

- 2023-04-02 - Added `--weights` which allows you to see the average magnitude
  and strength of your LoRA UNet and Text Encoder weights.

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

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

- 2023-04-03 — Add clip_skip, segment off LoCon/conv layers in average weights
- 2023-04-03 — Add noise_offset, min_snr_gamma (when added to kohya-ss), and network_args (for LoCon values)
- 2023-04-02 — Add `--weights` which allows you to see the average magnitude
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

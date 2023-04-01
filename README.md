# LoRA inspector

<!--toc:start-->

- [LoRA inspector](#lora-inspector)
  - [Install](#install)
  - [Usage](#usage)
    - [Inspect](#inspect)
    - [Save meta](#save-meta)
  - [Development](#development)
  - [Future](#future)
  - [Reference](#reference)
  <!--toc:end-->

Inspect LoRA files for meta info (and hopefully quantitative analysis of the
LoRA weights)

- view training parameters
- extract metadata to be stored (we can store it in JSON currently)

---

_NOTE_ this is a work in progress and not meant for production use. _NOTE_
 
---

## Install

Clone this repo or download the python script file.

Requires dependencies:

```
torch
safetensors
```

Can install them one of the following:

- `pip install torch safetensors`,
- make/use with a venv/conda
- add this script to your training directory (to access the dependencies).

## Usage

### Inspect

```bash
$ python lora-inspector.py
usage: lora-inspector.py [-h] [-s] lora_file_or_dir
lora-inspector.py: error: the following arguments are required: lora_file_or_dir
```

You can add a directory or file:

```bash
$ python lora-inspector.py ~/loras/
```

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
$ python lora-inspector.py ~/loras/alorafile.safetensors
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

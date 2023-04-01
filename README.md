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

Inspect LoRA files for meta info (and hopefully quantitative analysis of the LoRA weights)

- view training parameters
- extract metadata to be stored (we can store it in json currently)

## Install

Clone this repo or download the python script file.

Requires dependencies:

```
torch
safetensors
```

Can install them using `pip install torch safetensors`, make a venv/conda with them, or add this script to your training directory (to access the dependencies).

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
$ python lora-inspector.py ~/loras/alorafile.safetensors
```

### Save meta

We also have support for saving the meta that is extracted and converted from strings. We can then save those to a JSON file. These will save the metadata into `meta/alorafile.safetensors-{session_id}.json` in the meta directory. 

```bash
$ python lora-inspector.py ~/loras/alorafile.safetensors --save_meta
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

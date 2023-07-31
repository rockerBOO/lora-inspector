import argparse
import json
import math
import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Union

import torch
from safetensors import safe_open
from torch import Tensor


def to_datetime(str: str):
    return datetime.fromtimestamp(float(str))


class NameSpace(argparse.ArgumentParser):
    lora_file_or_dir: str
    save_meta: bool
    weights: bool
    tags: bool


parsers: dict[str, Callable] = {
    "int": int,
    "float": float,
    "json": json.loads,
    "bool": bool,
    "dt": to_datetime,
    "str": str,
}


schema: dict[str, str] = {
    "ss_learning_rate": "float",
    "ss_max_bucket_reso": "int",
    "ss_text_encoder_lr": "float",
    "ss_epoch": "int",
    "ss_unet_lr": "float",
    "ss_seed": "int",
    "ss_max_train_steps": "int",
    "ss_sd_model_name": "str",
    "ss_new_vae_hash": "str",
    "ss_resolution": "str",
    "ss_full_fp16": "bool",
    "ss_vae_hash": "str",
    "ss_gradient_checkpoint": "bool",
    "ss_output_name": "bool",
    "ss_bucket_info": "json",
    "sshs_model_hash": "str",
    "sshs_legacy_hash": "str",
    "ss_caption_dropout_rate": "float",
    "ss_caption_dropout_every_n_epochs": "int",
    "ss_caption_tag_dropout_rate": "float",
    "ss_sd_scripts_commit_hash": "str",
    "ss_gradient_checkpointing": "bool",
    "ss_training_finished_at": "dt",
    "ss_vae_name": "str",
    "ss_total_batch_size": "int",
    "ss_batch_size_per_device": "int",
    "ss_color_aug": "bool",
    "ss_flip_aug": "bool",
    "ss_lr_warmup_steps": "int",
    "ss_lr_scheduler": "str",
    "ss_lr_scheduler_power": "float",
    "ss_num_epochs": "int",
    "ss_mixed_precision": "str",
    "ss_shuffle_caption": "bool",
    "ss_training_started_at": "dt",
    "ss_v2": "bool",
    "ss_keep_tokens": "bool",
    "ss_random_crop": "bool",
    "ss_cache_latents": "bool",
    "ss_gradient_accumulation_steps": "int",
    "ss_clip_skip": "int",
    "ss_dataset_dirs": "json",
    "ss_training_comment": "str",
    "ss_network_module": "str",
    "ss_network_args": "json",
    "ss_network_alpha": "float",
    "ss_network_dim": "float",
    "ss_reg_dataset_dirs": "json",
    "ss_num_batches_per_epoch": "int",
    "ss_num_reg_images": "int",
    "ss_max_token_length": "int",
    "ss_sd_new_model_hash": "int",
    "ss_face_crop_aug_range": "str",
    "ss_min_bucket_reso": "int",
    "ss_bucket_no_upscale": "bool",
    "ss_prior_loss_weight": "float",
    "ss_enable_bucket": "bool",
    "ss_num_train_images": "int",
    "ss_lowram": "bool",
    "ss_optimizer": "str",
    "ss_tag_frequency": "json",
    "ss_session_id": "str",
    "ss_max_grad_norm": "float",
    "ss_noise_offset": "float",
    "ss_multires_noise_discount": "float",
    "ss_multires_noise_iterations": "float",
    "ss_min_snr_gamma": "float",
    "ss_sd_model_hash": "str",
    "ss_new_sd_model_hash": "str",
    "ss_datasets": "json",
    "ss_loss_func": "str",
    "ss_network_dropout": "float",
    "ss_scale_weight_norms": "float",
    "ss_adaptive_noise_scale": "float",
    "ss_steps": "int",
    "ss_base_model_version": "str",
    "ss_zero_terminal_snr": "bool",
}


def parse_item(key: str, value: str) -> int | float | bool | datetime | str | None:
    if key not in schema:
        print(f"invalid key in schema {key}")
        print(value)
        return value

    if schema[key] == "int" and value == "None":
        return None

    if schema[key] == "float" and value == "None":
        return None

    # print(key)
    return parsers[schema[key]](value)


def parse(entries: dict[str, str]):
    results = {}
    for k in entries.keys():
        v = entries[k]
        results[k] = parse_item(k, v)
    return results


def key_start_match(key, match):
    return key[0 : len(match)] == match


def key_match(key, match):
    return match in key


def avg_weights(results, name=""):
    num_results = len(results)

    avg_mag = 0
    avg_str = 0

    if num_results > 0:
        sum_mag = 0  # average magnitude
        sum_str = 0  # average strength
        for k in results.keys():
            sum_mag += get_vector_data_magnitude(results[k])
            sum_str += get_vector_data_strength(results[k])

        avg_mag = sum_mag / num_results
        avg_str = sum_str / num_results

        print(f"{name} weight average magnitude: {avg_mag}")
        print(f"{name} weight average strength: {avg_str}")

    return avg_mag, avg_str


def find_vectors_weights(vectors):
    weight = ".weight"

    unet_attn_weight_results = {}
    unet_conv_weight_results = {}
    text_encoder1_weight_results = {}
    text_encoder2_weight_results = {}

    print(f"model key count: {len(vectors.keys())}")

    for k in vectors.keys():
        unet = "lora_unet"
        if key_start_match(k, unet) or key_start_match(k, unet):
            if k.endswith(weight):
                if key_match(k, "conv"):
                    unet_conv_weight_results[k] = torch.flatten(
                        vectors.get_tensor(k)
                    ).tolist()
                else:
                    unet_attn_weight_results[k] = torch.flatten(
                        vectors.get_tensor(k)
                    ).tolist()

        # SD 1.x 2.x text encoder
        text_encoder = "lora_te_text_model_encoder_layers_"
        if key_start_match(k, text_encoder):
            if k.endswith(weight):
                text_encoder1_weight_results[k] = torch.flatten(
                    vectors.get_tensor(k)
                ).tolist()

        # SDXL text encoder 1
        text_encoder = "lora_te1_text_model_encoder_layers"
        if key_start_match(k, text_encoder):
            if k.endswith(weight):
                text_encoder1_weight_results[k] = torch.flatten(
                    vectors.get_tensor(k)
                ).tolist()

        # SDXL text encoder 2
        text_encoder = "lora_te2_text_model_encoder_layers_"
        if key_start_match(k, text_encoder):
            if k.endswith(weight):
                text_encoder2_weight_results[k] = torch.flatten(
                    vectors.get_tensor(k)
                ).tolist()

    avg_weights(unet_attn_weight_results, name="UNet")
    avg_weights(unet_conv_weight_results, name="UNet Conv")
    avg_weights(text_encoder1_weight_results, name="Text Encoder (1)")
    avg_weights(text_encoder2_weight_results, name="Text Encoder (2)")

    if len(unet_attn_weight_results) == 0 and len(unet_conv_weight_results) == 0:
        print("No UNet found in this LoRA")

    if (
        len(text_encoder1_weight_results) == 0
        and len(text_encoder2_weight_results) == 0
    ):
        print("No Text Encoder found in this LoRA")

    return {
        "unet": unet_attn_weight_results,
        "unet_conv": unet_conv_weight_results,
        "text_encoder1": text_encoder1_weight_results,
        "text_encoder2": text_encoder2_weight_results,
    }


def get_vector_data_strength(data: dict[int, Tensor]) -> float:
    value = 0
    for n in data:
        value += abs(n)

    # the average value of each vector (ignoring negative values)
    return value / len(data)


def get_vector_data_magnitude(data: dict[int, Tensor]) -> float:
    value = 0
    for n in data:
        value += pow(n, 2)
    return math.sqrt(value)


def find_safetensor_files(path: str | Path):
    return Path(path).rglob("*.safetensors")


def save_metadata(file: Path, metadata):
    dir = Path("meta/")
    if dir.is_dir() is False:
        print(f"creating directory {dir.resolve()}")
        os.mkdir(dir)

    with open(Path(dir / file.stem / ".json"), "w+") as f:
        json.dump(metadata, f, default=str)


def process_safetensor_file(file: Path, args) -> dict[str, Any]:
    with safe_open(file, framework="pt", device="cpu") as f:
        metadata = f.metadata()

        filename = os.path.basename(file)
        print(file)

        parsed = {}

        if metadata is not None:
            parsed = parse_metadata(metadata)
        else:
            parsed = {}

        parsed["file"] = file
        parsed["filename"] = filename

        if args.weights:
            find_vectors_weights(f)

        if args.tags:
            tags(parsed)

        print("----------------------")
        return parsed


def print_list(list):
    print(" ".join(list).strip(" "))


def parse_metadata(metadata):
    if "sshs_model_hash" in metadata:
        items = parse(metadata)

        # TODO if we are missing this value, they may not be saving the metadata
        # to the file or are missing key components. Should evaluate if we need
        # to do more in the case that this is missing when we get more examples
        if "ss_network_dim" not in items:
            for item in items:
                print(item)
            return items

        # print(json.dumps(items, indent=4, sort_keys=True, default=str))

        def item(items, key, name):
            if key in items and items.get(key) is not None and items.get(key) != "None":
                return f"{name}: {items.get(key, '')}"

            return ""

        print(
            f"epoch: {items['ss_epoch']} batches: {items['ss_num_batches_per_epoch']}"
        )

        print(
            f"train images: {items['ss_num_train_images']} {item(items, 'ss_num_reg_images', 'regularization images')}"
        )

        # if (
        #     items["ss_num_reg_images"] > 0
        #     and items["ss_num_reg_images"] < items["ss_num_train_images"]
        # ):
        #     print(
        #         f"Possibly not enough regularization images ({items['ss_num_reg_images']}) to training images ({items['ss_num_train_images']})."
        #     )

        print(
            f"learning rate: {items['ss_learning_rate']} unet: {items['ss_unet_lr']} text encoder: {items['ss_text_encoder_lr']}"
        )

        results = [
            item(items, "ss_lr_scheduler", "scheduler"),
            item(items, "ss_lr_scheduler_args", "scheduler args"),
        ]

        print_list(results)

        results = [
            item(items, "ss_optimizer", "optimizer"),
            item(items, "ss_optimizer_args", "optimizer_args"),
        ]

        print_list(results)

        if "loss_func" in items:
            results = [
                item(items, "ss_loss_func", "loss func"),
            ]

            print_list(results)

        print(
            f"network dim/rank: {items['ss_network_dim']} network alpha: {items['ss_network_alpha']} network module: {items['ss_network_module']} {items.get('ss_network_args')}"
        )

        results = [
            item(items, "ss_noise_offset", "noise offset"),
            item(items, "ss_adaptive_noise_scale", "adaptive noise scale"),
            item(items, "ss_multires_noise_iterations", "multires noise iterations"),
            item(items, "ss_multires_noise_discount", "multires noise discount"),
        ]

        print_list(results)

        results = [
            item(items, "ss_min_snr_gamma", "min snr gamma"),
            item(items, "ss_zero_terminal_snr", "zero terminal snr"),
            item(items, "ss_clip_skip", "clip skip"),
        ]

        print_list(results)

        return items
    else:
        print(
            "Please submit the following keys so we can get a parser made for it:",
            metadata.keys(),
        )
        return {}


def print_tags(freq):
    """
    freq: Tag frequency
    """

    print("----------------------")
    print("Tags")
    print("----------------------")

    tags = []
    longest_tag = 0
    for k in freq.keys():
        for kitem in freq[k].keys():
            if int(freq[k][kitem]) > 3:
                tags.append((kitem, freq[k][kitem]))

                if len(kitem) > longest_tag:
                    longest_tag = len(kitem)

    ordered = OrderedDict(reversed(sorted(tags, key=lambda t: t[1])))

    justify_to = longest_tag + 1 if longest_tag < 60 else 60

    for k, v in ordered.items():
        print(k.ljust(justify_to), v)


def tags(results: Union[list[dict[str, Any]], dict[str, Any]]):
    if type(results) == list:
        for result in results:
            if "ss_tag_frequency" in result:
                print_tags(result["ss_tag_frequency"])
            else:
                print("No tags found")
    elif type(results) == dict:
        if "ss_tag_frequency" in results:
            print_tags(results["ss_tag_frequency"])
        else:
            print("No tags found")


def save_meta(results: Union[list[dict[str, Any]], dict[str, Any]]):
    if type(results) == list:
        for result in results:
            # print("result", json.dumps(result, indent=4, sort_keys=True, default=str))
            if "ss_session_id" in result:
                newfile = Path(
                    "meta" / f"{str(result['filename'])}-{result['ss_session_id']}"
                )
            else:
                newfile = Path("meta" / str(result["filename"]))
            save_metadata(newfile, result)
            print(f"Metadata saved to {newfile}.json")
    else:
        if "ss_session_id" in results:
            newfile = Path(
                "meta" / str(results["filename"]) + "-" + results["ss_session_id"]
            )
        else:
            newfile = Path("meta/" + str(results["filename"]))
        save_metadata(newfile, results)
        print(f"Metadata saved to {newfile}.json")


def process(args: type[NameSpace]):
    file = Path(args.lora_file_or_dir)
    if file.is_dir():
        results = []
        files = sorted(find_safetensor_files(file))
        for path in files:
            results.append(process_safetensor_file(path, args))

        return results
    else:
        return process_safetensor_file(file, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "lora_file_or_dir", type=str, help="Directory containing the lora files"
    )

    parser.add_argument(
        "-s",
        "--save_meta",
        action="store_true",
        help="Should we save the metadata to a file?",
    )

    parser.add_argument(
        "-w",
        "--weights",
        action="store_true",
        help="Show the average magnitude and strength of the weights",
    )

    parser.add_argument(
        "-t",
        "--tags",
        action="store_true",
        help="Show the most common tags in the training set",
    )

    args = parser.parse_args(namespace=NameSpace)
    results = process(args)

    if args.save_meta:
        save_meta(results)

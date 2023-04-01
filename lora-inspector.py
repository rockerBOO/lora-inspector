from safetensors import safe_open
import json
import os
from functools import reduce
from pathlib import Path
import argparse
from tqdm import tqdm
from datetime import datetime
from typing import Callable


def to_datetime(str: str):
    return datetime.fromtimestamp(float(str))


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
    "ss_network_module": "str",
    "ss_lr_scheduler": "str",
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
    "ss_network_args": "json",
    "ss_enable_bucket": "bool",
    "ss_num_train_images": "int",
    "ss_lowram": "bool",
    "ss_optimizer": "str",
    "ss_tag_frequency": "json",
    "ss_session_id": "str",
    "ss_max_grad_norm": "float",
    "ss_noise_offset": "float",
    "ss_sd_model_hash": "str",
    "ss_new_sd_model_hash": "str",
    "ss_datasets": "json",
}


def parse_item(key: str, value: str) -> int | float | bool | datetime | str | None:
    if key not in schema:
        print(f"invalid key in schema {key}")
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


def find_safetensor_files(path):
    return Path(path).rglob("*.safetensors")


def save_metadata(file, metadata):
    dir = "meta/"
    if os.path.isdir(dir) is False:
        print(f"creating directory {dir}")
        os.mkdir(dir)

    with open(dir + os.path.basename(file) + ".json", "w+") as f:
        json.dump(metadata, f, default=str)


def process_safetensor_file(file):
    if os.path.isdir(file):
        progress = tqdm(find_safetensor_files(file))
        results = []
        for path in progress:
            results.append(process_safetensor_file(path))
            progress.update(1)

        return results
    else:
        with safe_open(file, framework="pt") as f:
            metadata = f.metadata()

            if metadata is not None:
                filename = os.path.basename(file)
                print(filename)
                parsed = parse_metadata(metadata)
                parsed["file"] = file
                parsed["filename"] = filename
                print("----------------------")
                return parsed


def parse_metadata(metadata):
    if "ss_tag_frequency" in metadata:
        items = parse(metadata)

        # print(json.dumps(items, indent=4, sort_keys = True, default=str))

        print(
            f"train images: {items['ss_num_train_images']} regularization images: {items['ss_num_reg_images']}"
        )
        if (
            items["ss_num_reg_images"] > 0
            and items["ss_num_reg_images"] < items["ss_num_train_images"]
        ):
            print("Possibly not enough regularization images to training images.")

        print(
            f"learning rate: {items['ss_learning_rate']} unet: {items['ss_unet_lr']} text encoder: {items['ss_text_encoder_lr']}"
        )

        print(
            f"epoch: {items['ss_epoch']} batches: {items['ss_num_batches_per_epoch']}"
        )
        print(
            f"optimizer: {items.get('ss_optimizer', '')} lr scheduler: {items['ss_lr_scheduler']}"
        )

        print(
            f"network dim/rank: {items['ss_network_dim']} alpha: {items['ss_network_alpha']} module: {items['ss_network_module']}"
        )

        return items

        # print(items)

        # for _group_name, tag_list in tags.items():
        #     for tag, frequency in tag_list.items():
        #         # print(frequency, "\t", tag)
        # print(metadata.get("ss_max_train_samples"), metadata.get("ss_max_train_epochs"), metadata.get("ss_network_module"))
        # print(metadata)
        # print(
        #     metadata.get("ss_num_train_images"),
        #     metadata.get("ss_unet_lr"),
        #     metadata.get("ss_text_encoder_lr"),
        # )
        #
        # num_reg_images = int(metadata.get("ss_num_reg_images"))
        # num_train_images = int(metadata.get("ss_num_train_images"))
        #
        # print(f"num reg {num_reg_images}, train {num_train_images}")
        #
        # if num_reg_images < num_train_images:
        #     print("Possibly not enough regularization images to training images")
        #
        # newfile = os.path.dirname(file) + metadata.get("ss_new_sd_model_hash") + ".json"
        # save_metadata(newfile, items)
    else:
        print(
            "Please submit the following keys so we can get a parser made for it:",
            metadata.keys(),
        )
        # save_metadata(file, metadata)
        return {}


def parse_ss(metadata):
    print(
        metadata.get("ss_num_train_images"),
        metadata.get("ss_unet_lr"),
        metadata.get("ss_text_encoder_lr"),
    )

    metadata = dict(metadata)

    return metadata


def process(args):
    file = args.lora_file_or_dir
    if os.path.isdir(file):
        progress = tqdm(find_safetensor_files(file))
        results = []
        for path in progress:
            results.append(process_safetensor_file(path))
            progress.update(1)

        return results
    else:
        return process_safetensor_file(file)


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

    args = parser.parse_args()
    results = process(args)

    # if type(results) == list:
    #     for result in results:
    #         if "filename" in result:
    #             print(result["filename"])
    # else:
    #     if "filename" in result:
    #         print(results["filename"])

    if args.save_meta:
        if type(results) == list:
            for result in results:
                # print("result", json.dumps(result, indent=4, sort_keys=True, default=str))
                if "ss_session_id" in result:
                    newfile = (
                        "meta/"
                        + str(result["filename"])
                        + "-"
                        + result["ss_session_id"]
                    )
                else:
                    newfile = "meta/" + str(result["filename"])
                print(f"newfile: {newfile}")
                save_metadata(newfile, result)
        else:
            newfile = "meta/" + results["filename"] + "-" + results["ss_session_id"]
            print(f"newfile: {newfile}")
            save_metadata(newfile, results)
    # print(results)

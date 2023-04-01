import json
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
}


def parse_item(key: str, value: str) -> int | float | bool | datetime | str | None:
    if key not in schema:
        print(f"invalid key in schema {key}")
        return value

    if schema[key] == "int" and value == "None":
        return None

    if schema[key] == "float" and value == "None":
        return None

    print(key)
    return parsers[schema[key]](value)


def parse(entries: dict[str, str]):
    results = {}
    for k in entries.keys():
        v = entries[k]
        results[k] = parse_item(k, v)
    return results


# if __name__ == "__main__":
#     with open("meta/last.safetensors.json") as f:
#         loaded: dict[str, str] = json.load(f)
#         results = {}
#         for (k, v) in parse(loaded):
#             # v = loaded[k]
#             results[k] = v
#
#         print(json.dumps(results, default=str))

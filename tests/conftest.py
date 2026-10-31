import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def inspector():
    """Load lora-inspector.py (the hyphen prevents a normal import)."""
    spec = importlib.util.spec_from_file_location(
        "lora_inspector_script", ROOT / "lora-inspector.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def args():
    from argparse import Namespace

    return Namespace(
        lora_file_or_dir="",
        save_meta=False,
        weights=False,
        tags=False,
        dataset=False,
    )


@pytest.fixture
def metadata():
    return {
        "sshs_model_hash": "abc123",
        "ss_network_dim": "16",
        "ss_network_alpha": "8",
        "ss_network_module": "networks.lora",
        "ss_learning_rate": "0.0001",
        "ss_optimizer": "AdamW8bit",
        "ss_lr_scheduler": "cosine",
        "ss_epoch": "10",
        "ss_num_train_images": "42",
        "ss_session_id": "1234",
        "ss_tag_frequency": '{"img": {"cat": 5, "dog": 9}}',
        "ss_dataset_dirs": '{"img": {"n_repeats": 1, "img_count": 42}}',
    }


@pytest.fixture
def tensors():
    return {
        "lora_unet_down_blocks_0_attn.lora_down.weight": torch.ones(2, 2),
        "lora_unet_down_blocks_0_conv.lora_down.weight": torch.ones(2, 2),
        "lora_te_text_model_encoder_layers_0_q.lora_down.weight": torch.ones(2, 2),
    }


@pytest.fixture
def lora_file(tmp_path, tensors, metadata):
    path = tmp_path / "test_lora.safetensors"
    save_file(tensors, str(path), metadata=metadata)
    return path


@pytest.fixture
def lora_file_no_metadata(tmp_path, tensors):
    path = tmp_path / "bare.safetensors"
    save_file(tensors, str(path))
    return path

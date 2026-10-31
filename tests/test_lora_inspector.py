import json
import subprocess
import sys
from datetime import datetime

import pytest
from safetensors import safe_open

from conftest import ROOT


class TestParseItem:
    def test_int(self, inspector):
        assert inspector.parse_item("ss_epoch", "10") == 10

    def test_float(self, inspector):
        assert inspector.parse_item("ss_learning_rate", "0.0001") == 0.0001

    def test_none_string_becomes_none(self, inspector):
        assert inspector.parse_item("ss_epoch", "None") is None
        assert inspector.parse_item("ss_learning_rate", "None") is None

    def test_dynamic_network_dim(self, inspector):
        assert inspector.parse_item("ss_network_dim", "Dynamic") == "Dynamic"
        assert inspector.parse_item("ss_network_alpha", "Dynamic") == "Dynamic"

    def test_json(self, inspector):
        assert inspector.parse_item("ss_tag_frequency", '{"a": {"b": 1}}') == {
            "a": {"b": 1}
        }

    def test_datetime(self, inspector):
        assert isinstance(
            inspector.parse_item("ss_training_started_at", "1700000000.0"), datetime
        )

    def test_unknown_key_passes_through(self, inspector):
        assert inspector.parse_item("not_a_real_key", "value") == "value"

    def test_parse_all_entries(self, inspector):
        assert inspector.parse({"ss_epoch": "3", "ss_seed": "7"}) == {
            "ss_epoch": 3,
            "ss_seed": 7,
        }


class TestHelpers:
    def test_get_item_present(self, inspector):
        assert inspector.get_item({"k": 5}, "k", "Name") == "Name: 5"

    @pytest.mark.parametrize("items", [{}, {"k": None}, {"k": "None"}])
    def test_get_item_missing_or_none(self, inspector, items):
        assert inspector.get_item(items, "k", "Name") == ""

    def test_print_list_skips_empty(self, inspector, capsys):
        inspector.print_list(["A: 1", "", "B: 2"])
        assert capsys.readouterr().out.strip() == "A: 1  B: 2"

    def test_print_list_all_empty_prints_nothing(self, inspector, capsys):
        inspector.print_list(["", ""])
        assert capsys.readouterr().out == ""

    def test_key_matching(self, inspector):
        assert inspector.key_start_match("lora_unet_x", "lora_unet")
        assert not inspector.key_start_match("x_lora_unet", "lora_unet")
        assert inspector.key_match("a_conv_b", "conv")

    def test_vector_magnitude_and_strength(self, inspector):
        assert inspector.get_vector_data_magnitude([3.0, 4.0]) == 5.0
        assert inspector.get_vector_data_strength([-1.0, 3.0]) == 2.0

    def test_find_safetensor_files_recurses(self, inspector, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.safetensors").touch()
        (tmp_path / "sub" / "b.safetensors").touch()
        (tmp_path / "ignore.txt").touch()
        found = {p.name for p in inspector.find_safetensor_files(tmp_path)}
        assert found == {"a.safetensors", "b.safetensors"}


class TestProcessFile:
    def test_parses_metadata(self, inspector, lora_file, args):
        result = inspector.process_safetensor_file(lora_file, args)
        assert result["ss_network_dim"] == 16
        assert result["ss_network_alpha"] == 8
        assert result["ss_epoch"] == 10
        assert result["filename"] == "test_lora.safetensors"
        assert result["file"] == lora_file

    def test_detects_dtype(self, inspector, lora_file, args):
        assert inspector.process_safetensor_file(lora_file, args)["dtype"] == "F32"

    def test_prints_summary(self, inspector, lora_file, args, capsys):
        inspector.process_safetensor_file(lora_file, args)
        out = capsys.readouterr().out
        assert "Network Dim/Rank: 16" in out
        assert "Optimizer: AdamW8bit" in out
        assert "Train images: 42" in out

    def test_no_metadata(self, inspector, lora_file_no_metadata, args):
        result = inspector.process_safetensor_file(lora_file_no_metadata, args)
        assert result["filename"] == "bare.safetensors"
        assert "ss_network_dim" not in result

    def test_unrecognized_metadata_returns_empty(self, inspector, args, capsys):
        assert inspector.parse_metadata({"foo": "bar"}, args) == {}
        assert "Please submit the following keys" in capsys.readouterr().out

    def test_dataset_flag(self, inspector, lora_file, args, capsys):
        args.dataset = True
        inspector.process_safetensor_file(lora_file, args)
        out = capsys.readouterr().out
        assert "Dataset dirs: 1" in out
        assert "[img] 42 images" in out

    def test_weights_flag(self, inspector, lora_file, args, capsys):
        args.weights = True
        inspector.process_safetensor_file(lora_file, args)
        out = capsys.readouterr().out
        # 2x2 tensor of ones: magnitude sqrt(4) = 2, strength 1
        assert "UNet weight average magnitude: 2.0" in out
        assert "UNet Conv weight average strength: 1.0" in out
        assert "Text Encoder (1) weight average magnitude: 2.0" in out
        assert "No Text Encoder found" not in out

    def test_find_vectors_weights_groups_keys(self, inspector, lora_file):
        with safe_open(lora_file, framework="pt", device="cpu") as f:
            groups = inspector.find_vectors_weights(f)
        assert len(groups["unet"]) == 1
        assert len(groups["unet_conv"]) == 1
        assert len(groups["text_encoder1"]) == 1
        assert groups["text_encoder2"] == {}


class TestProcessDirectory:
    def test_directory_returns_list(self, inspector, lora_file, args):
        args.lora_file_or_dir = str(lora_file.parent)
        results = inspector.process(args)
        assert isinstance(results, list)
        assert [r["filename"] for r in results] == ["test_lora.safetensors"]

    def test_single_file_returns_dict(self, inspector, lora_file, args):
        args.lora_file_or_dir = str(lora_file)
        assert isinstance(inspector.process(args), dict)


class TestTags:
    FREQ = {"img": {"cat": 5, "dog": 9}}

    def test_print_tags_sorted_by_count(self, inspector, capsys):
        inspector.print_tags(self.FREQ)
        out = capsys.readouterr().out
        assert out.index("dog") < out.index("cat")

    def test_tags_without_frequency(self, inspector, capsys):
        inspector.tags({})
        assert "No tags found" in capsys.readouterr().out

    def test_tags_from_parsed_file(self, inspector, lora_file, args, capsys):
        args.tags = True
        inspector.process_safetensor_file(lora_file, args)
        assert "dog" in capsys.readouterr().out


class TestSaveMeta:
    def test_writes_json_to_meta_dir(self, inspector, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        inspector.save_meta({"filename": "x.safetensors", "ss_epoch": 3})
        saved = json.loads((tmp_path / "meta" / "x.json").read_text())
        assert saved["ss_epoch"] == 3


class TestCli:
    def run(self, tmp_path, *cli_args):
        return subprocess.run(
            [sys.executable, str(ROOT / "lora-inspector.py"), *cli_args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

    def test_prints_summary(self, lora_file, tmp_path):
        proc = self.run(tmp_path, str(lora_file))
        assert proc.returncode == 0, proc.stderr
        assert "Network Dim/Rank: 16" in proc.stdout

    def test_save_meta_flag(self, lora_file, tmp_path):
        out_dir = tmp_path / "run"
        out_dir.mkdir()
        proc = self.run(out_dir, str(lora_file), "--save_meta")
        assert proc.returncode == 0, proc.stderr
        assert list((out_dir / "meta").glob("*.json"))

    def test_requires_path_argument(self, tmp_path):
        proc = self.run(tmp_path)
        assert proc.returncode != 0
        assert "lora_file_or_dir" in proc.stderr

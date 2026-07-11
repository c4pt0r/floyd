import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

from tools.make_v4_real_layer0_oracle import (
    write_layer0_oracle,
    write_base_forward_oracle,
    write_dspark_oracle,
    write_layer3_hca_oracle,
    write_layers_0_2_oracle,
    write_layers_3_4_oracle,
)


def test_real_layer0_oracle():
    model_dir = os.environ["DSPARK"]
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "oracle.safetensors"
        write_layer0_oracle(model_dir, output, token_id=3)
        with safe_open(output, framework="pt") as oracle:
            shapes = {
                "input": [4, 4096],
                "attn.post": [4],
                "attn.comb": [4, 4],
                "attn.collapsed": [4096],
                "attn.norm": [4096],
                "attn.output": [4096],
                "after_attn": [4, 4096],
                "ffn.post": [4],
                "ffn.comb": [4, 4],
                "ffn.collapsed": [4096],
                "ffn.norm": [4096],
                "router.scores": [256],
                "router.weights": [6],
                "router.indices": [6],
                "moe.output": [4096],
                "output": [4, 4096],
            }
            assert set(oracle.keys()) == set(shapes)
            for name, shape in shapes.items():
                tensor = oracle.get_tensor(name)
                assert list(tensor.shape) == shape
                if tensor.is_floating_point():
                    assert torch.isfinite(tensor).all()


def test_real_layers_0_2_oracle():
    model_dir = os.environ["DSPARK"]
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "layers_0_2.safetensors"
        write_layers_0_2_oracle(model_dir, output, [3, 14, 15, 9])
        with safe_open(output, framework="pt") as oracle:
            for layer in range(3):
                assert oracle.get_slice(f"layer.{layer}.input").get_shape() == [4, 4, 4096]
                assert oracle.get_slice(f"layer.{layer}.output").get_shape() == [4, 4, 4096]
            assert oracle.get_slice("layer.2.compressor.kv").get_shape() == [1, 512]
            assert oracle.get_slice("layer.2.indexer.scores").get_shape() == [4, 1]
            assert oracle.get_slice("layer.2.indexer.indices").get_shape() == [4, 1]
            assert oracle.get_slice("layer.2.block_bias").get_shape() == [4, 1]


def test_real_layer3_hca_oracle():
    model_dir = os.environ["DSPARK"]
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "layer3_hca.safetensors"
        write_layer3_hca_oracle(model_dir, output, [3] * 128)
        with safe_open(output, framework="pt") as oracle:
            shapes = {
                "layer.3.input": [128, 4, 4096],
                "layer.3.hca.kv": [128, 512],
                "layer.3.hca.gate": [128, 512],
                "layer.3.hca.output": [1, 512],
                "layer.3.router.scores": [128, 256],
                "layer.3.router.weights": [128, 6],
                "layer.3.router.indices": [128, 6],
                "layer.3.output": [128, 4, 4096],
            }
            assert set(oracle.keys()) == set(shapes)
            for name, shape in shapes.items():
                tensor = oracle.get_tensor(name)
                assert list(tensor.shape) == shape
                if tensor.is_floating_point():
                    assert torch.isfinite(tensor).all()


def test_real_layers_3_4_oracle():
    model_dir = os.environ["DSPARK"]
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "layers_3_4.safetensors"
        write_layers_3_4_oracle(model_dir, output, [3, 14, 15, 9])
        with safe_open(output, framework="pt") as oracle:
            for layer in (3, 4):
                assert oracle.get_slice(f"layer.{layer}.input").get_shape() == [4, 4, 4096]
                assert oracle.get_slice(f"layer.{layer}.output").get_shape() == [4, 4, 4096]
            assert oracle.get_slice("layer.3.hca.kv").get_shape() == [4, 512]
            assert oracle.get_slice("layer.3.hca.gate").get_shape() == [4, 512]
            assert oracle.get_slice("layer.3.hca.output").get_shape() == [0, 512]
            assert oracle.get_slice("layer.4.compressor.kv").get_shape() == [1, 512]
            assert oracle.get_slice("layer.4.indexer.scores").get_shape() == [4, 1]
            assert oracle.get_slice("layer.4.indexer.indices").get_shape() == [4, 1]
            assert oracle.get_slice("layer.4.router.indices").get_shape() == [4, 6]


def test_real_base_forward_oracle():
    model_dir = os.environ["DSPARK"]
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "base_forward.safetensors"
        write_base_forward_oracle(model_dir, output, [3, 14, 15, 9])
        with safe_open(output, framework="pt") as oracle:
            for layer in (4, 20, 40, 41, 42):
                assert oracle.get_slice(f"layer.{layer}.output").get_shape() == [4, 4, 4096]
            for layer in (40, 41, 42):
                assert oracle.get_slice(f"layer.{layer}.mean").get_shape() == [4, 4096]
            assert oracle.get_slice("final.hidden").get_shape() == [4, 4096]
            assert oracle.get_slice("final.norm").get_shape() == [4, 4096]
            assert oracle.get_slice("final.logits").get_shape() == [129280]
            assert oracle.get_slice("final.argmax").get_shape() == [1]


def test_real_base_decode_oracle():
    model_dir = os.environ["DSPARK"]
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "base_decode.safetensors"
        write_base_forward_oracle(model_dir, output, [3, 14, 15, 9, 5])
        with safe_open(output, framework="pt") as oracle:
            assert oracle.get_slice("layer.42.output").get_shape() == [5, 4, 4096]
            assert oracle.get_slice("final.hidden").get_shape() == [5, 4096]
            assert oracle.get_slice("final.norm").get_shape() == [5, 4096]
            assert oracle.get_slice("final.logits").get_shape() == [129280]
            assert oracle.get_slice("final.argmax").get_shape() == [1]


def test_real_dspark_oracle():
    model_dir = os.environ["DSPARK"]
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "dspark.safetensors"
        write_dspark_oracle(model_dir, output, [3, 14, 15, 9])
        with safe_open(output, framework="pt") as oracle:
            assert oracle.get_slice("dspark.main_x").get_shape() == [4, 4096]
            for stage in range(3):
                assert oracle.get_slice(f"dspark.prefill_kv.{stage}").get_shape() == [4, 512]
                assert oracle.get_slice(f"dspark.stage.{stage}.output").get_shape() == [5, 4, 4096]
            assert oracle.get_slice("dspark.hidden").get_shape() == [5, 4096]
            assert oracle.get_slice("dspark.output_ids").get_shape() == [6]
            assert oracle.get_slice("dspark.logit_argmax").get_shape() == [5]
            assert oracle.get_slice("dspark.confidence").get_shape() == [5]


if __name__ == "__main__":
    test_real_layer0_oracle()
    test_real_layers_0_2_oracle()
    test_real_layer3_hca_oracle()
    test_real_layers_3_4_oracle()
    test_real_base_forward_oracle()
    test_real_base_decode_oracle()
    test_real_dspark_oracle()
    print("v4 real layer0 oracle tests: ok")

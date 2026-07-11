import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

from tools.make_v4_real_layer0_oracle import write_layer0_oracle, write_layers_0_2_oracle


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


if __name__ == "__main__":
    test_real_layer0_oracle()
    test_real_layers_0_2_oracle()
    print("v4 real layer0 oracle tests: ok")

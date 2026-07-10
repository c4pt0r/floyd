import json
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

from tools.make_v4_oracle import PROMPT_IDS, build_tiny, write_tiny


def test_build_tiny():
    model, config = build_tiny()

    assert config.layer_types == [
        "sliding_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
    ]
    assert config.mlp_layer_types == ["hash_moe", "moe", "moe"]
    assert config.hc_mult == 2
    assert config.scoring_func == "sqrtsoftplus"

    hash_ids = model.model.layers[0].mlp.gate.tid2eid
    assert hash_ids[3].tolist() == [3, 2]

    input_ids = torch.tensor([PROMPT_IDS])
    with torch.no_grad():
        output = model(
            input_ids,
            use_cache=False,
            output_hidden_states=True,
            output_router_logits=True,
        )

    assert output.logits.shape == (1, len(PROMPT_IDS), 128)
    assert [tuple(t.shape) for t in output.hidden_states] == [
        (1, len(PROMPT_IDS), 2, 64),
        (1, len(PROMPT_IDS), 2, 64),
        (1, len(PROMPT_IDS), 2, 64),
        (1, len(PROMPT_IDS), 64),
    ]
    assert len(output.router_logits) == 2


def test_write_tiny_is_complete_and_deterministic():
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        first_ref = Path(first) / "ref.json"
        second_ref = Path(second) / "ref.json"
        write_tiny(Path(first) / "fixture", first_ref, ngen=2)
        write_tiny(Path(second) / "fixture", second_ref, ngen=2)

        ref = json.loads(first_ref.read_text())
        assert ref == json.loads(second_ref.read_text())
        assert ref["prompt_ids"] == PROMPT_IDS
        assert len(ref["full_ids"]) == len(PROMPT_IDS) + 2
        assert len(ref["tf_pred"]) == len(ref["full_ids"])

        fixture = Path(first) / "fixture"
        assert (fixture / "config.json").is_file()
        assert (fixture / "model.safetensors").is_file()
        assert (fixture / "oracle.safetensors").is_file()

        with safe_open(fixture / "oracle.safetensors", framework="pt") as oracle:
            expected_keys = {
                "hidden.0", "hidden.1", "hidden.2", "hidden.3",
                "input_ids", "logits", "router.0", "router.1",
            }
            for layer in range(3):
                for field in ("input", "output", "logits", "weights", "indices"):
                    expected_keys.add(f"moe.{layer}.{field}")
            assert set(oracle.keys()) == expected_keys
            n_tokens = len(ref["full_ids"])
            assert oracle.get_slice("input_ids").get_shape() == [n_tokens]
            assert oracle.get_slice("logits").get_shape() == [n_tokens, 128]
            assert oracle.get_slice("hidden.0").get_shape() == [n_tokens, 2, 64]
            assert oracle.get_slice("hidden.3").get_shape() == [n_tokens, 64]
            assert oracle.get_slice("router.0").get_shape() == [n_tokens, 8]
            for layer in range(3):
                assert oracle.get_slice(f"moe.{layer}.input").get_shape() == [n_tokens, 64]
                assert oracle.get_slice(f"moe.{layer}.output").get_shape() == [n_tokens, 64]
                assert oracle.get_slice(f"moe.{layer}.logits").get_shape() == [n_tokens, 8]
                assert oracle.get_slice(f"moe.{layer}.weights").get_shape() == [n_tokens, 2]
                assert oracle.get_slice(f"moe.{layer}.indices").get_shape() == [n_tokens, 2]


if __name__ == "__main__":
    test_build_tiny()
    test_write_tiny_is_complete_and_deterministic()
    print("v4 oracle builder tests: ok")

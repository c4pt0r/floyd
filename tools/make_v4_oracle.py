import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM


PROMPT_IDS = [3, 14, 15, 9, 26, 5, 35, 8]


def build_tiny():
    config = DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        q_lora_rank=32,
        num_experts_per_tok=2,
        n_routed_experts=8,
        n_shared_experts=1,
        max_position_embeddings=128,
        layer_types=[
            "sliding_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
        ],
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 8,
        },
        hc_mult=2,
        hc_sinkhorn_iters=4,
        mlp_layer_types=["hash_moe", "moe", "moe"],
        sliding_window=8,
        o_groups=2,
        o_lora_rank=16,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=2,
        num_nextn_predict_layers=0,
        partial_rotary_factor=0.25,
        use_cache=False,
    )

    torch.manual_seed(1234)
    model = DeepseekV4ForCausalLM(config).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            gate = layer.mlp.gate
            if hasattr(gate, "tid2eid"):
                token_ids = torch.arange(config.vocab_size)
                gate.tid2eid[:, 0] = token_ids % config.n_routed_experts
                gate.tid2eid[:, 1] = (token_ids * 3 + 1) % config.n_routed_experts
            else:
                gate.e_score_correction_bias.copy_(
                    torch.linspace(-0.05, 0.05, config.n_routed_experts)
                )
    return model, config


def greedy_ids(model, prompt_ids, ngen):
    full_ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(ngen):
            input_ids = torch.tensor([full_ids], dtype=torch.long)
            logits = model(input_ids, use_cache=False).logits
            full_ids.append(int(logits[0, -1].argmax()))
    return full_ids


def forward_with_moe_captures(model, input_ids):
    captures = {}
    handles = []

    for layer_index, layer in enumerate(model.model.layers):
        def make_pre_hook(index):
            def hook(module, args, kwargs):
                captures[f"moe.{index}.input"] = args[0][0].detach().float().contiguous().clone()
            return hook

        def make_output_hook(index):
            def hook(module, args, kwargs, output):
                captures[f"moe.{index}.output"] = output[0].detach().float().contiguous().clone()
            return hook

        def make_gate_hook(index):
            def hook(module, args, kwargs, output):
                logits, weights, indices = output
                captures[f"moe.{index}.logits"] = logits.detach().float().contiguous().clone()
                captures[f"moe.{index}.weights"] = weights.detach().float().contiguous().clone()
                captures[f"moe.{index}.indices"] = indices.detach().contiguous().clone()
            return hook

        handles.append(layer.mlp.register_forward_pre_hook(
            make_pre_hook(layer_index), with_kwargs=True
        ))
        handles.append(layer.mlp.register_forward_hook(
            make_output_hook(layer_index), with_kwargs=True
        ))
        handles.append(layer.mlp.gate.register_forward_hook(
            make_gate_hook(layer_index), with_kwargs=True
        ))

    try:
        with torch.no_grad():
            output = model(
                input_ids,
                use_cache=False,
                output_hidden_states=True,
                output_router_logits=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    return output, captures


def write_tiny(out_dir, ref_path, ngen=8):
    out_dir = Path(out_dir)
    ref_path = Path(ref_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path.parent.mkdir(parents=True, exist_ok=True)

    model, _ = build_tiny()
    full_ids = greedy_ids(model, PROMPT_IDS, ngen)
    input_ids = torch.tensor([full_ids], dtype=torch.long)
    output, moe_captures = forward_with_moe_captures(model, input_ids)

    model.save_pretrained(out_dir, safe_serialization=True)
    oracle = {"logits": output.logits[0].float().contiguous()}
    for i, hidden in enumerate(output.hidden_states):
        oracle[f"hidden.{i}"] = hidden[0].float().contiguous()
    for i, router in enumerate(output.router_logits):
        oracle[f"router.{i}"] = router.float().contiguous().clone()
    oracle.update(moe_captures)
    save_file(oracle, out_dir / "oracle.safetensors")

    reference = {
        "prompt_ids": PROMPT_IDS,
        "full_ids": full_ids,
        "tf_pred": output.logits[0].argmax(dim=-1).tolist(),
    }
    ref_path.write_text(json.dumps(reference, indent=2) + "\n")
    return reference


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    tiny = subparsers.add_parser("tiny")
    tiny.add_argument("--out", default="fixture_tiny_v4")
    tiny.add_argument("--ref", default="ref_v4_tiny.json")
    tiny.add_argument("--ngen", type=int, default=8)
    args = parser.parse_args()

    reference = write_tiny(args.out, args.ref, args.ngen)
    print(f"saved {args.out} and {args.ref}")
    print(f"prompt={reference['prompt_ids']}")
    print(f"full={reference['full_ids']}")


if __name__ == "__main__":
    main()

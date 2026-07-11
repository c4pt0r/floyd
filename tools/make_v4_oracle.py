import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
from transformers.models.deepseek_v4.modeling_deepseek_v4 import apply_rotary_pos_emb


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
            "heavily_compressed_attention": 128,
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

        def make_attn_pre_hook(index):
            def hook(module, args, kwargs):
                captures[f"attn.{index}.input"] = args[0][0].detach().float().contiguous().clone()
            return hook

        def make_attn_output_hook(index):
            def hook(module, args, kwargs, output):
                captures[f"attn.{index}.output"] = output[0][0].detach().float().contiguous().clone()
            return hook

        def make_hc_pre_hook(index, site):
            def hook(module, args, kwargs):
                captures[f"hc.{index}.{site}.input"] = (
                    args[0][0].detach().float().contiguous().clone()
                )
            return hook

        def make_hc_output_hook(index, site):
            def hook(module, args, kwargs, output):
                post, comb, collapsed = output
                captures[f"hc.{index}.{site}.post"] = post[0].detach().float().contiguous().clone()
                captures[f"hc.{index}.{site}.comb"] = comb[0].detach().float().contiguous().clone()
                captures[f"hc.{index}.{site}.collapsed"] = (
                    collapsed[0].detach().float().contiguous().clone()
                )
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
        handles.append(layer.self_attn.register_forward_pre_hook(
            make_attn_pre_hook(layer_index), with_kwargs=True
        ))
        handles.append(layer.self_attn.register_forward_hook(
            make_attn_output_hook(layer_index), with_kwargs=True
        ))
        for site, module in (("attn", layer.attn_hc), ("ffn", layer.ffn_hc)):
            handles.append(module.register_forward_pre_hook(
                make_hc_pre_hook(layer_index, site), with_kwargs=True
            ))
            handles.append(module.register_forward_hook(
                make_hc_output_hook(layer_index, site), with_kwargs=True
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


def build_compressor_captures(model):
    captures = {}
    cases = (
        ("hca", model.model.layers[1].self_attn.compressor, 256, 1),
        ("csa", model.model.layers[2].self_attn.compressor, 12, 2),
    )

    with torch.no_grad():
        for mode, compressor, n_tokens, layer_index in cases:
            hidden = torch.linspace(
                -1.0, 1.0, n_tokens * model.config.hidden_size, dtype=torch.float32
            ).reshape(1, n_tokens, model.config.hidden_size)
            q_residual = torch.linspace(
                0.5, -0.5, n_tokens * model.config.q_lora_rank, dtype=torch.float32
            ).reshape(1, n_tokens, model.config.q_lora_rank)
            position_ids = torch.arange(n_tokens, dtype=torch.long).unsqueeze(0)
            kv = compressor.kv_proj(hidden)
            gate = compressor.gate_proj(hidden)
            handles = []
            if mode == "csa":
                def capture_scorer(module, args, output):
                    q, compressed_kv, scorer_hidden = args
                    captures["indexer.q"] = q[0].detach().float().contiguous().clone()
                    captures["indexer.kv"] = (
                        compressed_kv[0].detach().float().contiguous().clone()
                    )
                    captures["indexer.weights"] = (
                        module.weights_proj(scorer_hidden)[0]
                        .detach().float().contiguous().clone()
                    )
                    captures["indexer.scores"] = (
                        output[0].detach().float().contiguous().clone()
                    )

                def capture_indexer(module, args, output):
                    captures["indexer.positions"] = (
                        args[2][0].detach().contiguous().clone()
                    )
                    captures["indexer.indices"] = (
                        output[0].detach().contiguous().clone()
                    )

                handles.append(compressor.indexer.scorer.register_forward_hook(capture_scorer))
                handles.append(compressor.indexer.register_forward_hook(capture_indexer))
            try:
                compressed, _ = compressor(
                    hidden, q_residual, position_ids, None, layer_index
                )
            finally:
                for handle in handles:
                    handle.remove()

            captures[f"compress.{mode}.kv"] = kv[0].float().contiguous()
            captures[f"compress.{mode}.gate"] = gate[0].float().contiguous()
            captures[f"compress.{mode}.ape"] = (
                compressor.position_bias.detach().float().contiguous().clone()
            )
            captures[f"compress.{mode}.norm"] = (
                compressor.kv_norm.weight.detach().float().contiguous().clone()
            )
            captures[f"compress.{mode}.output"] = compressed[0, 0].float().contiguous()

    return captures


def build_sliding_key_capture(model, attention_input):
    attention = model.model.layers[0].self_attn
    hidden = attention_input.unsqueeze(0)
    positions = torch.arange(hidden.shape[1], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        key = attention.kv_norm(attention.kv_proj(hidden))
        key = key.view(1, hidden.shape[1], -1, attention.head_dim).transpose(1, 2)
        cosine, sine = model.model.rotary_emb(
            hidden, position_ids=positions, layer_type="main"
        )
        key = apply_rotary_pos_emb(key, cosine, sine)
    return key[0, 0].float().contiguous()


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
    oracle = {
        "input_ids": input_ids[0].contiguous().clone(),
        "logits": output.logits[0].float().contiguous(),
    }
    for i, hidden in enumerate(output.hidden_states):
        oracle[f"hidden.{i}"] = hidden[0].float().contiguous()
    for i, router in enumerate(output.router_logits):
        oracle[f"router.{i}"] = router.float().contiguous().clone()
    oracle.update(moe_captures)
    oracle.update(build_compressor_captures(model))
    oracle["attn.0.keys"] = build_sliding_key_capture(
        model, moe_captures["attn.0.input"]
    )
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

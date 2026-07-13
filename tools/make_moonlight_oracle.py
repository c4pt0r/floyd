"""Capture Moonlight activations and greedy tokens for Metal parity tests."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, DeepseekV3ForCausalLM


def _tensor(value):
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not torch.is_tensor(value):
        raise TypeError(f"expected tensor output, got {type(value).__name__}")
    value = value.detach().cpu().clone()
    if value.ndim and value.shape[0] == 1:
        value = value[0]
    if value.is_floating_point():
        value = value.float()
    return value.contiguous()


def _capture_output(captured, name):
    def hook(_module, _inputs, output):
        captured[name] = _tensor(output)

    return hook


def _capture_input(captured, name):
    def hook(_module, inputs):
        captured[name] = _tensor(inputs)

    return hook


def _capture_router(captured, layer):
    def hook(_module, _inputs, output):
        scores, weights, indices = output
        captured[f"layer.{layer}.router_scores"] = _tensor(scores)
        captured[f"layer.{layer}.router_weights"] = _tensor(weights)
        captured[f"layer.{layer}.router_ids"] = _tensor(indices)

    return hook


def _install_hooks(model, captured):
    hooks = []

    def output(module, name):
        hooks.append(module.register_forward_hook(_capture_output(captured, name)))

    def input_(module, name):
        hooks.append(module.register_forward_pre_hook(_capture_input(captured, name)))

    output(model.model.embed_tokens, "embed")
    input_(model.model.norm, "final_input")
    output(model.model.norm, "final_norm")
    output(model.lm_head, "logits")
    for layer_index, layer in enumerate(model.model.layers):
        prefix = f"layer.{layer_index}"
        input_(layer, f"{prefix}.input")
        output(layer.input_layernorm, f"{prefix}.input_norm")
        output(layer.self_attn.q_proj, f"{prefix}.q")
        output(layer.self_attn.kv_a_proj_with_mqa, f"{prefix}.kv_a")
        output(layer.self_attn.kv_a_layernorm, f"{prefix}.kv_norm")
        output(layer.self_attn.kv_b_proj, f"{prefix}.kv_b")
        output(layer.self_attn, f"{prefix}.attn")
        input_(layer.post_attention_layernorm, f"{prefix}.post_attn")
        output(layer.post_attention_layernorm, f"{prefix}.post_norm")
        output(layer.mlp, f"{prefix}.mlp")
        output(layer, f"{prefix}.output")
        if hasattr(layer.mlp, "gate"):
            hooks.append(
                layer.mlp.gate.register_forward_hook(
                    _capture_router(captured, layer_index)
                )
            )
        if hasattr(layer.mlp, "shared_experts"):
            output(layer.mlp.shared_experts, f"{prefix}.shared_mlp")
    return hooks


def _greedy(model, prompt_ids, max_tokens):
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    generated = []
    past_key_values = None
    with torch.no_grad():
        for _ in range(max_tokens):
            output = model(
                input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            token = int(output.logits[0, -1].argmax())
            generated.append(token)
            input_ids = torch.cat(
                (input_ids, torch.tensor([[token]], dtype=torch.long)), dim=1
            )
            past_key_values = output.past_key_values
    return generated


def _load_model(model_path):
    options = {
        "dtype": torch.float32,
        "device_map": None,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, **options
        )
    except ImportError as error:
        print(
            f"remote Moonlight code is incompatible ({error}); using native DeepSeek V3"
        )
        model = DeepseekV3ForCausalLM.from_pretrained(model_path, **options)
    return model.eval()


def capture_checkpoint(model_path, output, prompt_ids, max_tokens=2):
    model_path = Path(model_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = _load_model(model_path)

    captured = {"input_ids": torch.tensor(prompt_ids, dtype=torch.int64)}
    hooks = _install_hooks(model, captured)
    with torch.no_grad():
        model(torch.tensor([prompt_ids], dtype=torch.long), use_cache=False)
    for hook in hooks:
        hook.remove()

    for layer_index, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "shared_experts"):
            prefix = f"layer.{layer_index}"
            captured[f"{prefix}.routed_mlp"] = (
                captured[f"{prefix}.mlp"] - captured[f"{prefix}.shared_mlp"]
            ).contiguous()

    greedy_ids = _greedy(model, prompt_ids, max_tokens)
    full_ids = list(prompt_ids) + greedy_ids
    second_input_ids = full_ids + [int(prompt_ids[-1])]
    second_greedy_ids = _greedy(model, second_input_ids, max_tokens)
    tf_pred = [int(token) for token in captured["logits"].argmax(dim=-1)]
    save_file(captured, output / "oracle.safetensors")
    (output / "ref.json").write_text(
        json.dumps(
            {
                "prompt_ids": list(prompt_ids),
                "full_ids": full_ids,
                "greedy_ids": greedy_ids,
                "tf_pred": tf_pred,
                "turns": [
                    {"input_ids": list(prompt_ids), "output_ids": greedy_ids},
                    {"input_ids": second_input_ids, "output_ids": second_greedy_ids},
                ],
            },
            separators=(",", ":"),
        )
    )
    (output / "cpu_ref.json").write_text(
        json.dumps(
            {
                "prompt_ids": list(prompt_ids),
                "full_ids": list(prompt_ids),
                "tf_pred": tf_pred,
            },
            separators=(",", ":"),
        )
    )


def _prompt_ids(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    tiny = subparsers.add_parser("tiny")
    tiny.add_argument("--model", required=True)
    tiny.add_argument("--output", required=True)
    tiny.add_argument(
        "--prompt-ids",
        default="3,14,159,26,53,58,200,11,77,240,5,99",
    )
    real = subparsers.add_parser("real")
    real.add_argument("--source", required=True)
    real.add_argument("--output", required=True)
    real.add_argument(
        "--prompt", default="What is the capital of France? Answer in one word."
    )
    for command in (tiny, real):
        command.add_argument("--max-tokens", type=int, default=2)
    args = parser.parse_args()

    if args.mode == "tiny":
        prompt_ids = [int(value) for value in args.prompt_ids.split(",")]
        capture_checkpoint(args.model, args.output, prompt_ids, args.max_tokens)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True)
    capture_checkpoint(
        args.source,
        args.output,
        _prompt_ids(tokenizer, args.prompt),
        args.max_tokens,
    )


if __name__ == "__main__":
    main()

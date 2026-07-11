import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


class Checkpoint:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        index = json.loads((self.model_dir / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]

    def tensor(self, name):
        path = self.model_dir / self.weight_map[name]
        with safe_open(path, framework="pt", device="cpu") as source:
            return source.get_tensor(name)

    def row(self, name, index):
        path = self.model_dir / self.weight_map[name]
        with safe_open(path, framework="pt", device="cpu") as source:
            return source.get_slice(name)[index]


def fp8_linear(checkpoint, prefix, x, row_start=0, row_count=None):
    weight = checkpoint.tensor(f"{prefix}.weight")
    rows = weight.shape[0]
    if row_count is None:
        row_count = rows - row_start
    assert row_start % 128 == 0 and row_count % 128 == 0
    weight = weight[row_start:row_start + row_count].float()
    scale = checkpoint.tensor(f"{prefix}.scale")
    scale = scale[row_start // 128:(row_start + row_count) // 128].float()
    expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    weight.mul_(expanded[:, :weight.shape[1]])
    output = torch.mv(weight, x.float())
    del weight, scale, expanded
    gc.collect()
    return output


def fp4_linear(checkpoint, prefix, x):
    packed = checkpoint.tensor(f"{prefix}.weight").view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    weight = FP4_TABLE[torch.stack((low, high), dim=-1).long()].flatten(1)
    scale = checkpoint.tensor(f"{prefix}.scale").float().repeat_interleave(32, 1)
    weight.mul_(scale)
    output = torch.mv(weight, x.float())
    del packed, low, high, weight, scale
    gc.collect()
    return output


def dense_linear(checkpoint, name, x):
    weight = checkpoint.tensor(name).float()
    output = torch.mv(weight, x.float())
    del weight
    gc.collect()
    return output


def rmsnorm(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.square().mean() + eps) * weight.float()


def hyper_connection(checkpoint, site, streams):
    prefix = f"layers.0.hc_{site}"
    fn = checkpoint.tensor(f"{prefix}_fn").float()
    base = checkpoint.tensor(f"{prefix}_base").float()
    scale = checkpoint.tensor(f"{prefix}_scale").float()
    flat = streams.flatten().float()
    mixes = torch.mv(fn, flat * torch.rsqrt(flat.square().mean() + 1e-6))
    pre = torch.sigmoid(mixes[:4] * scale[0] + base[:4]) + 1e-6
    post = 2.0 * torch.sigmoid(mixes[4:8] * scale[1] + base[4:8])
    comb = (mixes[8:].reshape(4, 4) * scale[2] + base[8:].reshape(4, 4)).softmax(-1) + 1e-6
    comb = comb / (comb.sum(0, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(0, keepdim=True) + 1e-6)
    collapsed = (pre[:, None] * streams.float()).sum(0)
    return collapsed, post, comb


def hc_post(block_output, residual, post, comb):
    return post[:, None] * block_output[None, :] + torch.matmul(comb, residual.float())


def attention(checkpoint, x):
    q_a = fp8_linear(checkpoint, "layers.0.attn.wq_a", x)
    q_norm_weight = checkpoint.tensor("layers.0.attn.q_norm.weight")
    q_a = rmsnorm(q_a, q_norm_weight)
    q = fp8_linear(checkpoint, "layers.0.attn.wq_b", q_a).reshape(64, 512)
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)

    kv = fp8_linear(checkpoint, "layers.0.attn.wkv", x)
    kv = rmsnorm(kv, checkpoint.tensor("layers.0.attn.kv_norm.weight"))
    scores = (q * kv[None, :]).sum(-1) / math.sqrt(512.0)
    sinks = checkpoint.tensor("layers.0.attn.attn_sink").float()
    maximum = torch.maximum(scores, sinks)
    key_probability = torch.exp(scores - maximum) / (
        torch.exp(scores - maximum) + torch.exp(sinks - maximum)
    )
    context = key_probability[:, None] * kv[None, :]

    grouped = []
    for group in range(8):
        group_input = context[group * 8:(group + 1) * 8].reshape(-1)
        grouped.append(fp8_linear(
            checkpoint, "layers.0.attn.wo_a", group_input,
            row_start=group * 1024, row_count=1024,
        ))
    return fp8_linear(checkpoint, "layers.0.attn.wo_b", torch.cat(grouped))


def expert(checkpoint, prefix, x, fp4):
    linear = fp4_linear if fp4 else fp8_linear
    gate = linear(checkpoint, f"{prefix}.w1", x).clamp(max=10.0)
    up = linear(checkpoint, f"{prefix}.w3", x).clamp(min=-10.0, max=10.0)
    activated = F.silu(gate) * up
    return linear(checkpoint, f"{prefix}.w2", activated)


def moe(checkpoint, x, token_id):
    logits = dense_linear(checkpoint, "layers.0.ffn.gate.weight", x)
    scores = F.softplus(logits).sqrt()
    indices = checkpoint.row("layers.0.ffn.gate.tid2eid", token_id).long()
    weights = scores[indices]
    weights = weights / weights.sum() * 1.5
    output = torch.zeros(4096, dtype=torch.float32)
    for weight, index in zip(weights, indices):
        output += weight * expert(
            checkpoint, f"layers.0.ffn.experts.{int(index)}", x, fp4=True
        )
    output += expert(checkpoint, "layers.0.ffn.shared_experts", x, fp4=False)
    return output, scores, weights, indices


def write_layer0_oracle(model_dir, output_path, token_id=3):
    torch.set_num_threads(8)
    checkpoint = Checkpoint(model_dir)
    embedding = checkpoint.row("embed.weight", token_id).float()
    streams = embedding.repeat(4, 1)
    captures = {"input": streams.clone()}

    collapsed, post, comb = hyper_connection(checkpoint, "attn", streams)
    captures.update({"attn.post": post, "attn.comb": comb, "attn.collapsed": collapsed})
    attn_norm = rmsnorm(collapsed, checkpoint.tensor("layers.0.attn_norm.weight"))
    attn_output = attention(checkpoint, attn_norm)
    streams = hc_post(attn_output, streams, post, comb)
    captures.update({"attn.norm": attn_norm, "attn.output": attn_output, "after_attn": streams})

    collapsed, post, comb = hyper_connection(checkpoint, "ffn", streams)
    captures.update({"ffn.post": post, "ffn.comb": comb, "ffn.collapsed": collapsed})
    ffn_norm = rmsnorm(collapsed, checkpoint.tensor("layers.0.ffn_norm.weight"))
    moe_output, scores, weights, indices = moe(checkpoint, ffn_norm, token_id)
    streams = hc_post(moe_output, streams, post, comb)
    captures.update({
        "ffn.norm": ffn_norm,
        "router.scores": scores,
        "router.weights": weights,
        "router.indices": indices,
        "moe.output": moe_output,
        "output": streams,
    })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: value.detach().contiguous() for name, value in captures.items()}, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--token-id", type=int, default=3)
    args = parser.parse_args()
    write_layer0_oracle(args.model, args.output, args.token_id)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

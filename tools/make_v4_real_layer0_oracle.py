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


def quant_linear_batch(checkpoint, prefix, x, fp4=False, row_start=0, row_count=None):
    if fp4:
        packed = checkpoint.tensor(f"{prefix}.weight").view(torch.uint8)
        low, high = packed & 0x0F, (packed >> 4) & 0x0F
        weight = FP4_TABLE[torch.stack((low, high), dim=-1).long()].flatten(1)
        weight.mul_(checkpoint.tensor(f"{prefix}.scale").float().repeat_interleave(32, 1))
    else:
        weight = checkpoint.tensor(f"{prefix}.weight")
        if row_count is None:
            row_count = weight.shape[0] - row_start
        weight = weight[row_start:row_start + row_count].float()
        scale = checkpoint.tensor(f"{prefix}.scale").float()
        scale = scale[row_start // 128:(row_start + row_count) // 128]
        weight.mul_(scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:, :weight.shape[1]])
    output = F.linear(x.float(), weight)
    del weight
    gc.collect()
    return output


def norm_batch(x, weight):
    return x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * weight.float()


def hc_batch(checkpoint, layer, site, streams):
    prefix = f"layers.{layer}.hc_{site}"
    flat = streams.flatten(1).float()
    normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
    mixes = F.linear(normalized, checkpoint.tensor(f"{prefix}_fn").float())
    base = checkpoint.tensor(f"{prefix}_base").float()
    scale = checkpoint.tensor(f"{prefix}_scale").float()
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + 1e-6
    post = 2.0 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
    comb = (mixes[:, 8:].reshape(-1, 4, 4) * scale[2] + base[8:].reshape(1, 4, 4)).softmax(-1) + 1e-6
    comb = comb / (comb.sum(1, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(2, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(1, keepdim=True) + 1e-6)
    return (pre.unsqueeze(-1) * streams.float()).sum(1), post, comb


def hc_post_batch(block, residual, post, comb):
    return post.unsqueeze(-1) * block.unsqueeze(1) + torch.einsum("sij,sjd->sid", comb, residual.float())


def rope_frequencies(compressed):
    dim, base = 64, 160000.0 if compressed else 10000.0
    frequencies = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    if compressed:
        def correction(rotations):
            return dim * math.log(65536 / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(correction(32)), 0)
        high = min(math.ceil(correction(1)), dim - 1)
        ramp = ((torch.arange(dim // 2).float() - low) / (high - low)).clamp(0, 1)
        smooth = 1 - ramp
        frequencies = frequencies / 16 * (1 - smooth) + frequencies * smooth
    return frequencies


def rope_batch(x, positions, compressed, inverse=False):
    rope = x[..., -64:].float().reshape(*x.shape[:-1], 32, 2)
    angles = positions.float()[:, None] * rope_frequencies(compressed)[None, :]
    cosine, sine = angles.cos(), angles.sin()
    while cosine.ndim < rope.ndim - 1:
        cosine, sine = cosine.unsqueeze(1), sine.unsqueeze(1)
    if inverse:
        sine = -sine
    first, second = rope[..., 0], rope[..., 1]
    rotated = torch.stack((first * cosine - second * sine,
                           second * cosine + first * sine), dim=-1).flatten(-2)
    return torch.cat((x[..., :-64].float(), rotated), dim=-1)


def compressor_batch(checkpoint, prefix, hidden, head_dim, ratio):
    projected_kv = F.linear(hidden.float(), checkpoint.tensor(f"{prefix}.wkv.weight").float())
    projected_gate = F.linear(hidden.float(), checkpoint.tensor(f"{prefix}.wgate.weight").float())
    assert hidden.shape[0] % ratio == 0
    blocks = hidden.shape[0] // ratio
    kv = projected_kv.reshape(blocks, ratio, -1)
    gate = projected_gate.reshape(blocks, ratio, -1)
    gate = gate + checkpoint.tensor(f"{prefix}.ape").float()
    if ratio == 4:
        pooled_blocks = []
        for block in range(blocks):
            values = kv[block, :, head_dim:]
            scores = gate[block, :, head_dim:]
            if block > 0:
                values = torch.cat((kv[block - 1, :, :head_dim], values), 0)
                scores = torch.cat((gate[block - 1, :, :head_dim], scores), 0)
            pooled_blocks.append((values * scores.softmax(0)).sum(0))
        pooled = torch.stack(pooled_blocks)
    else:
        assert projected_kv.shape[1] == head_dim
        pooled = (kv * gate.softmax(1)).sum(1)
    pooled = norm_batch(pooled, checkpoint.tensor(f"{prefix}.norm.weight"))
    pooled = rope_batch(pooled, torch.arange(blocks) * ratio, compressed=True)
    return pooled, projected_kv, projected_gate


def attention_batch(checkpoint, layer, hidden, captures):
    prefix = f"layers.{layer}.attn"
    positions = torch.arange(hidden.shape[0])
    compressed = layer >= 2
    q_residual = norm_batch(
        quant_linear_batch(checkpoint, f"{prefix}.wq_a", hidden),
        checkpoint.tensor(f"{prefix}.q_norm.weight"),
    )
    q = quant_linear_batch(checkpoint, f"{prefix}.wq_b", q_residual).reshape(-1, 64, 512)
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)
    q = rope_batch(q, positions, compressed)
    kv = norm_batch(
        quant_linear_batch(checkpoint, f"{prefix}.wkv", hidden),
        checkpoint.tensor(f"{prefix}.kv_norm.weight"),
    )
    kv = rope_batch(kv, positions, compressed)
    ratio = 0 if layer < 2 else (4 if layer % 2 == 0 else 128)
    compressed_kv, index_scores, index_ids, block_bias = None, None, None, None
    if ratio:
        compressed_kv, projected_kv, projected_gate = compressor_batch(
            checkpoint, f"{prefix}.compressor", hidden, 512, ratio
        )
    if ratio == 4:
        index_kv = compressor_batch(
            checkpoint, f"{prefix}.indexer.compressor", hidden, 128, ratio
        )[0]
        index_q = quant_linear_batch(checkpoint, f"{prefix}.indexer.wq_b", q_residual).reshape(-1, 64, 128)
        index_q = rope_batch(index_q, positions, compressed=True)
        head_weights = F.linear(hidden.float(), checkpoint.tensor(f"{prefix}.indexer.weights_proj.weight").float())
        dots = torch.einsum("shd,td->sht", index_q, index_kv).relu() / math.sqrt(128.0)
        index_scores = (dots * (head_weights / 8.0).unsqueeze(-1)).sum(1)
        available = (positions + 1) // 4
        masked_scores = index_scores.masked_fill(
            torch.arange(index_kv.shape[0])[None, :] >= available[:, None],
            float("-inf"),
        )
        index_ids = masked_scores.topk(index_kv.shape[0], dim=-1).indices
        index_ids = torch.where(index_ids < available[:, None], index_ids, -1)
        block_bias = torch.where(index_ids >= 0, torch.zeros_like(index_scores),
                                 torch.full_like(index_scores, float("-inf")))
        captures[f"layer.{layer}.compressor.kv"] = compressed_kv
        captures[f"layer.{layer}.indexer.scores"] = index_scores
        captures[f"layer.{layer}.indexer.indices"] = index_ids
        captures[f"layer.{layer}.block_bias"] = block_bias
    elif ratio == 128:
        captures[f"layer.{layer}.hca.kv"] = projected_kv
        captures[f"layer.{layer}.hca.gate"] = projected_gate
        captures[f"layer.{layer}.hca.output"] = compressed_kv

    sinks = checkpoint.tensor(f"{prefix}.attn_sink").float()
    context = torch.zeros_like(q)
    for token in range(hidden.shape[0]):
        keys = kv[max(0, token - 127):token + 1]
        if ratio == 4:
            selected = index_ids[token][index_ids[token] >= 0]
            if selected.numel():
                keys = torch.cat((keys, compressed_kv[selected]), dim=0)
        elif ratio == 128:
            available = (token + 1) // ratio
            if available:
                keys = torch.cat((keys, compressed_kv[:available]), dim=0)
        scores = torch.einsum("hd,td->ht", q[token], keys) / math.sqrt(512.0)
        maximum = torch.maximum(scores.max(-1).values, sinks)
        probabilities = torch.exp(scores - maximum[:, None])
        denominator = probabilities.sum(-1) + torch.exp(sinks - maximum)
        context[token] = torch.einsum("ht,td->hd", probabilities / denominator[:, None], keys)
    context = rope_batch(context, positions, compressed, inverse=True)
    grouped = []
    for group in range(8):
        grouped.append(quant_linear_batch(
            checkpoint, f"{prefix}.wo_a",
            context[:, group * 8:(group + 1) * 8].reshape(-1, 4096),
            row_start=group * 1024, row_count=1024,
        ))
    return quant_linear_batch(checkpoint, f"{prefix}.wo_b", torch.cat(grouped, -1))


def moe_batch(checkpoint, layer, hidden, token_ids, captures=None):
    prefix = f"layers.{layer}.ffn"
    logits = F.linear(hidden.float(), checkpoint.tensor(f"{prefix}.gate.weight").float())
    scores = F.softplus(logits).sqrt()
    if layer < 3:
        indices = torch.stack([
            checkpoint.row(f"{prefix}.gate.tid2eid", token_id).long()
            for token_id in token_ids
        ])
    else:
        bias = checkpoint.tensor(f"{prefix}.gate.bias").float()
        indices = (scores + bias).topk(6, dim=-1).indices
    weights = scores.gather(1, indices)
    weights = weights / weights.sum(-1, keepdim=True) * 1.5
    if captures is not None:
        captures[f"layer.{layer}.router.scores"] = scores
        captures[f"layer.{layer}.router.weights"] = weights
        captures[f"layer.{layer}.router.indices"] = indices
    output = torch.zeros_like(hidden.float())
    for expert_id in indices.unique().tolist():
        token, rank = torch.where(indices == expert_id)
        ep = f"{prefix}.experts.{expert_id}"
        expert_input = hidden[token]
        gate = quant_linear_batch(checkpoint, f"{ep}.w1", expert_input, fp4=True).clamp(max=10)
        up = quant_linear_batch(checkpoint, f"{ep}.w3", expert_input, fp4=True).clamp(-10, 10)
        activated = weights[token, rank, None] * F.silu(gate) * up
        output[token] += quant_linear_batch(checkpoint, f"{ep}.w2", activated, fp4=True)
    shared = f"{prefix}.shared_experts"
    gate = quant_linear_batch(checkpoint, f"{shared}.w1", hidden).clamp(max=10)
    up = quant_linear_batch(checkpoint, f"{shared}.w3", hidden).clamp(-10, 10)
    output += quant_linear_batch(checkpoint, f"{shared}.w2", F.silu(gate) * up)
    return output


def write_layers_0_2_oracle(model_dir, output_path, token_ids):
    assert len(token_ids) == 4
    torch.set_num_threads(8)
    checkpoint = Checkpoint(model_dir)
    streams = torch.stack([checkpoint.row("embed.weight", token).float() for token in token_ids])
    streams = streams[:, None, :].repeat(1, 4, 1)
    captures = {}
    for layer in range(3):
        captures[f"layer.{layer}.input"] = streams.clone()
        collapsed, post, comb = hc_batch(checkpoint, layer, "attn", streams)
        hidden = norm_batch(collapsed, checkpoint.tensor(f"layers.{layer}.attn_norm.weight"))
        streams = hc_post_batch(attention_batch(checkpoint, layer, hidden, captures), streams, post, comb)
        collapsed, post, comb = hc_batch(checkpoint, layer, "ffn", streams)
        hidden = norm_batch(collapsed, checkpoint.tensor(f"layers.{layer}.ffn_norm.weight"))
        streams = hc_post_batch(moe_batch(checkpoint, layer, hidden, token_ids), streams, post, comb)
        captures[f"layer.{layer}.output"] = streams.clone()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: value.detach().contiguous() for name, value in captures.items()}, output_path)


def write_layer3_hca_oracle(model_dir, output_path, token_ids):
    assert len(token_ids) == 128
    torch.set_num_threads(8)
    checkpoint = Checkpoint(model_dir)
    streams = torch.stack([checkpoint.row("embed.weight", token).float() for token in token_ids])
    streams = streams[:, None, :].repeat(1, 4, 1)
    captures = {}
    for layer in range(4):
        layer_captures = captures if layer == 3 else {}
        if layer == 3:
            captures["layer.3.input"] = streams.clone()
        collapsed, post, comb = hc_batch(checkpoint, layer, "attn", streams)
        hidden = norm_batch(collapsed, checkpoint.tensor(f"layers.{layer}.attn_norm.weight"))
        streams = hc_post_batch(
            attention_batch(checkpoint, layer, hidden, layer_captures), streams, post, comb
        )
        collapsed, post, comb = hc_batch(checkpoint, layer, "ffn", streams)
        hidden = norm_batch(collapsed, checkpoint.tensor(f"layers.{layer}.ffn_norm.weight"))
        streams = hc_post_batch(
            moe_batch(checkpoint, layer, hidden, token_ids, layer_captures), streams, post, comb
        )
    captures["layer.3.output"] = streams.clone()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: value.detach().contiguous() for name, value in captures.items()}, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers-0-2-output")
    parser.add_argument("--layer3-hca-output")
    parser.add_argument("--token-id", type=int, default=3)
    args = parser.parse_args()
    write_layer0_oracle(args.model, args.output, args.token_id)
    if args.layers_0_2_output:
        write_layers_0_2_oracle(args.model, args.layers_0_2_output, [3, 14, 15, 9])
    if args.layer3_hca_output:
        write_layer3_hca_oracle(args.model, args.layer3_hca_output, [3] * 128)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

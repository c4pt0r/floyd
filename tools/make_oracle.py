"""floyd oracle 生成器（对应 colibrì tools/make_glm_oracle.py，架构换成 deepseek_v3）。

tiny 模式: 随机权重的微型 DeepseekV3（真架构: MLA q_lora=null + sigmoid/noaux_tc 路由
           + 2 共享专家 + 首层 dense），保存 fixture 目录 + ref JSON。
real 模式: 加载本地 Moonlight-16B-A3B-Instruct（CPU, float32），同样输出 ref JSON。

ref schema: {"prompt_ids", "full_ids", "tf_pred"}  —— 引擎 TF=1/generate 直接消费。
"""
import argparse, json, torch


def build_tiny():
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM
    torch.manual_seed(1234)
    cfg = DeepseekV3Config(
        vocab_size=512, hidden_size=256,
        intermediate_size=512,          # 首层 dense MLP
        moe_intermediate_size=64,       # expert
        num_hidden_layers=4, first_k_dense_replace=1,
        num_attention_heads=4, num_key_value_heads=4,
        n_routed_experts=64, num_experts_per_tok=6, n_shared_experts=2,
        q_lora_rank=None,               # Moonlight 的关键差异点
        kv_lora_rank=64, qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
        n_group=1, topk_group=1, norm_topk_prob=True, routed_scaling_factor=2.446,
        scoring_func="sigmoid", topk_method="noaux_tc",
        rope_theta=50000.0, rope_scaling=None,
        tie_word_embeddings=False, rms_norm_eps=1e-5,
        max_position_embeddings=4096, attention_bias=False,
        num_nextn_predict_layers=0,
    )
    cfg._attn_implementation = "eager"
    model = DeepseekV3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.05)
        for layer in model.model.layers:
            if hasattr(layer.mlp, "gate") and hasattr(layer.mlp.gate, "e_score_correction_bias"):
                layer.mlp.gate.e_score_correction_bias.copy_(
                    torch.linspace(-0.1, 0.1, cfg.n_routed_experts))
    return model, cfg


def convert_experts(sd, moe_inter):
    """transformers >=5 的 state_dict 用 fused expert 张量:
       experts.gate_up_proj (E, 2*moe_inter, hidden) —— modeling_deepseek_v3.py (5.13.0) L211:
         gate, up = F.linear(x, gate_up_proj[e]).chunk(2, dim=-1)
         => 行 [:moe_inter] = gate_proj, 行 [moe_inter:] = up_proj
       experts.down_proj (E, hidden, moe_inter)      —— down_proj[e] 即 (hidden, moe_inter)
       引擎（和 Moonlight 4.46 checkpoint）要 per-expert:
         mlp.experts.E.{gate,up,down}_proj.weight
       其余张量名原样保留。返回新的 name->tensor dict。"""
    out = {}
    for n, t in sd.items():
        if n.endswith(".mlp.experts.gate_up_proj"):
            base = n[: -len(".gate_up_proj")]
            for e in range(t.shape[0]):
                out[f"{base}.{e}.gate_proj.weight"] = t[e, :moe_inter, :].contiguous().clone()
                out[f"{base}.{e}.up_proj.weight"] = t[e, moe_inter:, :].contiguous().clone()
        elif n.endswith(".mlp.experts.down_proj"):
            base = n[: -len(".down_proj")]
            for e in range(t.shape[0]):
                out[f"{base}.{e}.down_proj.weight"] = t[e].contiguous().clone()
        else:
            out[n] = t.contiguous().clone()
    return out


def verify_expert_split(model, cfg, sd_split, layer_idx=1, ntok=3):
    """数值验证 split 顺序: 用 split 后的 per-expert 张量手工复算 MoE 前向
       (sigmoid 路由 + e_score_correction_bias top-k 选择, norm_topk_prob 归一,
       routed_scaling_factor, + shared experts), 对比 layer.mlp(x)。不匹配即 split 反了。"""
    torch.manual_seed(4321)
    D, K = cfg.hidden_size, cfg.num_experts_per_tok
    x = torch.randn(1, ntok, D)
    mlp = model.model.layers[layer_idx].mlp
    with torch.no_grad():
        want = mlp(x)[0]                                        # (ntok, D)
        xf = x[0]
        scores = (xf.float() @ mlp.gate.weight.float().T).sigmoid()
        choice = scores + mlp.gate.e_score_correction_bias      # n_group=topk_group=1: 组掩码为恒等
        topk_idx = choice.topk(K, dim=-1).indices
        w = scores.gather(1, topk_idx)
        if cfg.norm_topk_prob:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        w = w * cfg.routed_scaling_factor
        P = f"model.layers.{layer_idx}.mlp"
        got = torch.zeros_like(xf)
        for t in range(ntok):
            for j in range(K):
                e = topk_idx[t, j].item()
                g = sd_split[f"{P}.experts.{e}.gate_proj.weight"]
                u = sd_split[f"{P}.experts.{e}.up_proj.weight"]
                d = sd_split[f"{P}.experts.{e}.down_proj.weight"]
                h = torch.nn.functional.silu(g @ xf[t]) * (u @ xf[t])
                got[t] += w[t, j] * (d @ h)
            sg = sd_split[f"{P}.shared_experts.gate_proj.weight"]
            su = sd_split[f"{P}.shared_experts.up_proj.weight"]
            sdn = sd_split[f"{P}.shared_experts.down_proj.weight"]
            got[t] += sdn @ (torch.nn.functional.silu(sg @ xf[t]) * (su @ xf[t]))
    diff = (got - want).abs().max().item()
    print(f"expert-split verify (layer {layer_idx}, {ntok} tok): max|diff| = {diff:.3e}")
    assert diff < 1e-4, f"split order WRONG: max abs diff {diff}"
    return diff


def make_ref(model, prompt_ids, ngen):
    ids = torch.tensor([prompt_ids])
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=ngen, do_sample=False, use_cache=True)
    full = out[0].tolist()
    with torch.no_grad():
        lg = model(torch.tensor([full]), use_cache=False).logits[0]
    return {"prompt_ids": prompt_ids, "full_ids": full, "tf_pred": lg.argmax(-1).tolist()}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("tiny"); t.add_argument("--out", default="fixture_tiny"); t.add_argument("--ref", default="ref_tiny.json"); t.add_argument("--ngen", type=int, default=20)
    r = sub.add_parser("real"); r.add_argument("--model", required=True); r.add_argument("--ref", default="ref_moonlight.json")
    r.add_argument("--prompt", default="What is the capital of France? Answer in one word."); r.add_argument("--ngen", type=int, default=20)
    a = ap.parse_args()

    if a.mode == "tiny":
        import os
        from safetensors.torch import save_file
        model, cfg = build_tiny()
        # transformers >=5 fuses expert weights (experts.gate_up_proj); 引擎要
        # Moonlight(4.46) 的 per-expert 布局, 所以自己转换并写 safetensors。
        sd = convert_experts(model.state_dict(), cfg.moe_intermediate_size)
        verify_expert_split(model, cfg, sd)
        for n, p in sd.items():
            e = n.split(".mlp.experts.")
            if len(e) == 2 and not e[1].startswith("0."):
                continue                        # 每层只打印 expert 0（其余 63 个同形状）
            print(f"  {n:60s} {tuple(p.shape)}")
        ref = make_ref(model, [3, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99], a.ngen)
        os.makedirs(a.out, exist_ok=True)
        save_file(sd, f"{a.out}/model.safetensors", metadata={"format": "pt"})
        cfg_dict = cfg.to_dict()
        if "rope_theta" not in cfg_dict:
            # newer transformers nests rope_theta under rope_parameters; the
            # C engine (Task 3) reads it top-level, so surface it explicitly.
            rp = cfg_dict.get("rope_parameters") or {}
            cfg_dict["rope_theta"] = rp.get("rope_theta", 50000.0)
        json.dump(cfg_dict, open(f"{a.out}/config.json", "w"), default=str)
        json.dump(ref, open(a.ref, "w"))
        print(f"\nprompt: {ref['prompt_ids']}\nfull  : {ref['full_ids']}\ntf    : {ref['tf_pred']}")
        print(f"salvato: {a.out}/ e {a.ref}")
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    how = "remote_code"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            a.model, trust_remote_code=True, dtype=torch.float32, device_map=None).eval()
    except Exception as e:
        print(f"[fallback] remote code fallito ({type(e).__name__}: {e}); provo il nativo transformers")
        from transformers import DeepseekV3ForCausalLM
        model = DeepseekV3ForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
        how = "native"
    msgs = [{"role": "user", "content": a.prompt}]
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    if hasattr(prompt_ids, "keys"):                    # transformers 5.x: BatchEncoding
        prompt_ids = prompt_ids["input_ids"]
    if prompt_ids and isinstance(prompt_ids[0], list): # batched form
        prompt_ids = prompt_ids[0]
    prompt_ids = [int(t) for t in prompt_ids]
    ref = make_ref(model, prompt_ids, a.ngen)
    ref["loader"] = how
    ref["text"] = tok.decode(ref["full_ids"][len(prompt_ids):])
    json.dump(ref, open(a.ref, "w"))
    print(f"loader={how}\nprompt({len(prompt_ids)} tok) -> gen: {ref['text']!r}\nsalvato: {a.ref}")


if __name__ == "__main__":
    main()

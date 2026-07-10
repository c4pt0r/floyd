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
        model, cfg = build_tiny()
        for n, p in model.state_dict().items():
            print(f"  {n:60s} {tuple(p.shape)}")
        ref = make_ref(model, [3, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99], a.ngen)
        model.save_pretrained(a.out, safe_serialization=True)
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
            a.model, trust_remote_code=True, torch_dtype=torch.float32, device_map=None).eval()
    except Exception as e:
        print(f"[fallback] remote code fallito ({type(e).__name__}: {e}); provo il nativo transformers")
        from transformers import DeepseekV3ForCausalLM
        model = DeepseekV3ForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
        how = "native"
    msgs = [{"role": "user", "content": a.prompt}]
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    ref = make_ref(model, prompt_ids, a.ngen)
    ref["loader"] = how
    ref["text"] = tok.decode(ref["full_ids"][len(prompt_ids):])
    json.dump(ref, open(a.ref, "w"))
    print(f"loader={how}\nprompt({len(prompt_ids)} tok) -> gen: {ref['text']!r}\nsalvato: {a.ref}")


if __name__ == "__main__":
    main()

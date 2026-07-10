# floyd — Moonlight 转换+对拍（pure C + Metal）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `~/floyd` 用 pure C 引擎（源自 colibrì glm.c）跑通 Moonlight-16B-A3B（deepseek_v3）的量化转换与 transformers oracle 逐 token 对拍，并加一个可选 Metal 加速后端。

**Architecture:** 拷贝 colibrì 的 `glm.c` 为 `floyd.c`，做两处架构适配（顶层 `rope_theta` 读取、`q_lora_rank=null` 时的 q 直投影分支）；fixture 用 transformers 原生 DeepseekV3 随机小模型直读（引擎支持未量化 safetensors 运行时量化），真模型走 BF16→int8/int4 离线转换器；Metal 后端以 C API + Objective-C 实现挂在 `matmul_qt` 的批量路径上。

**Tech Stack:** C11 + clang/libomp（macOS arm64，NEON/SDOT 路径已存在）、Objective-C + Metal（可选 `METAL=1`）、Python venv（torch CPU、transformers、safetensors——仅离线）。

## Global Constraints

- 运行时 pure C；Python 只用于离线转换与 oracle（venv：`~/floyd/.venv`，spec 规定）。
- 对拍精确性门槛以 **CPU 路径 + f32**（ebits=dbits=16）为准：fixture TF **32/32**、greedy **20/20**；量化/Metal 结果如实报数、不设门槛。
- 引擎调用形式（沿用 colibrì）：`SNAP=<模型目录> [TF=1] REF=<ref.json> ./floyd <cap> <ebits> <dbits>`。
- 构建：`make` 零新警告（保留 colibrì 既有 `-Wall -Wextra -Wno-…` 组合）；`METAL=1` 仅 macOS。
- Metal 启用失败必须启动时报错退出，禁止静默回退 CPU（spec 规定）。
- 真模型权重在 `~/floyd/models/Moonlight-16B-A3B-Instruct`（后台下载中，任务 b30vraavo），`models/`、`.venv/` 已在 `.gitignore`。
- 源 repo 引用路径：`/Users/dongxu/colibri`。每个 Task 结束提交一次，commit message 结尾加 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。

---

### Task 1: repo 骨架 + macOS 构建

**Files:**
- Create: `floyd.c`（拷贝自 `/Users/dongxu/colibri/c/glm.c`，仅本 Task 不改内容）
- Create: `st.h` `json.h` `compat.h` `tok.h` `tok_unicode.h`（原样拷贝）
- Create: `Makefile`
- Create: `tests/test_json.c` `tests/test_st.c`（原样拷贝）

**Interfaces:**
- Produces: 可执行 `./floyd`（行为暂与 glm 完全一致）；`make`、`make test-c` 目标。后续所有 Task 依赖此构建。

- [ ] **Step 1: 拷贝文件**

```bash
cd ~/floyd
cp /Users/dongxu/colibri/c/glm.c floyd.c
cp /Users/dongxu/colibri/c/{st.h,json.h,compat.h,tok.h,tok_unicode.h} .
mkdir -p tests && cp /Users/dongxu/colibri/c/tests/{test_json.c,test_st.c} tests/
```

- [ ] **Step 2: 写 Makefile**

以 colibrì `c/Makefile` 的 Darwin 分支为基础，目标改名 floyd，去掉 CUDA/olmoe/python 目标（`floyd.c` 里的 `#ifdef COLI_CUDA` 代码块保留不动——不定义宏即死代码，避免大面积手术）：

```make
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
CC      = clang
OMPDIR := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(OMPDIR),)
OMPC    = -Xclang -fopenmp -I$(OMPDIR)/include
OMPL    = -L$(OMPDIR)/lib -lomp
else
$(warning libomp non trovato: build SINGLE-THREAD. brew install libomp)
OMPC    =
OMPL    =
endif
CFLAGS  = -O3 $(OMPC) -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm $(OMPL)
else
CC      = gcc
ARCH   ?= native
CFLAGS  = -O3 -march=$(ARCH) -fopenmp -Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
LDFLAGS = -lm -fopenmp
endif

METAL    ?= 0
METAL_OBJ =
ifeq ($(METAL),1)
ifneq ($(UNAME_S),Darwin)
$(error METAL=1 e' supportato solo su macOS)
endif
CFLAGS  += -DFLOYD_METAL
METAL_OBJ = backend_metal.o
LDFLAGS += -framework Metal -framework Foundation
endif

TEST_BINS = tests/test_json tests/test_st

all: floyd

floyd: floyd.c st.h json.h tok.h tok_unicode.h compat.h $(METAL_OBJ)
	$(CC) $(CFLAGS) floyd.c $(METAL_OBJ) -o floyd $(LDFLAGS)

backend_metal.o: backend_metal.m backend_metal.h kernels_metal.h
	clang -O2 -fobjc-arc -c backend_metal.m -o $@

kernels_metal.h: kernels.metal
	xxd -i kernels.metal > kernels_metal.h

metal-test: backend_metal.o tests/test_backend_metal.c backend_metal.h
	clang -O2 $(OMPC) tests/test_backend_metal.c backend_metal.o -o tests/test_backend_metal \
		-framework Metal -framework Foundation -lm $(OMPL)
	./tests/test_backend_metal

tests/test_json: tests/test_json.c json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st: tests/test_st.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test-c: $(TEST_BINS)
	@for t in $(TEST_BINS); do ./$$t || exit 1; done

clean:
	rm -f floyd *.o kernels_metal.h tests/test_json tests/test_st tests/test_backend_metal

.PHONY: all test-c metal-test clean portable
```

注意 `metal-test` 等 Metal 目标本 Task 只是占位（文件还不存在，不会被 `all`/`test-c` 触发）。

- [ ] **Step 3: 构建并跑测试**

```bash
brew list libomp >/dev/null 2>&1 || brew install libomp
make && make test-c
```
Expected: `floyd` 二进制生成；test_json/test_st 全过。若 clang 报 x86 专属问题（不应该——glm.c 的 SIMD 都有 `#ifdef` 门），记录并修最小化。

- [ ] **Step 4: 冒烟**

```bash
SNAP=/nonexistent ./floyd 2>&1 | head -2
```
Expected: 报 `config.json` 打不开退出（证明 main 正常走通参数解析）。

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "task 1: skeleton from colibri glm.c, macOS build + C tests pass"
```

---

### Task 2: venv 依赖 + oracle/fixture 生成器

**Files:**
- Create: `tools/make_oracle.py`

**Interfaces:**
- Produces: `python tools/make_oracle.py tiny --out fixture_tiny --ref ref_tiny.json`（随机小模型 + 参考）；`python tools/make_oracle.py real --model <dir> --ref ref_moonlight.json --ngen 20`（真模型参考）。
- ref JSON schema（floyd.c 的 TF/generate 模式直接消费，沿用 colibrì）：`{"prompt_ids": [int], "full_ids": [int], "tf_pred": [int]}`，`tf_pred[i]` = 对 `full_ids` 整体 teacher-forcing 后位置 i 的 argmax。
- fixture 目录 = `save_pretrained` 的 safetensors + `config.json`（引擎直读未量化张量，运行时量化——colibrì 的 "oracolo tiny" 路径，无需转换器）。

- [ ] **Step 1: 安装 venv 依赖**

```bash
~/floyd/.venv/bin/pip install -q torch transformers safetensors numpy 'accelerate>=0.30'
~/floyd/.venv/bin/python -c "from transformers import DeepseekV3Config, DeepseekV3ForCausalLM; print('native deepseek_v3 OK')"
```
Expected: `native deepseek_v3 OK`。如 import 失败（旧 transformers），`pip install -U transformers` 后重试；仍失败则 fixture 也改用 Moonlight 仓库的 remote code（`AutoConfig.from_pretrained(<moonlight dir>, trust_remote_code=True)` 改小参数）——实现者在此分叉，优先原生。

- [ ] **Step 2: 写 `tools/make_oracle.py`**

```python
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
        json.dump(cfg.to_dict(), open(f"{a.out}/config.json", "w"), default=str)
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
```

- [ ] **Step 3: 生成 fixture 并检查产物**

```bash
cd ~/floyd && .venv/bin/python tools/make_oracle.py tiny
ls fixture_tiny/ && .venv/bin/python -c "
import json; r=json.load(open('ref_tiny.json'))
assert len(r['full_ids'])==len(r['prompt_ids'])+20==32 and len(r['tf_pred'])==32; print('ref OK: 32 pos')"
```
Expected: `model.safetensors + config.json`；`ref OK: 32 pos`。同时肉眼确认 state_dict 打印里有 `self_attn.q_proj.weight`（而非 q_a/q_b）——这是 Task 3 引擎分支要匹配的名字。若 config.json 里 `rope_theta` 不在顶层，记下实际键名（Task 3 Step 1 依赖）。

- [ ] **Step 4: Commit**

```bash
git add tools/make_oracle.py && git commit -m "task 2: deepseek_v3 tiny fixture + oracle generator (tiny/real)"
```

---

### Task 3: floyd.c 架构适配（rope_theta + q 直投影）→ fixture 对拍门槛

**Files:**
- Modify: `floyd.c`（三处，行号参照 colibrì glm.c 原位置）

**Interfaces:**
- Consumes: Task 2 的 `fixture_tiny/` + `ref_tiny.json`。
- Produces: 通过 `SNAP=fixture_tiny TF=1 REF=ref_tiny.json ./floyd 8 16 16` 的引擎；`q_lora==0` 语义 = q 直投影（后续 Task 全部依赖）。

- [ ] **Step 1: rope_theta 顶层读取（load_cfg，约 623 行）**

旧：
```c
    jval *rp=json_get(r,"rope_parameters"); jval *th=rp?json_get(rp,"rope_theta"):NULL;
    c->theta = th?(float)th->num:10000.f;
```
新：
```c
    jval *rp=json_get(r,"rope_parameters"); jval *th=rp?json_get(rp,"rope_theta"):NULL;
    if(!th) th=json_get(r,"rope_theta");   /* deepseek_v3/Moonlight: chiave top-level */
    c->theta = th?(float)th->num:10000.f;
```

- [ ] **Step 2: q 直投影加载分支（model_init，约 718-720 行）**

旧：
```c
        l->q_a   = qt_load(m,P("self_attn.q_a_proj.weight"), c->q_lora, D, dbits);
        l->q_a_ln= ld(m,P("self_attn.q_a_layernorm.weight"));
        l->q_b   = qt_load(m,P("self_attn.q_b_proj.weight"), H*c->qk_head, c->q_lora, dbits);
```
新：
```c
        if(c->q_lora>0){
            l->q_a   = qt_load(m,P("self_attn.q_a_proj.weight"), c->q_lora, D, dbits);
            l->q_a_ln= ld(m,P("self_attn.q_a_layernorm.weight"));
            l->q_b   = qt_load(m,P("self_attn.q_b_proj.weight"), H*c->qk_head, c->q_lora, dbits);
        } else {
            /* q_lora_rank=null (Moonlight/V2-Lite): proiezione q diretta, niente q_a/rmsnorm */
            l->q_b   = qt_load(m,P("self_attn.q_proj.weight"), H*c->qk_head, D, dbits);
        }
```
`l->q_a` 留在 calloc 的全零状态：先确认 `qt_bytes` 对全零 QT 返回 0（`grep -n 'qt_bytes' floyd.c` 检查实现，若按 `O*I` 计算则 0*0=0 天然安全）。

- [ ] **Step 3: q 直投影前向分支（attention，约 986-992 行）**

旧：
```c
        float *qresid=QR+(int64_t)s*c->q_lora;
        matmul_qt(qresid, xs, &l->q_a, 1);
        rmsnorm(qresid, qresid, l->q_a_ln, c->q_lora, c->eps);
        float *qfull=Q+(int64_t)s*H*qh; matmul_qt(qfull, qresid, &l->q_b, 1);
```
新：
```c
        float *qfull=Q+(int64_t)s*H*qh;
        if(c->q_lora>0){
            float *qresid=QR+(int64_t)s*c->q_lora;
            matmul_qt(qresid, xs, &l->q_a, 1);
            rmsnorm(qresid, qresid, l->q_a_ln, c->q_lora, c->eps);
            matmul_qt(qfull, qresid, &l->q_b, 1);
        } else {
            matmul_qt(qfull, xs, &l->q_b, 1);   /* q diretta: [H*qk_head, hidden] */
        }
```
同函数上方 `float *QR=falloc((int64_t)S*c->q_lora)` 改为 `falloc((int64_t)S*(c->q_lora>0?c->q_lora:1))`（防 0 长度分配的边界行为）。DSA 依赖 QR（`ix_wq` 内维是 q_lora）：在 has_dsa 检测（约 791 行）把条件加上 `c->q_lora>0 &&`——Moonlight 本来就无 indexer 权重，这是防御性收紧。

- [ ] **Step 4: 全量核对 q_a/q_b 其余引用**

```bash
grep -n 'q_a\|q_b' floyd.c
```
Expected 仅剩：MTP 加载块（has_mtp=0 时不执行，不改）、resident_bytes 统计（零 QT 记 0 字节，不改）、以上已改三处。有多出来的引用则逐个判断是否走 q_lora>0 守卫。

- [ ] **Step 5: 构建 + TF 对拍（核心门槛）**

```bash
make && SNAP=fixture_tiny TF=1 REF=ref_tiny.json ./floyd 8 16 16
```
Expected: `PREFILL (teacher-forcing) C vs oracolo: 32/32 posizioni`。
若不是 32/32：调用 superpowers:systematic-debugging，用 colibrì 方法论定位——先单层（fixture 改 num_hidden_layers=1 重新生成）对拍，Python 侧 `model.model.layers[0]` hook 中间激活 vs C 侧 printf，按 embed→q/kv 投影→rope→attention→router→expert 顺序二分。已知高危差异点：rope interleave 顺序、`norm_topk_prob` 的归一化时机、`routed_scaling_factor` 应用位置、sigmoid 路由的 bias 只影响选择不影响权重。

- [ ] **Step 6: greedy 生成对拍**

```bash
SNAP=fixture_tiny REF=ref_tiny.json ./floyd 8 16 16
```
Expected: `Token coincidenti: 20/20`。

- [ ] **Step 7: Commit**

```bash
git add floyd.c && git commit -m "task 3: deepseek_v3 adaptation (top-level rope_theta, direct q_proj); tiny fixture TF 32/32, greedy 20/20"
```

---

### Task 4: BF16 → int8/int4 转换器

**Files:**
- Create: `tools/convert_moonlight.py`（基于 `/Users/dongxu/colibri/c/tools/convert_fp8_to_int4.py` 的量化函数与容器格式）

**Interfaces:**
- Consumes: 本地 HF 目录（`*.safetensors` BF16 + config.json）。
- Produces: 容器目录：`out-NNNNN.safetensors`（打包字节张量 `name` U8 + 每行 scale `name.qs` F32，专家三矩阵按 gate/up/down 相邻写入）+ 原样拷贝的 `config.json`。`--ebits {8,4}` 专家位宽、`--dbits 8` 稠密位宽、embed/lm_head/所有 1D 张量存 F32 原样（引擎 io_bits 逻辑自己处理）。
- 用法：`python tools/convert_moonlight.py --indir <hf目录> --outdir <容器目录> --ebits 8 --dbits 8`。

- [ ] **Step 1: 写转换器**

从 colibrì 转换器拷 `quant_int8` / `quant_int4`（保持逐位一致的实现——不重写），新写主流程（BF16 无需 FP8 反量化；本地目录模式，无下载逻辑）：

```python
"""Moonlight (BF16 safetensors) -> container floyd/colibrì (int8/int4 per-row scale).
张量分类:
  - model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight -> ebits, 每专家三矩阵相邻写入
  - 其余 2D weight (attn/dense mlp/shared/router gate.weight)  -> dbits
  - embed_tokens / lm_head / 所有 1D (norm, e_score_correction_bias) -> F32 原样
容器: name = U8 打包字节, name+".qs" = F32 per-row scale (与 colibrì 引擎 qt_from_disk 一致)。
"""
import argparse, glob, json, os, re, shutil
import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

def quant_int8(w):
    s = np.abs(w).max(axis=1, keepdims=True) / 127.0 + 1e-12
    q = np.clip(np.round(w / s), -127, 127).astype(np.int8)
    return q.view(np.uint8).reshape(q.shape), s[:, 0].astype(np.float32)

def quant_int4(w):
    s = np.abs(w).max(axis=1, keepdims=True) / 7.0 + 1e-12
    q = np.clip(np.round(w / s), -7, 7).astype(np.int32) + 8      # 0..15
    O, I = q.shape
    if I % 2: q = np.concatenate([q, np.full((O, 1), 8, q.dtype)], axis=1)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed, s[:, 0].astype(np.float32)

EXPERT = re.compile(r"model\.layers\.\d+\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")

def convert(a):
    os.makedirs(a.outdir, exist_ok=True)
    shutil.copy(os.path.join(a.indir, "config.json"), a.outdir)
    shards = sorted(glob.glob(os.path.join(a.indir, "*.safetensors")))
    assert shards, f"nessun safetensors in {a.indir}"
    for i, sp in enumerate(shards):
        outp = os.path.join(a.outdir, f"out-{i:05d}.safetensors")
        if os.path.exists(outp):
            print(f"[{i+1}/{len(shards)}] {outp} esiste, skip (resume)"); continue
        out = {}
        with safe_open(sp, framework="pt") as f:
            names = sorted(f.keys())   # sorted: experts.E.{down,gate,up} 相邻 => slab 连续
            for name in names:
                t = f.get_tensor(name)
                if name.endswith(".weight") and t.dim() == 2 and "embed_tokens" not in name and "lm_head" not in name:
                    w = t.float().numpy()
                    bits = a.ebits if EXPERT.search(name) else a.dbits
                    qb, s = quant_int8(w) if bits == 8 else quant_int4(w)
                    out[name], out[name + ".qs"] = qb, s
                else:
                    out[name] = t.float().numpy()   # embed/lm_head/1D: F32 原样
        save_file(out, outp)
        print(f"[{i+1}/{len(shards)}] {os.path.basename(sp)} -> {outp} ({len(out)} tensori)", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True); ap.add_argument("--outdir", required=True)
    ap.add_argument("--ebits", type=int, default=8, choices=(4, 8))
    ap.add_argument("--dbits", type=int, default=8, choices=(4, 8))
    convert(ap.parse_args())
```

先核对 colibrì `quant_int8`/`quant_int4` 的实际数值约定（`sed -n '32,66p' /Users/dongxu/colibri/c/tools/convert_fp8_to_int4.py`）：int4 打包的 ±7+8 偏移、低位在前、奇数列补齐必须与引擎 `matmul_qt` fmt==2 的解包（`(b&0xF)-8`、`(b>>4)-8`）逐位一致——如与上面代码有出入，以 colibrì 为准改这里。

- [ ] **Step 2: 用 fixture 验证转换器（int8 往返）**

```bash
cd ~/floyd
.venv/bin/python tools/convert_moonlight.py --indir fixture_tiny --outdir fixture_tiny_i8 --ebits 8 --dbits 8
SNAP=fixture_tiny_i8 TF=1 REF=ref_tiny.json ./floyd 8 8 8
```
Expected: 引擎走"容器已量化"路径（无运行时量化提示），TF 结果记录之（int8 量化后可能 30-32/32，如实记下——门槛仍是 Task 3 的 f32 32/32）。若崩溃/张量缺失，先查 `.qs` 命名与专家 slab 读取路径。

- [ ] **Step 3: Commit**

```bash
git add tools/convert_moonlight.py && git commit -m "task 4: BF16->int8/int4 container converter, fixture roundtrip validated"
```

---

### Task 5: 真 Moonlight——oracle、转换、对拍报告

**Files:**
- Create: `ref_moonlight.json`（不进 git——`.gitignore` 已排除 `ref*.json`）
- Create: `docs/parity-report.md`（进 git 的结果报告）

**Interfaces:**
- Consumes: `models/Moonlight-16B-A3B-Instruct`（后台下载任务 b30vraavo 完成后）、Task 2 real 模式、Task 4 转换器。
- Produces: int8 与 int4 容器目录 `models/moonlight_i8` `models/moonlight_i4`；TF 匹配率与 greedy 匹配数报告。

- [ ] **Step 1: 确认下载完成**

```bash
ls -la ~/floyd/models/Moonlight-16B-A3B-Instruct/*.safetensors | wc -l
tail -3 ~/floyd/models/download.log
```
Expected: 全部 shard 就位（HF 页面 core safetensors 共 ~31.9 GB）。未完成则等待/重跑同一条 `hf download`（断点续传）。

- [ ] **Step 2: 生成真模型 oracle（CPU f32，约 64 GB RAM，本机 512 GB）**

```bash
cd ~/floyd && .venv/bin/nohup true; .venv/bin/python tools/make_oracle.py real \
  --model models/Moonlight-16B-A3B-Instruct --ref ref_moonlight.json --ngen 20 \
  > oracle_real.log 2>&1 &
```
Expected（完成后）: `ref_moonlight.json` 就位，log 里 `loader=remote_code`（或 fallback `native`）+ 生成文本可读（说明 oracle 自身是清醒的）。CPU f32 的 16B 生成 20 token 可能要几十分钟——期间并行做 Step 3。

- [ ] **Step 3: 转换 int8 与 int4 容器**

```bash
.venv/bin/python tools/convert_moonlight.py --indir models/Moonlight-16B-A3B-Instruct --outdir models/moonlight_i8 --ebits 8 --dbits 8
.venv/bin/python tools/convert_moonlight.py --indir models/Moonlight-16B-A3B-Instruct --outdir models/moonlight_i4 --ebits 4 --dbits 8
du -sh models/moonlight_i8 models/moonlight_i4
```
Expected: i8 ≈ 16-17 GB、i4 ≈ 9-10 GB（专家 int4 + 稠密 int8 + f32 embed/head）。

- [ ] **Step 4: 真权重对拍（TF + greedy，int8 与 int4）**

```bash
SNAP=models/moonlight_i8 TF=1 REF=ref_moonlight.json ./floyd 64 8 8
SNAP=models/moonlight_i8       REF=ref_moonlight.json ./floyd 64 8 8
SNAP=models/moonlight_i4 TF=1 REF=ref_moonlight.json ./floyd 64 4 8
SNAP=models/moonlight_i4       REF=ref_moonlight.json ./floyd 64 4 8
```
Expected: 四个数字（int8 TF x/N、int8 greedy y/20、int4 TF、int4 greedy）。int8 预期接近满分；int4 如实记录。任何段错误/维度报错都回 fixture 复现后修。

- [ ] **Step 5: 写报告并 Commit**

`docs/parity-report.md`：机器（M-系列/512GB）、模型 revision、oracle loader 路径、四个匹配数、TF pos/s、RSS、容器大小。

```bash
git add docs/parity-report.md && git commit -m "task 5: real Moonlight parity report (int8/int4 TF + greedy vs f32 oracle)"
```

---

### Task 6: Metal 后端（内核 + 单测）

**Files:**
- Create: `kernels.metal` `backend_metal.h` `backend_metal.m` `tests/test_backend_metal.c`
- Modify: `Makefile`（Task 1 已放好目标，无改动即可用）

**Interfaces:**
- Produces（Task 7 消费，签名精确如下）:
```c
int  fm_init(void);   /* 1 = device+kernel 就绪; 0 = 不可用（调用方决定报错退出） */
const char *fm_device_name(void);
void fm_matmul_q8(float *y, const float *x, const int8_t  *w, const float *s, int O, int I, int S);
void fm_matmul_q4(float *y, const float *x, const uint8_t *w, const float *s, int O, int I, int S);
/* 语义与 CPU matmul_qt fmt1/fmt2 相同: y[S,O], x[S,I], w 每行 scale s[O]; int4 低半字节在前, -8 偏移 */
```
- 权重 MTLBuffer 按 `w` 指针在后端内部缓存（首用一次 `newBufferWithBytes` 拷贝，之后复用——对应 colibrì CUDA "lazy upload once"；统一内存下真 zero-copy 留作测量后的优化，报告里注明）。

- [ ] **Step 1: 写 `kernels.metal`**

```metal
#include <metal_stdlib>
using namespace metal;

kernel void matmul_q8(device const char  *w [[buffer(0)]],
                      device const float *s [[buffer(1)]],
                      device const float *x [[buffer(2)]],
                      device float       *y [[buffer(3)]],
                      constant uint3     &d [[buffer(4)]],   // O, I, S
                      uint2 g [[thread_position_in_grid]]) {
    uint o = g.x, t = g.y;
    if (o >= d.x || t >= d.z) return;
    device const char  *wr = w + (ulong)o * d.y;
    device const float *xr = x + (ulong)t * d.y;
    float acc = 0.0f;
    for (uint i = 0; i < d.y; i++) acc += (float)wr[i] * xr[i];
    y[(ulong)t * d.x + o] = acc * s[o];
}

kernel void matmul_q4(device const uchar *w [[buffer(0)]],
                      device const float *s [[buffer(1)]],
                      device const float *x [[buffer(2)]],
                      device float       *y [[buffer(3)]],
                      constant uint3     &d [[buffer(4)]],
                      uint2 g [[thread_position_in_grid]]) {
    uint o = g.x, t = g.y;
    if (o >= d.x || t >= d.z) return;
    uint Ib = (d.y + 1) / 2;
    device const uchar *wr = w + (ulong)o * Ib;
    device const float *xr = x + (ulong)t * d.y;
    float acc = 0.0f;
    for (uint i = 0; i + 1 < d.y; i += 2) {
        uchar b = wr[i >> 1];
        acc += (float)((int)(b & 0xF) - 8) * xr[i] + (float)((int)(b >> 4) - 8) * xr[i + 1];
    }
    if (d.y & 1) { uchar b = wr[d.y >> 1]; acc += (float)((int)(b & 0xF) - 8) * xr[d.y - 1]; }
    y[(ulong)t * d.x + o] = acc * s[o];
}
```

- [ ] **Step 2: 写 `backend_metal.h`**（接口块的四个声明 + include guard + `#include <stdint.h>`，如上）

- [ ] **Step 3: 写 `backend_metal.m`**

```objc
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "backend_metal.h"
#include "kernels_metal.h"      /* xxd -i kernels.metal: unsigned char kernels_metal[], kernels_metal_len */

static id<MTLDevice> g_dev; static id<MTLCommandQueue> g_q;
static id<MTLComputePipelineState> g_p8, g_p4;
#define FM_WCACHE 4096
static struct { const void *host; id<MTLBuffer> buf; } g_wc[FM_WCACHE];
static int g_nwc;

int fm_init(void) {
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) return 0;
        g_q = [g_dev newCommandQueue];
        NSString *src = [[NSString alloc] initWithBytes:kernels_metal length:kernels_metal_len encoding:NSUTF8StringEncoding];
        NSError *err = nil;
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { fprintf(stderr, "[METAL] compile: %s\n", err.localizedDescription.UTF8String); return 0; }
        g_p8 = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"matmul_q8"] error:&err];
        g_p4 = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"matmul_q4"] error:&err];
        return (g_p8 && g_p4) ? 1 : 0;
    }
}
const char *fm_device_name(void) { return g_dev ? g_dev.name.UTF8String : "(none)"; }

static id<MTLBuffer> wbuf(const void *host, size_t len) {
    for (int i = 0; i < g_nwc; i++) if (g_wc[i].host == host) return g_wc[i].buf;
    id<MTLBuffer> b = [g_dev newBufferWithBytes:host length:len options:MTLResourceStorageModeShared];
    if (g_nwc < FM_WCACHE) { g_wc[g_nwc].host = host; g_wc[g_nwc].buf = b; g_nwc++; }
    return b;
}

static void run(id<MTLComputePipelineState> p, const void *w, size_t wlen,
                const float *s, const float *x, float *y, int O, int I, int S) {
    @autoreleasepool {
        id<MTLBuffer> bw = wbuf(w, wlen);
        id<MTLBuffer> bs = wbuf(s, (size_t)O * 4);
        id<MTLBuffer> bx = [g_dev newBufferWithBytes:x length:(size_t)S * I * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [g_dev newBufferWithLength:(size_t)S * O * 4 options:MTLResourceStorageModeShared];
        uint32_t d[3] = { (uint32_t)O, (uint32_t)I, (uint32_t)S };
        id<MTLCommandBuffer> cb = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:bw offset:0 atIndex:0]; [e setBuffer:bs offset:0 atIndex:1];
        [e setBuffer:bx offset:0 atIndex:2]; [e setBuffer:by offset:0 atIndex:3];
        [e setBytes:d length:sizeof(d) atIndex:4];
        MTLSize grid = MTLSizeMake(O, S, 1);
        NSUInteger tw = p.maxTotalThreadsPerThreadgroup > 256 ? 256 : p.maxTotalThreadsPerThreadgroup;
        [e dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        memcpy(y, by.contents, (size_t)S * O * 4);
    }
}
void fm_matmul_q8(float *y, const float *x, const int8_t *w, const float *s, int O, int I, int S)
{ run(g_p8, w, (size_t)O * I, s, x, y, O, I, S); }
void fm_matmul_q4(float *y, const float *x, const uint8_t *w, const float *s, int O, int I, int S)
{ run(g_p4, w, (size_t)O * ((I + 1) / 2), s, x, y, O, I, S); }
```

- [ ] **Step 4: 写 `tests/test_backend_metal.c`（先跑，预期失败在缺文件时）**

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "../backend_metal.h"

static float frand(void){ return (float)rand()/RAND_MAX*2.f-1.f; }
int main(void){
    if(!fm_init()){ fprintf(stderr,"metal non disponibile\n"); return 1; }
    printf("device: %s\n", fm_device_name());
    srand(42);
    int O=128, I=257, S=4;                      /* I dispari: esercita il tail q4 */
    int8_t *w8=malloc((size_t)O*I); float *s=malloc(O*4), *x=malloc((size_t)S*I*4);
    for(int i=0;i<O*I;i++) w8[i]=rand()%255-127;
    for(int o=0;o<O;o++) s[o]=frand()*0.01f+0.02f;
    for(int i=0;i<S*I;i++) x[i]=frand();
    float *y=malloc((size_t)S*O*4);
    fm_matmul_q8(y,x,w8,s,O,I,S);
    double maxrel=0;
    for(int t=0;t<S;t++) for(int o=0;o<O;o++){
        double ref=0; for(int i=0;i<I;i++) ref+=(double)w8[(size_t)o*I+i]*x[(size_t)t*I+i];
        ref*=s[o];
        double d=fabs(y[(size_t)t*O+o]-ref), rel=d/(fabs(ref)+1e-6);
        if(rel>maxrel) maxrel=rel;
    }
    printf("q8 max rel err: %.2e\n", maxrel);
    if(maxrel>1e-3){ printf("FAIL q8\n"); return 1; }
    int Ib=(I+1)/2; uint8_t *w4=malloc((size_t)O*Ib);
    for(int i=0;i<O*Ib;i++) w4[i]=rand()&0xFF;
    fm_matmul_q4(y,x,w4,s,O,I,S);
    maxrel=0;
    for(int t=0;t<S;t++) for(int o=0;o<O;o++){
        double ref=0;
        for(int i=0;i+1<I;i+=2){ uint8_t b=w4[(size_t)o*Ib+(i>>1)];
            ref+=((int)(b&0xF)-8)*(double)x[(size_t)t*I+i]+((int)(b>>4)-8)*(double)x[(size_t)t*I+i+1]; }
        if(I&1){ uint8_t b=w4[(size_t)o*Ib+(I>>1)]; ref+=((int)(b&0xF)-8)*(double)x[(size_t)t*I+I-1]; }
        ref*=s[o];
        double d=fabs(y[(size_t)t*O+o]-ref), rel=d/(fabs(ref)+1e-6);
        if(rel>maxrel) maxrel=rel;
    }
    printf("q4 max rel err: %.2e\n", maxrel);
    if(maxrel>1e-3){ printf("FAIL q4\n"); return 1; }
    printf("OK\n"); return 0;
}
```

- [ ] **Step 5: 跑内核单测**

```bash
make metal-test
```
Expected: `device: Apple M…`、两个 `max rel err` ≤ 1e-3、`OK`。

- [ ] **Step 6: Commit**

```bash
git add kernels.metal backend_metal.h backend_metal.m tests/test_backend_metal.c
git commit -m "task 6: Metal backend (q8/q4 matmul kernels) + kernel-level unit test vs CPU reference"
```

---

### Task 7: 引擎 Metal 挂钩 + CPU/METAL A/B

**Files:**
- Modify: `floyd.c`（matmul_qt 顶部 + main 初始化，约 446 / 2294 行）

**Interfaces:**
- Consumes: Task 6 的 `fm_*` API。
- Produces: `FLOYD_METAL=1` 启用 GPU 批量 matmul（`S>=FM_MIN_S`，默认 8，即 prefill/TF 路径）；不可用时启动即报错退出。

- [ ] **Step 1: main 初始化（main 开头、SNAP 检查之后）**

```c
#ifdef FLOYD_METAL
    g_metal = (getenv("FLOYD_METAL") && atoi(getenv("FLOYD_METAL"))) ? 1 : 0;
    g_metal_min_s = getenv("FM_MIN_S") ? atoi(getenv("FM_MIN_S")) : 8;
    if(g_metal){
        if(!fm_init()){ fprintf(stderr,"[METAL] richiesto ma non disponibile\n"); return 2; }
        fprintf(stderr,"[METAL] attivo: %s (batch S>=%d)\n", fm_device_name(), g_metal_min_s);
    }
#else
    if(getenv("FLOYD_METAL") && atoi(getenv("FLOYD_METAL"))){
        fprintf(stderr,"FLOYD_METAL richiede: make METAL=1\n"); return 2; }
#endif
```
文件顶部（AVX2/NEON include 区附近）加：
```c
#ifdef FLOYD_METAL
#include "backend_metal.h"
static int g_metal=0, g_metal_min_s=8;
#endif
```

- [ ] **Step 2: matmul_qt 挂钩（约 446 行函数体开头）**

```c
static void matmul_qt(float *y, const float *x, QT *w, int S){
#ifdef FLOYD_METAL
    if(g_metal && S>=g_metal_min_s && (w->fmt==1 || w->fmt==2)){
        if(w->fmt==1) fm_matmul_q8(y,x,w->q8,w->s,w->O,w->I,S);
        else          fm_matmul_q4(y,x,w->q4,w->s,w->O,w->I,S);
        return;
    }
#endif
```
注意：挂钩必须放在任何 int8-activation 整数路径分派之前，直接短路；专家 slab 里的 QT 是 slab 内的"视图"，`w->q8/q4/s` 指针照常有效——但 LRU 逐出后 slab 复用会让权重缓存失效！防御：挂钩条件加 `&& w->cuda_eligible_metal`——不引新字段的简化方案：**只对稠密常驻张量启用**，即在 Task 6 的 `wbuf` 缓存满（FM_WCACHE）后直接每次拷贝不缓存是不够的。正确且最小的方案：`fm_matmul_*` 的权重缓存键改为 `(host指针, 长度)` 之外再由调用方传 `cache` 布尔——修改接口：

```c
void fm_matmul_q8(float *y, const float *x, const int8_t  *w, const float *s, int O, int I, int S, int cache);
void fm_matmul_q4(float *y, const float *x, const uint8_t *w, const float *s, int O, int I, int S, int cache);
```
（Task 6 实现与单测同步用此签名；`cache=0` → 每次 `newBufferWithBytes` 临时上传。）挂钩处：稠密张量 `cache=1`；判断"是专家 slab 视图"的最简办法：attention/共享专家/router 调用点都是稠密——直接看 QT 是否属于 `m->L[i]` 不可行（matmul_qt 无 m）——用启发式：`cache = (w->slab_backed==0)`。`QT` 结构体（约 68 行）加一个 `int slab_backed;` 字段，`expert_load` 里构造视图 QT 时置 1（grep `expert_load` 内的 QT 赋值处，加一行）。这是本 Task 唯一的结构体改动。

- [ ] **Step 3: 构建两个版本，fixture A/B**

```bash
make clean && make && SNAP=fixture_tiny_i8 TF=1 REF=ref_tiny.json ./floyd 8 8 8   # CPU 基线
make clean && make METAL=1
SNAP=fixture_tiny_i8 TF=1 REF=ref_tiny.json FLOYD_METAL=1 ./floyd 8 8 8            # Metal
SNAP=fixture_tiny_i8 TF=1 REF=ref_tiny.json ./floyd 8 8 8                          # METAL 构建但未启用 -> 仍是 CPU
```
Expected: 三次 TF 分数一致（浮点归约顺序差异理论上可翻平票 argmax——若差 1 个位置，逐位置 diff 确认是平票即记录为已知差异）；`[METAL] attivo: Apple M…` 出现在第二次。

- [ ] **Step 4: 真模型 A/B（int8 容器）**

```bash
SNAP=models/moonlight_i8 TF=1 REF=ref_moonlight.json ./floyd 64 8 8
SNAP=models/moonlight_i8 TF=1 REF=ref_moonlight.json FLOYD_METAL=1 ./floyd 64 8 8
```
Expected: 匹配数一致（±平票），记录两者 `pos/s` ——诚实报告 Metal 在 TF 批量路径上是否更快（首次含上传成本，跑两遍取第二遍）。

- [ ] **Step 5: Commit**

```bash
git add floyd.c backend_metal.h backend_metal.m tests/test_backend_metal.c
git commit -m "task 7: FLOYD_METAL hook on batch matmul path (dense cached, expert slab uncached); CPU/Metal TF A/B"
```

---

### Task 8: README + 收尾

**Files:**
- Create: `README.md`
- Modify: `docs/parity-report.md`（补 Metal A/B 数字）

- [ ] **Step 1: 写 README**

内容（诚实数字风格，向 colibrì 致谢并链接）：项目是什么（Moonlight-16B-A3B 的 colibrì 式 pure C 引擎 + Metal）、与 colibrì 的关系与差异（deepseek_v3 适配两点）、快速开始（venv → make_oracle tiny → make → TF 对拍 → 真模型三步）、对拍结果表（fixture 32/32 & 20/20、真模型 int8/int4 TF/greedy、Metal A/B）、已知限制（无 tokenizer/chat、无 MTP/DSA、Metal 只挂批量路径）。

- [ ] **Step 2: 全链路复跑（验证 README 里的命令照抄能跑）**

```bash
cd ~/floyd && make clean && make && make test-c
SNAP=fixture_tiny TF=1 REF=ref_tiny.json ./floyd 8 16 16
SNAP=fixture_tiny REF=ref_tiny.json ./floyd 8 16 16
```
Expected: 32/32、20/20 复现。

- [ ] **Step 3: Commit**

```bash
git add README.md docs/parity-report.md && git commit -m "task 8: README + final parity/Metal report"
```

---

## Self-Review 记录

- **Spec 覆盖**：转换（Task 4/5）、对拍两级（Task 3 门槛 + Task 5 真模型）、Metal（Task 6/7）、错误处理（转换 resume、Metal 硬失败、q_lora 诊断在引擎 CKR 内）、README（Task 8）——spec 各节均有对应任务。spec 中"零拷贝 bytesNoCopy"在 Task 6 落地为"首用一次拷贝 + 缓存"（malloc 指针页对齐无保证，bytesNoCopy 有页对齐+页整长硬约束）；偏差已在 Task 6 接口说明与报告要求中注明，属测量后再优化项。
- **占位符**：无 TBD/TODO；所有代码块完整。
- **类型一致性**：`fm_matmul_*` 签名在 Task 6 定义与 Task 7 挂钩处已统一为带 `cache` 参数的 8 参版本（Task 6 单测调用处传 `cache=0` 对应更新——实现者在 Task 6 Step 4 写测试时用 8 参签名：`fm_matmul_q8(y,x,w8,s,O,I,S,0)`）。

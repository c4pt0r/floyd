r"""Moonlight (BF16/F32 safetensors) -> container floyd/colibri (int8/int4 per-row scale).

Tensor classification (matches colibri's classify() in
/Users/dongxu/colibri/c/tools/convert_fp8_to_int4.py, adapted for a local BF16/F32
checkout with no FP8 dequant and no DSA-indexer/MTP tensors, which Moonlight-16B-A3B
does not have -- config.json has num_nextn_predict_layers=0 and no indexer keys):

  - model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight -> ebits, per-expert three
    matrices adjacent in `names` (sorted) so the engine's coalesced expert pread can
    treat them as one contiguous slab (perf only, see expert_load() fallback in floyd.c).
  - other 2D ".weight" tensors (attn / dense mlp / shared-expert mlp) -> dbits.
  - model.embed_tokens.weight / lm_head.weight (2D) -> F32 raw, unquantized. The engine's
    qt_load() picks io_bits = dbits>=8 ? 16 : dbits, and qt_alloc() maps bits>=16 to
    fmt=0 (raw f32 read, no .qs expected) -- so these tensors must NOT get a ".qs"
    sibling or the engine will auto-detect a quantized fmt from the byte count and
    silently treat embed/lm_head as low precision.
  - model.layers.N.mlp.gate.weight (the MoE router, 2D!) -> F32 raw, NOT quantized.
    floyd.c loads it with `l->router=ld(m,P("mlp.gate.weight"))`: ld() is the raw-f32
    1D reader (see floyd.c ~line 692) and has no ".qs" support at all -- it would read
    packed/scaled bytes as if they were raw floats and silently corrupt the router.
    Confirmed with `grep -n 'router=ld\|=ld(m,' floyd.c`: the ONLY 2D tensor ever passed
    to ld() is mlp.gate.weight: every other ld() target (norms, e_score_correction_bias,
    k_norm, enorm/hnorm, ...) is already 1D and excluded by the dim()==2 check below.
    This matches colibri's classify(): `if name.endswith("mlp.gate.weight"): return "f32"`.
  - all other 1D tensors (norms, e_score_correction_bias, rotary_emb.inv_freq, ...) ->
    F32 raw (never quantized; the engine either reads them with ld() or ignores them).

Container: name = U8 packed bytes, name+".qs" = F32 per-row scale, exactly what the
engine's qt_from_disk()/qt_load() expect (see floyd.c: `st_has(&m->S, name+".qs")` picks
the quantized-container path; byte count alone selects int8 (nb==O*I) vs int4
(nb==O*ceil(I/2))).
"""
import argparse, glob, os, re, shutil
import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

# ---------- quantizzazione: copiata bit-per-bit da colibri
# (sed -n '32,66p' /Users/dongxu/colibri/c/tools/convert_fp8_to_int4.py), NON riscritta.
# Deve restare numericamente identica a quanto il motore C si aspetta in unpack:
#   int8 (fmt==1): valore = int8_t grezzo * scale (nessun offset)
#   int4 (fmt==2): nibble BASSO = indice pari, nibble ALTO = indice dispari, valore
#                  memorizzato come q+8 (q in [-8,7]); il motore fa (b&0xF)-8 / (b>>4)-8
#                  (vedi floyd.c: qt_addrow/qt_matvec_rows/embed_row, fmt==2).
def quant_int8(w, bits):                       # w: [O,I] f32 -> (qbytes U8 [O*I], scale f32 [O])
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -qmax - 1, qmax).astype(np.int8)
    return q.reshape(-1).view(np.uint8).copy(), s[:, 0].astype(np.float32)

def quant_int4(w, bits):                        # -> (qbytes U8 [O*ceil(I/2)], scale f32 [O])
    O, I = w.shape
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = np.maximum(amax / qmax, 1e-8)
    q = np.clip(np.rint(w / s), -8, qmax).astype(np.int32)  # nibble [-8,7]
    rb = (I + 1) // 2
    out = np.zeros((O, rb), np.uint8)
    v0 = (q[:, 0::2] + 8).astype(np.uint8)
    out[:, :v0.shape[1]] = v0
    if I > 1:
        v1 = (q[:, 1::2] + 8).astype(np.uint8)
        out[:, :v1.shape[1]] |= (v1 << 4)
    return out.reshape(-1), s[:, 0].astype(np.float32)

EXPERT = re.compile(r"model\.layers\.\d+\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
ROUTER = re.compile(r"mlp\.gate\.weight$")     # router (noaux_tc): NON gate_proj, tenuto F32 (ld(), no .qs)

def is_quantizable(name, t):
    """2D .weight tensor that should be packed to ebits/dbits; everything else -> F32 raw."""
    if not (name.endswith(".weight") and t.dim() == 2):
        return False
    if "embed_tokens" in name or name == "lm_head.weight":
        return False
    if ROUTER.search(name):
        return False
    return True

def convert(a):
    os.makedirs(a.outdir, exist_ok=True)
    cfg_src = os.path.join(a.indir, "config.json")
    if os.path.exists(cfg_src):
        shutil.copy(cfg_src, a.outdir)
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
                if is_quantizable(name, t):
                    w = t.float().numpy()
                    bits = a.ebits if EXPERT.search(name) else a.dbits
                    qb, s = quant_int8(w, bits) if bits == 8 else quant_int4(w, bits)
                    out[name], out[name + ".qs"] = qb, s
                else:
                    out[name] = t.float().numpy()   # embed/lm_head/router/1D: F32 原样
        save_file(out, outp)
        print(f"[{i+1}/{len(shards)}] {os.path.basename(sp)} -> {outp} ({len(out)} tensori)", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True); ap.add_argument("--outdir", required=True)
    ap.add_argument("--ebits", type=int, default=8, choices=(4, 8))
    ap.add_argument("--dbits", type=int, default=8, choices=(4, 8))
    convert(ap.parse_args())

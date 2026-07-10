"""Genera tok_unicode.h: tabelle di range per le classi Unicode usate dal
pre-tokenizer cl100k (regex del tokenizer GLM-5.2) e dal pre-tokenizer moonshot:
  - \\p{L}  lettere   (categoria Unicode che inizia per 'L')
  - \\p{N}  numeri    (categoria che inizia per 'N')
  - \\s     whitespace (proprieta' Unicode White_Space)
  - \\p{Han}            script Han (CJK unificato, via range costanti)
  - [\\p{Lu}\\p{Lt}]    lettere maiuscole/titlecase (Lu unione Lt)
  - \\p{Ll}             lettere minuscole
  - \\p{Lm}             lettere modificatrici
  - \\p{Lo}             altre lettere (senza maiuscola/minuscola, es. CJK/Hangul/Arabo)
  - [\\p{Mn}\\p{Mc}\\p{Me}]  marchi combinanti (Mn unione Mc unione Me)
Ogni classe diventa un array ordinato di range [lo,hi] inclusivi; il C fa ricerca
binaria. Eseguire una volta: python3 tools/gen_unicode.py > tok_unicode.h

NOTA: uni_L/uni_N/uni_S (le tre tabelle originali, usate dal tokenizer GLM) sono
generate con la stessa logica di sempre (categoria/whitespace via unicodedata) e
DEVONO restare identiche byte-per-byte a ogni rigenerazione con lo stesso
unicodedata.unidata_version del file esistente: se si rigenera con un Python che
porta una versione Unicode piu' recente, uni_L/uni_N cambieranno (nuovi codepoint
assegnati) e vanno NON usati per sovrascrivere le tabelle esistenti — solo le
tabelle nuove (uni_Han/uni_LuLt/uni_Ll/uni_Lm/uni_Lo/uni_M) vanno innestate nel
tok_unicode.h esistente. Vedi chat-task-1-report.md per i dettagli di questa
rigenerazione.
"""
import sys, random, unicodedata

WHITE_SPACE = {0x09,0x0A,0x0B,0x0C,0x0D,0x20,0x85,0xA0,0x1680,
    0x2000,0x2001,0x2002,0x2003,0x2004,0x2005,0x2006,0x2007,0x2008,0x2009,0x200A,
    0x2028,0x2029,0x202F,0x205F,0x3000}

def ranges(pred):
    out=[]; lo=None
    for cp in range(0x110000):
        if 0xD800<=cp<=0xDFFF:        # surrogati: mai
            if lo is not None: out.append((lo,cp-1)); lo=None
            continue
        if pred(cp):
            if lo is None: lo=cp
        else:
            if lo is not None: out.append((lo,cp-1)); lo=None
    if lo is not None: out.append((lo,0x10FFFF))
    return out

def cat(cp):
    try: return unicodedata.category(chr(cp))
    except ValueError: return "Cn"

L = ranges(lambda c: cat(c).startswith("L"))
N = ranges(lambda c: cat(c).startswith("N"))
S = ranges(lambda c: c in WHITE_SPACE)

# moonshot pat_str 需要的子类（unicodedata.category 前缀匹配）
def cat_pred(prefixes):
    return lambda cp: any(unicodedata.category(chr(cp)).startswith(p) for p in prefixes)

# \p{Han}: unicodedata 没有 script 属性 —— 用 CJK 统一表意区间常量（Unicode 16 Scripts.txt 的 Han 主区间）
HAN_RANGES = [(0x2E80,0x2E99),(0x2E9B,0x2EF3),(0x2F00,0x2FD5),(0x3005,0x3005),(0x3007,0x3007),
    (0x3021,0x3029),(0x3038,0x303B),(0x3400,0x4DBF),(0x4E00,0x9FFF),(0xF900,0xFA6D),
    (0xFA70,0xFAD9),(0x20000,0x2A6DF),(0x2A700,0x2B739),(0x2B740,0x2B81D),(0x2B820,0x2CEA1),
    (0x2CEB0,0x2EBE0),(0x2EBF0,0x2EE5D),(0x2F800,0x2FA1D),(0x30000,0x3134A),(0x31350,0x323AF)]

Han   = HAN_RANGES
LuLt  = ranges(cat_pred(("Lu","Lt")))
Ll    = ranges(cat_pred(("Ll",)))
Lm    = ranges(cat_pred(("Lm",)))
Lo    = ranges(cat_pred(("Lo",)))
M     = ranges(cat_pred(("Mn","Mc","Me")))

def emit(name, rs):
    print(f"static const uint32_t {name}[][2] = {{")
    for i in range(0,len(rs),6):
        chunk="".join(f"{{0x{lo:X},0x{hi:X}}}," for lo,hi in rs[i:i+6])
        print("    "+chunk)
    print("};")
    print(f"static const int {name}_n = {len(rs)};\n")

def in_ranges(cp, rs):
    for lo, hi in rs:
        if lo <= cp <= hi: return True
    return False

def validate_sampling(n_samples=2000):
    """Campiona n_samples codepoint per ciascuna classe nuova e confronta il
    predicato usato per costruire le tabelle con il modulo 'regex' (supporta
    \\p{...} sulle proprieta' Unicode complete, a differenza di 're')."""
    try:
        import regex
    except ImportError:
        print("WARN: modulo 'regex' non disponibile — salto la validazione a campione", file=sys.stderr)
        return
    checks = [
        ("Han",  lambda cp: in_ranges(cp, Han),  r'\p{Han}'),
        ("LuLt", lambda cp: in_ranges(cp, LuLt), r'[\p{Lu}\p{Lt}]'),
        ("Ll",   lambda cp: in_ranges(cp, Ll),   r'\p{Ll}'),
        ("Lm",   lambda cp: in_ranges(cp, Lm),   r'\p{Lm}'),
        ("Lo",   lambda cp: in_ranges(cp, Lo),   r'\p{Lo}'),
        ("M",    lambda cp: in_ranges(cp, M),    r'[\p{Mn}\p{Mc}\p{Me}]'),
    ]
    random.seed(20260710)
    all_ok = True
    for name, pred, pat in checks:
        rx = regex.compile(pat)
        mismatches = 0
        checked = 0
        while checked < n_samples:
            cp = random.randint(0, 0x10FFFF)
            if 0xD800 <= cp <= 0xDFFF:
                continue
            checked += 1
            ours = pred(cp)
            theirs = bool(rx.match(chr(cp)))
            if ours != theirs:
                mismatches += 1
                if mismatches <= 5:
                    print(f"  MISMATCH {name}: U+{cp:04X} ours={ours} regex={theirs}", file=sys.stderr)
        ok = mismatches == 0
        all_ok = all_ok and ok
        print(f"[validate] {name}: {checked} sampled, {mismatches} mismatches -> {'OK' if ok else 'FAIL'}", file=sys.stderr)
    if not all_ok:
        print("[validate] ONE OR MORE CLASSES FAILED SAMPLING VALIDATION", file=sys.stderr)

validate_sampling()

print("/* GENERATO da tools/gen_unicode.py — non modificare a mano. */")
print("#ifndef TOK_UNICODE_H\n#define TOK_UNICODE_H\n#include <stdint.h>\n")
emit("uni_L", L); emit("uni_N", N); emit("uni_S", S)
emit("uni_Han", Han); emit("uni_LuLt", LuLt); emit("uni_Ll", Ll)
emit("uni_Lm", Lm); emit("uni_Lo", Lo); emit("uni_M", M)
print("""static int uni_in(const uint32_t t[][2], int n, uint32_t cp){
    int lo=0, hi=n-1;
    while(lo<=hi){ int m=(lo+hi)>>1;
        if(cp<t[m][0]) hi=m-1; else if(cp>t[m][1]) lo=m+1; else return 1; }
    return 0;
}
static inline int is_L(uint32_t c){ return uni_in(uni_L,uni_L_n,c); }
static inline int is_N(uint32_t c){ return uni_in(uni_N,uni_N_n,c); }
static inline int is_S(uint32_t c){ return uni_in(uni_S,uni_S_n,c); }
static inline int is_Han(uint32_t c){ return uni_in(uni_Han,uni_Han_n,c); }
static inline int is_LuLt(uint32_t c){ return uni_in(uni_LuLt,uni_LuLt_n,c); }
static inline int is_Ll(uint32_t c){ return uni_in(uni_Ll,uni_Ll_n,c); }
static inline int is_Lm(uint32_t c){ return uni_in(uni_Lm,uni_Lm_n,c); }
static inline int is_Lo(uint32_t c){ return uni_in(uni_Lo,uni_Lo_n,c); }
static inline int is_M(uint32_t c){ return uni_in(uni_M,uni_M_n,c); }
#endif""")

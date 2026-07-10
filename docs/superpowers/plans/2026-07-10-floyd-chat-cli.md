# floyd chat + CLI flags 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `./floyd chat --model <dir>` 与 Moonlight 真实对话：C 原生 tiktoken tokenizer + moonshot 模板 + 多轮 KV 复用，外加正规的子命令/flags CLI。

**Architecture:** 新头文件 `tok_moon.h`（tiktoken.model 直读 + moonshot pre-tokenizer + rank 合并 BPE），复用 tok.h 的 hmap/UTF-8 基础设施与 `tok_unicode.h` 扩展后的字符类表；`run_chat` 仿 `run_text`/`run_serve` 现有模式（step/spec_decode/EmitStream 流式/KV 持久化）；CLI 为 main() 顶部的 flags→setenv 薄适配层，内部 getenv 逻辑零改动。

**Tech Stack:** C11（无新依赖）、Python venv 仅离线（生成对拍用例、Unicode 表）。

## Global Constraints

- 运行时纯 C；Python 仅离线工具（venv `/Users/dongxu/floyd/.venv`）。
- **tokenizer 精确门槛**：`tok_cases.json` 全部用例（≥40 条 + 3 轮模板序列）C vs AutoTokenizer 逐 id 精确匹配，decode 往返字节精确——任何一例不过 = 任务不过。
- 用户文本编码**永不匹配特殊 token**；模板用显式 special id 拼接。
- 停止 token 来自 `generation_config.json` 的 `eos_token_id`（Moonlight 为 [163586]=`<|im_end|>`），读取失败回退 `mtok_special("<|im_end|>")`。
- 兼容性：老调用形式 `SNAP=… [TF=1 REF=…] ./floyd <cap> <ebits> <dbits>` 必须继续逐字节一致地工作（README/parity-report 里的命令不失效）。
- tok.h / GLM 路径零行为变化。`make` 与 `make METAL=1` 零新警告。
- 调试开关保持 env-only：IDOT、DSA、MTP、PILOT、LOOKA、STATS、PIN、TOPP（专家 top-p）、SERVE、REPLAY、SCORE 等。
- 每个 Task 结束提交；commit message 结尾 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`。
- 参考文件：模型目录 `/Users/dongxu/floyd/models/Moonlight-16B-A3B-Instruct/`（tiktoken.model 163,584 行 base64、tokenization_moonshot.py 的 pat_str、tokenizer_config.json 的 added_tokens_decoder 与 chat_template、generation_config.json）。

---

### Task 1: Unicode 表扩展（Han/Lu∪Lt/Ll/Lm/Lo/M）

**Files:**
- Create: `tools/gen_unicode.py`（以 `/Users/dongxu/colibri/c/tools/gen_unicode.py` 为基线扩展，56 行的小文件）
- Modify: `tok_unicode.h`（重新生成，追加新表；现有 uni_L/uni_N/空白表必须逐字节不变）

**Interfaces:**
- Produces（Task 3 消费）：`static inline int is_Han(uint32_t c)`、`is_LuLt(c)`、`is_Ll(c)`、`is_Lm(c)`、`is_Lo(c)`、`is_M(c)`（M = Mn∪Mc∪Me），与现有 `is_L/is_N/is_S` 同风格（区间表 + `uni_in` 二分）。

- [ ] **Step 1: 拷贝并扩展生成器**

拷贝 colibri 的 gen_unicode.py 到 `tools/`，在现有 L/N/WS 三类之后追加：

```python
# moonshot pat_str 需要的子类（unicodedata.category 前缀匹配）
def cat_pred(prefixes):
    return lambda cp: any(unicodedata.category(chr(cp)).startswith(p) for p in prefixes)

# \p{Han}: unicodedata 没有 script 属性 —— 用 CJK 统一表意区间常量（Unicode 16 Scripts.txt 的 Han 主区间）
HAN_RANGES = [(0x2E80,0x2E99),(0x2E9B,0x2EF3),(0x2F00,0x2FD5),(0x3005,0x3005),(0x3007,0x3007),
    (0x3021,0x3029),(0x3038,0x303B),(0x3400,0x4DBF),(0x4E00,0x9FFF),(0xF900,0xFA6D),
    (0xFA70,0xFAD9),(0x20000,0x2A6DF),(0x2A700,0x2B739),(0x2B740,0x2B81D),(0x2B820,0x2CEA1),
    (0x2CEB0,0x2EBE0),(0x2EBF0,0x2EE5D),(0x2F800,0x2FA1D),(0x30000,0x3134A),(0x31350,0x323AF)]

emit("uni_Han", None, HAN_RANGES)                # 直接输出常量区间
emit("uni_LuLt", cat_pred(("Lu","Lt")))
emit("uni_Ll",  cat_pred(("Ll",)))
emit("uni_Lm",  cat_pred(("Lm",)))
emit("uni_Lo",  cat_pred(("Lo",)))
emit("uni_M",   cat_pred(("Mn","Mc","Me")))
```

（`emit` 沿用原文件的 ranges()+打印逻辑；给它加一个可选的显式区间参数。）在文件尾部生成对应的 `is_*` inline 函数，与现有 `is_L` 同款。

**生成器内置抽样验证**：若 venv 里可 `import regex`，对每个新类抽样 2000 个码点断言 `regex.match(rf'[\p{{{name}}}]', chr(cp))` 与表一致（Han 用 `\p{Han}`，LuLt 用 `[\p{Lu}\p{Lt}]` 等）；不可导入则打印跳过警告。

- [ ] **Step 2: 安装 regex 并重新生成**

```bash
cd ~/floyd && .venv/bin/pip install -q regex
.venv/bin/python tools/gen_unicode.py > tok_unicode.h.new
diff <(sed -n '/uni_L\[\]/,/uni_L_n/p' tok_unicode.h) <(sed -n '/uni_L\[\]/,/uni_L_n/p' tok_unicode.h.new)
```
Expected: 抽样验证全过；旧 uni_L 表 diff 为空（逐字节不变）。然后 `mv tok_unicode.h.new tok_unicode.h`。

- [ ] **Step 3: 编译回归**

```bash
make clean && make && make test-c && SNAP=fixture_tiny TF=1 REF=ref_tiny.json ./floyd 8 16 16
```
Expected: 零警告、test-c 过、TF 仍 32/32（GLM 路径无扰动）。

- [ ] **Step 4: Commit**

```bash
git add tools/gen_unicode.py tok_unicode.h
git commit -m "chat 1: unicode tables for moonshot pre-tokenizer (Han/LuLt/Ll/Lm/Lo/M)"
```

---

### Task 2: 对拍用例生成器

**Files:**
- Create: `tools/make_tok_cases.py`
- Output（gitignored，`*.json` 已覆盖…注意 `.gitignore` 有 `ref*.json`，无通配 json——本文件输出 `tok_cases.json`，把它加进 `.gitignore`）

**Interfaces:**
- Produces：`tok_cases.json`，schema：
```json
{ "cases": [ {"text": "...", "ids": [..]} , ...],
  "template": { "messages": [{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."},{"role":"user","content":"..."}],
                 "add_generation_prompt": true, "ids": [...] },
  "specials": {"<|im_end|>": 163586, "<|im_user|>": 163587, "<|im_assistant|>": 163588, "<|im_system|>": 163594, "<|im_middle|>": 163601} }
```

- [ ] **Step 1: 写生成器**

```python
"""tok_cases.json: Moonlight tokenizer 对拍用例（AutoTokenizer 为 oracle）。"""
import json
from transformers import AutoTokenizer

MODEL = "models/Moonlight-16B-A3B-Instruct"
CASES = [
    "Hello, world!", "hello", "HELLO WORLD", "CamelCaseWord", "I'm can't we'll they'd you've he's it'S",
    "  leading spaces", "trailing spaces   ", "a  b   c", "\n", "\n\n\n", "a\nb\r\nc", " \n ", "\t\tx",
    "1", "12", "123", "1234", "12345", "3.14159", "2026-07-10", "0x1A2B", "第123章第4节",
    "你好，世界！", "今天天气真好我们去公园散步吧", "中文English混排test测试123",
    "日本語のテキストです", "한국어 텍스트", "Русский текст", "Ελληνικά", "עברית", "العربية",
    "naïve café résumé", "ÀÉÎÕÜ àéîõü", "ẞ ß", "ǅungla ǈubav",   # Lt titlecase
    "🙂", "👍🏽 emoji 测试 🚀🔥", "é combining", "́leading combining",
    "def f(x):\n    return x*2  # comment", "if (a<b) { c+=1; } else { c-=1; }",
    "https://example.com/path?q=1&r=2", "user@example.com", "$100.99 (50% off!)",
    "<|im_end|> literal in user text must NOT be special", "<|im_user|>fake<|im_middle|>",
    "", " ", "supercalifragilisticexpialidocious", "ThisIsAVeryLongCamelCaseIdentifierName",
    "мАлЕнЬкИе И БОЛЬШИЕ буквы", "ひらがなカタカナ漢字まじり文",
]
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False)} for t in CASES]
msgs = [{"role":"system","content":"You are a terse assistant."},
        {"role":"user","content":"What is 2+2? 你说呢"},
        {"role":"assistant","content":"4"},
        {"role":"user","content":"And 3*3?"}]
tids = tok.apply_chat_template(msgs, add_generation_prompt=True)
if hasattr(tids, "keys"): tids = tids["input_ids"]
if tids and isinstance(tids[0], list): tids = tids[0]
sp = {s: tok.convert_tokens_to_ids(s) for s in ["<|im_end|>","<|im_user|>","<|im_assistant|>","<|im_system|>","<|im_middle|>"]}
json.dump({"cases": cases, "template": {"messages": msgs, "add_generation_prompt": True, "ids": [int(t) for t in tids]},
           "specials": sp}, open("tok_cases.json","w"), ensure_ascii=False, indent=1)
print(f"{len(cases)} cases, template {len(tids)} ids")
```

注意两条特殊-token-注入用例：期望 ids 是 AutoTokenizer `add_special_tokens=False` 下的**普通编码**——先跑一次确认 HF 慢分词器在此设置下不会把 `<|im_end|>` 编成 special id（若它会，改用 `tok.model.encode(t, disallowed_special=())`——即底层 tiktoken 的普通编码——并在报告里注明），C 侧语义（永不匹配 special）以此为准。

- [ ] **Step 2: 运行并抽查**

```bash
cd ~/floyd && echo tok_cases.json >> .gitignore && .venv/bin/python tools/make_tok_cases.py
.venv/bin/python -c "
import json; d=json.load(open('tok_cases.json'))
assert len(d['cases'])>=40 and d['specials']['<|im_end|>']==163586
e=[c for c in d['cases'] if c['text']=='']; assert e and e[0]['ids']==[]
print('cases OK:', len(d['cases']), 'template ids:', len(d['template']['ids']))"
```
Expected: `cases OK: 5x template ids: NN`。

- [ ] **Step 3: Commit**

```bash
git add tools/make_tok_cases.py .gitignore && git commit -m "chat 2: tokenizer parity case generator (AutoTokenizer oracle)"
```

---

### Task 3: `tok_moon.h` — C tiktoken tokenizer + 精确门槛

**Files:**
- Create: `tok_moon.h`
- Create: `tests/test_tok_moon.c`
- Modify: `Makefile`（追加 test-tok 目标）、`.gitignore`（`tests/test_tok_moon`）

**Interfaces:**
- Consumes：tok.h 的 `hmap/hm_init/hm_put/hm_get`、`u8_next`；tok_unicode.h 的 `is_L/is_N/is_S/is_Han/is_LuLt/is_Ll/is_Lm/is_Lo/is_M`。`tok_moon.h` 必须在 tok.h 之后 include。
- Produces（Task 4/5 消费，签名精确）：
```c
typedef struct { unsigned char *b; int len; } MEnt;
typedef struct { char *str; int len; int id; } MSpecial;
typedef struct {
    hmap rank;              /* 字节序列 -> id（0..n_base-1，rank 即 id） */
    MEnt *id2b; int n_base; /* id -> 字节序列（decode） */
    MSpecial *sp; int nsp;  /* added_tokens_decoder（含 reserved），按 id 升序 */
    int n_ids;              /* n_base + nsp */
} MTok;
void mtok_load(MTok *T, const char *model_dir);            /* 失败: 报错文件+行号, exit(1) */
int  mtok_encode(MTok *T, const char *text, int len, int *out, int max);  /* 纯文本; special 永不匹配 */
int  mtok_decode(MTok *T, const int *ids, int n, char *out, int max);     /* 返回字节数, out 不保证 NUL 安全时截断 */
int  mtok_special(MTok *T, const char *content);           /* 内容查 id; -1 缺失 */
```

- [ ] **Step 1: 先写失败测试**

`tests/test_tok_moon.c`：argv[1]=模型目录、argv[2]=tok_cases.json。用 json.h 解析 cases；对每条：`mtok_encode` 结果与期望 ids 逐一比较，失败打印**文本、期望、实际**（十进制 id 列表）并计数；再 `mtok_decode(期望 ids)` 与原文本 memcmp（模板与注入用例除外——注入用例 decode 出的就是字面文本，一样比）。模板用例在 Task 4 的构造器就位前先只做 decode 方向。最后 `printf("tokenizer parity: %d/%d\n", ok, tot)`，非全过 exit 1。

```c
#include <stdio.h>
#include "../json.h"
#include "../tok.h"        /* hmap, u8_next */
#include "../tok_moon.h"
/* main: 读 tok_cases.json, 逐例 encode/decode 对拍, 全过=0 否则 1 (完整代码由实现者按上述行为写, 断言必须真实比较) */
```

Makefile 追加（Darwin/Linux 通用区）：
```make
tests/test_tok_moon: tests/test_tok_moon.c tok_moon.h tok.h tok_unicode.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
test-tok: tests/test_tok_moon
	./tests/test_tok_moon models/Moonlight-16B-A3B-Instruct tok_cases.json
```
（`test-tok` 不加入 `test-c`：它依赖模型目录与生成的用例，非 hermetic。）

- [ ] **Step 2: 跑测试确认失败**

Run: `make test-tok` — Expected: 编译失败（tok_moon.h 不存在）。

- [ ] **Step 3: 实现 `tok_moon.h`**

三块，全部手写、无新依赖：

**(a) 加载**。`tiktoken.model`：逐行 `base64 rank`——手写 base64 解码（标准字母表，~25 行）；断言 rank 与行号一致、连续（否则报错行号）。hm_init 容量取 ≥2×行数的 2 的幂（163,584 → 524288）。`tokenizer_config.json`：json.h 解析 `added_tokens_decoder`（对象：id 字符串 → {content}），收进 sp[]；校验最小 special id == n_base（163584），否则报错。

**(b) BPE（tiktoken rank 合并）**。对 pre-tokenizer 给出的字节片段 `p[a..b)`：
```c
/* parts = 每字节一段; 反复找相邻对 (i,i+1) 使 rank(拼接) 最小, 合并; 无可合并即输出各段 id */
static void mbpe_piece(MTok *T, const unsigned char *p, int a, int b, int *out, int *no, int max){
    int n=b-a; if(n<=0) return;
    if(n==1){ int id=hm_get(&T->rank,(const char*)p+a,1); if(id>=0&&*no<max) out[(*no)++]=id; return; }
    /* 整段已是 token 的快捷路径 */
    { int id=hm_get(&T->rank,(const char*)p+a,n); if(id>=0){ if(*no<max) out[(*no)++]=id; return; } }
    int *st=malloc((n+1)*sizeof(int));            /* 段起点(相对 a), st[k]..st[k+1] */
    for(int i=0;i<=n;i++) st[i]=i; int ns=n;
    for(;;){
        int best=-1, bestrank=INT_MAX;
        for(int i=0;i<ns-1;i++){
            int r=hm_get(&T->rank,(const char*)p+a+st[i],st[i+2]-st[i]);
            if(r>=0 && r<bestrank){ bestrank=r; best=i; }
        }
        if(best<0) break;
        memmove(st+best+1, st+best+2, (ns-best-1)*sizeof(int)); ns--;
    }
    for(int i=0;i<ns;i++){ int id=hm_get(&T->rank,(const char*)p+a+st[i],st[i+1]-st[i]);
        if(id>=0 && *no<max) out[(*no)++]=id; }
    free(st);
}
```

**(c) moonshot pre-tokenizer**。仿 tok.h `pretok_chunk` 的结构（码点数组 + 字节 offset 数组），候选式**按 pat_str 顺序**，每位置取第一个成功匹配。类谓词：`#define UPC(c) ((is_LuLt(c)||is_Lm(c)||is_Lo(c)||is_M(c)) && !is_Han(c))`、`#define LOC(c) ((is_Ll(c)||is_Lm(c)||is_Lo(c)||is_M(c)) && !is_Han(c))`。

```
1) [\p{Han}]+                    : is_Han 连串。
2) [^\r\n\p{L}\p{N}]? U* l+ 缩写? : 可选前缀字符 pre（非\r\n、非L、非N —— 注意 M 类不属于 L！
   谓词写成 !is_L(c)&&!is_N(c)&&!ISNL(c) 且它不是 U/l 起点更优情形——直接按正则语义：
   先试带前缀（若当前字符满足 pre 且其后能匹配 U*l+），再试不带前缀）。
   U*l+ 的手写规则（U 与 l 有交集 Lm/Lo/M，需模拟回溯）：
     设 R 为从当前位置起 (UPC∪LOC) 的最大连串, uend = R 中最大的 k 使 R[0..k) 全 UPC。
     若 uend < |R| 且 R[uend] ∈ LOC: 匹配到 l-run 结束（j 从 uend 起 while LOC(cp[j]) j++）。
     否则若 uend ≥ 1 且 R[uend-1] ∈ LOC: 匹配 = uend（U* 回退让出末尾给 l+）。
     否则本候选失败。
   缩写后缀 (?i:'s|'t|'re|'ve|'m|'ll|'d)：匹配成功后若紧跟 ' + 对应字母（大小写不敏感，
   照抄 tok.h 的 LOW 宏逻辑）则并入。
3) [^\r\n\p{L}\p{N}]? U+ l* 缩写? : 同上但 U 至少 1 个、l 可零个：j 取 UPC 连串（≥1），
   继续吃 LOC 连串，再试缩写后缀。
4) \p{N}{1,3}                    : 照抄 tok.h 候选 3。
5)  ?[^\s\p{L}\p{N}]+[\r\n]*     : 照抄 tok.h 候选 4。
6) \s*[\r\n]+ / 7) \s+(?!\S) / 8) \s+ : 照抄 tok.h 候选 5/6 的实现（语义相同）。
```

**注意与 cl100k 的差异**：没有独立的缩写候选式（`'s` 只能作为 2/3 的后缀）；字母类拆成大小写两式且排除 Han；M（组合记号）算词内字符。**`mtok_encode` 不做 special 扫描**（与 tok_encode 不同——用户文本里的 `<|im_end|>` 走候选 5 当普通标点+字母编码）。

`mtok_decode`：id < n_base → 拷贝 id2b 字节；id ≥ n_base → 在 sp[] 里二分/线性找 content 输出字面量；未知 id 跳过。

- [ ] **Step 4: 跑通门槛（迭代到全过）**

```bash
make test-tok
```
Expected: `tokenizer parity: N/N`（全过）。不过时逐例排查——最可能的偏差点按概率排序：U*l+ 回溯规则、数字 {1,3} 分组、`\s+(?!\S)` 边界、缩写大小写、组合记号 M 的归类、Han 区间遗漏（生成器抽样验证过的类基本可信）。**禁止**为过例而在 C 侧塞特例表——修的是规则本身；若发现是 HAN_RANGES 常量不全，回 Task 1 的生成器补区间重新生成。

- [ ] **Step 5: 编译回归 + Commit**

```bash
make clean && make && make test-c
git add tok_moon.h tests/test_tok_moon.c Makefile .gitignore
git commit -m "chat 3: native C tiktoken tokenizer (moonshot pre-tokenizer + rank BPE), parity gate green"
```

---

### Task 4: moonshot 模板 + `run_chat`

**Files:**
- Modify: `floyd.c`（新增 `run_chat` + 模板辅助函数，紧邻 run_text；main() 增加 CHAT 分支）
- Modify: `tests/test_tok_moon.c`（启用模板对拍用例）

**Interfaces:**
- Consumes：`mtok_*`（Task 3）、`step/spec_decode/stops_arm/kv_alloc/kv_disk_*`（现有）。
- Produces：`static void run_chat(Model *m, const char *snap)`；模板构造器（test 也用，所以放 tok_moon.h 里）：
```c
/* 追加模板 ids 到 out（返回新 no）。role_txt/content 为 UTF-8 纯文本。 */
int mtok_tmpl_msg(MTok *T, const char *role, const char *content, int *out, int no, int max);
/* = sp(<|im_{role}|>) + encode(role) + sp(<|im_middle|>) + encode(content) + sp(<|im_end|>) */
int mtok_tmpl_genprompt(MTok *T, int *out, int no, int max);
/* = sp(<|im_assistant|>) + encode("assistant") + sp(<|im_middle|>) */
```
role→special 映射：system→`<|im_system|>`、user→`<|im_user|>`、assistant→`<|im_assistant|>`。

- [ ] **Step 1: 模板对拍测试先行**

test_tok_moon.c 启用 template 用例：按 `tok_cases.json` 的 messages 依序调 `mtok_tmpl_msg`（首条非 system 时先注入默认 system 块——本用例首条就是 system，不触发），末尾 `mtok_tmpl_genprompt`，与期望 ids 逐一比较。先跑：Expected 编译失败（函数未定义）。

- [ ] **Step 2: 实现模板构造器（tok_moon.h）**

按 chat_template 逐字复刻（见 spec；注意 role 文本本身也是 encode 出来的普通 token）。实现后 `make test-tok` Expected 全过（含模板）。

- [ ] **Step 3: 实现 `run_chat`**

结构照 run_serve 的骨架裁剪（去 \x01 协议、去 \x02PROMPT API 模式）：

```c
static void run_chat(Model *m, const char *snap){
    MTok T; mtok_load(&T, snap[0]?snap:".");     /* --model 目录即模型目录 */
    /* 停止 token: generation_config.json 的 eos_token_id (数或数组), 回退 <|im_end|> */
    int eos = -1; { /* json.h 读 <snap>/generation_config.json, 解析失败或缺键则 eos=-1 */ }
    if(eos<0) eos = mtok_special(&T, "<|im_end|>");
    if(eos<0){ fprintf(stderr,"tokenizer senza <|im_end|>\n"); exit(1); }
    stops_arm(&m->c, eos);
    if(g_temp<0) g_temp=0.7f;  g_nuc = getenv("NUCLEUS")?g_nuc:0.90f;
    int ngen=getenv("NGEN")?atoi(getenv("NGEN")):512;
    int maxctx=getenv("CTX")?atoi(getenv("CTX")):4096;
    const char *systxt=getenv("SYSTEM")?getenv("SYSTEM"):"You are a helpful assistant";
    kv_alloc(m,maxctx);
    int len=0, *hist=malloc(maxctx*sizeof(int));
    g_kvsave=getenv("KVSAVE")?atoi(getenv("KVSAVE")):1;
    snprintf(g_kv_path,sizeof(g_kv_path),"%s/.coli_kv",snap);
    { int r=kv_disk_load(m,hist,maxctx); if(r>0) len=r; }
    fprintf(stderr,"floyd chat — :reset azzera, :exit esce\n");
    char *line=NULL; size_t cap=0; ssize_t nr;
    for(;;){
        fputs("\n› ",stdout); fflush(stdout);
        if((nr=getline(&line,&cap,stdin))<=0) break;
        if(line[nr-1]=='\n') line[--nr]=0;
        if(!strcmp(line,":exit")) break;
        if(!strcmp(line,":reset")){ len=0; kv_disk_reset(); puts("(reset)"); continue; }
        if(!nr) continue;
        int no=len;
        if(len==0) no=mtok_tmpl_msg(&T,"system",systxt,hist,no,maxctx);
        no=mtok_tmpl_msg(&T,"user",line,hist,no,maxctx);
        no=mtok_tmpl_genprompt(&T,hist,no,maxctx);
        if(no+ngen+g_draft+2>=maxctx || no<0){ fprintf(stderr,"(contesto pieno: :reset)\n"); continue; }
        int k=no-len;
        float *logit=step(m,hist+len,k,len); len=no;
        fputs("◆ ",stdout);
        MoonEmit me={&T,now_s(),0};
        int prod=spec_decode(m,hist,len,ngen,eos,emit_moon,&me,&len);
        double dt=now_s()-me.t0;
        fprintf(stderr,"\n[%d tok, %.2f tok/s | ctx %d/%d | RSS %.2f GB]\n",prod,prod/dt,len,maxctx,rss_gb());
        kv_disk_append(m,hist,len);
    }
    free(hist);
}
```

`MoonEmit`/`emit_moon`：仿 EmitStream/emit_stream，但解码走 `mtok_decode` 且**不打印停止 token 的字面量**（id==eos 直接跳过输出）；原始字节 fwrite 到 stdout（UTF-8 跨 token 拼接由终端处理）。检查 spec_decode 对 hist/len 的追加约定与 serve 用法一致（照抄 serve 的调用形态：logits 来自 step，`&len` 传入）。溢出保护：`mtok_tmpl_msg` 在 max 不够时返回 -1，调用方拒绝该轮。

main() 增加分支（PROMPT 分支旁）：`if(getenv("CHAT")){ run_chat(&m,snap); if(stats) stats_dump(&m,stats); return 0; }`。

- [ ] **Step 4: 编译 + 冒烟（fixture 级）**

fixture 没有 tiktoken.model —— 冒烟直接用真模型（i8 容器目录没有 tokenizer 文件！`--model` 是容器目录 models/moonlight_i8…tiktoken.model 在 HF 目录）。**决策：转换器已把 config.json 拷进容器；chat 需要 tokenizer 三件套也在容器里**——本 Step 把 `tiktoken.model`、`tokenizer_config.json`、`generation_config.json` 拷入 `models/moonlight_i8/`（一次性 cp，并在 Task 6 给 convert_moonlight.py 加自动拷贝 + README 注明）：

```bash
cp models/Moonlight-16B-A3B-Instruct/{tiktoken.model,tokenizer_config.json,generation_config.json} models/moonlight_i8/
make && printf 'What is the capital of France? One word.\n:exit\n' | SNAP=models/moonlight_i8 CHAT=1 ./floyd 64 8 8
```
Expected: 流式输出含 "Paris"，在 `<|im_end|>` 干净停止（无字面 `<|im_end|>` 泄漏），退出码 0。

- [ ] **Step 5: Commit**

```bash
git add floyd.c tok_moon.h tests/test_tok_moon.c
git commit -m "chat 4: moonshot template builders + run_chat (streaming, multi-turn KV, <|im_end|> stop)"
```

---

### Task 5: CLI 子命令与 flags

**Files:**
- Modify: `floyd.c`（main() 顶部 flags→setenv 适配层 + usage()）
- Modify: `tools/convert_moonlight.py`（顺手：转换时自动拷贝 tiktoken.model/tokenizer_config.json/generation_config.json 到 outdir，若存在）

**Interfaces:**
- Produces：`floyd chat|run|tf|gen|help` + flags（见下表）。老式 `SNAP=… ./floyd 64 8 8` 走原路径不变（argv[1] 是数字或无参 → 跳过适配层）。

- [ ] **Step 1: 写 usage() 与解析器**

```c
static void usage(int code){
    fputs(
"floyd — Moonlight-16B-A3B in pure C\n"
"uso: floyd <comando> [flags] | (legacy) SNAP=<dir> floyd <cap> <ebits> <dbits>\n\n"
"comandi:\n"
"  chat   conversazione interattiva     floyd chat --model DIR\n"
"  run    generazione singola           floyd run  --model DIR --prompt \"...\"\n"
"  tf     teacher-forcing vs oracolo    floyd tf   --model DIR --ref ref.json\n"
"  gen    greedy vs oracolo             floyd gen  --model DIR --ref ref.json\n"
"  help   questo testo\n\n"
"flags globali:  --model DIR (obbligatorio) | --cap N (64) | --ebits 4|8|16 (8)\n"
"  --dbits 4|8|16 (8) | --ram GB | --metal\n"
"chat/run:  --ngen N (chat:512, run:256) | --ctx N (4096) | --temp T (0.7)\n"
"  --top-p P (0.90) | --system \"...\" | --prompt \"...\" (run) | --draft N | --no-kvsave\n\n"
"variabili d'ambiente: interfaccia legacy/debug (IDOT, DSA, MTP, PILOT, STATS, ...)\n",
    code?stderr:stdout); exit(code);
}
```

解析：argv[1] ∈ {chat,run,tf,gen,help} 才进入；`help`→usage(0)。循环 i=2..argc：`--model→setenv("SNAP")`、`--ref→REF`、`--ngen→NGEN`、`--ctx→CTX`、`--temp→TEMP`、`--top-p→NUCLEUS`、`--system→SYSTEM`、`--prompt→PROMPT`、`--draft→DRAFT`、`--ram→RAM_GB`、`--no-kvsave→KVSAVE=0`、`--metal→FLOYD_METAL=1`、`--cap/--ebits/--dbits→本地变量覆盖位置参数默认值(64/8/8)`。命令映射：chat→`setenv("CHAT","1")`；run→（PROMPT 已由 --prompt 设置，缺则 usage(2)）；tf→`setenv("TF","1")`；gen→什么都不设（默认 ref 对拍生成路径）。校验：缺 --model→usage(2)；未知 flag→`fprintf(stderr,"flag sconosciuto: %s\n",…)` + usage(2)；数值 flag 用 strtol/strtod 并检查范围（cap 1..4096、ebits/dbits ∈{4,8,16}、temp 0..2、top-p (0,1]、ngen/ctx ≥1）。数值形式的 argv[1]（legacy）与无参数路径**分毫不动**。

- [ ] **Step 2: 行为测试**

```bash
make
./floyd help | head -3                                   # 退出 0, 用法可读
./floyd tf --model fixture_tiny --ref ref_tiny.json --cap 8 --ebits 16 --dbits 16   # 期望 32/32
./floyd gen --model fixture_tiny --ref ref_tiny.json --cap 8 --ebits 16 --dbits 16  # 期望 20/20
SNAP=fixture_tiny TF=1 REF=ref_tiny.json ./floyd 8 16 16  # legacy 不变: 32/32
./floyd tf --ref x.json 2>&1 | head -2; echo "exit=$?"    # 缺 --model: 用法 + exit 2
./floyd chat --frobnicate 2>&1 | head -2; echo "exit=$?"  # 未知 flag: exit 2
```

- [ ] **Step 3: 转换器自动拷贝 tokenizer 文件**

convert_moonlight.py 的 convert() 里，config.json 拷贝旁：
```python
    for extra in ("tiktoken.model", "tokenizer_config.json", "generation_config.json"):
        src = os.path.join(a.indir, extra)
        if os.path.exists(src): shutil.copy(src, a.outdir)
```
验证：重跑 fixture 转换（fixture 无这些文件→不拷贝、不报错）。

- [ ] **Step 4: Commit**

```bash
git add floyd.c tools/convert_moonlight.py
git commit -m "chat 5: CLI subcommands + flags (env vars demoted to legacy/debug), converter copies tokenizer files"
```

---

### Task 6: E2E 冒烟 + README + 收尾

**Files:**
- Modify: `README.md`（chat 快速上手、CLI 参考表、env 变量移入"legacy/debug"一节）
- Create: `docs/chat-report.md`（简短：tokenizer 门槛 N/N、模板对拍、E2E 冒烟记录、多轮记忆验证、tok/s）

- [ ] **Step 1: E2E 多轮冒烟（真 i8）**

```bash
printf 'What is the capital of France? Answer in one word.\nWhat did I just ask you?\n:exit\n' \
  | ./floyd chat --model models/moonlight_i8 2>chat_smoke.err | tee chat_smoke.out
```
Expected: 第一答含 "Paris"；第二答引用第一问（记忆生效——例如提到 capital/France）；无 `<|im_end|>` 字面泄漏；stderr 有 tok/s 统计。把两轮输出原文记入 docs/chat-report.md。再验证 `:reset` 与 KV 持久化：先聊一轮退出，重开确认 warm 恢复（stderr 的 kv_disk_load 提示 / 或首轮 prefill token 数不含历史），`--no-kvsave` 下不写 `.coli_kv`。

- [ ] **Step 2: README 更新**

Quickstart 的主形态换成新 CLI（chat/tf/gen/run 各一条命令），新增 "CLI reference" 表（usage() 的内容），原 env 形式移到 "Legacy & debug environment variables" 小节并标注 tf/gen 的老命令仍有效。结果表加 tokenizer parity 行（N/N cases + template）。

- [ ] **Step 3: 全链路回归 + Commit**

```bash
make clean && make && make test-c && make test-tok
./floyd tf --model fixture_tiny --ref ref_tiny.json --cap 8 --ebits 16 --dbits 16   # 32/32
git add README.md docs/chat-report.md && git commit -m "chat 6: E2E chat smoke on real weights + README (CLI-first)"
```

---

## Self-Review 记录

- **Spec 覆盖**：组件 1→Task 1；组件 2→Task 3；组件 3→Task 4；组件 4→Task 5；两级验证→Task 3（门槛）+ Task 6（E2E）；错误处理（loader 报错/上下文满/输入超长——getline 无上限，64KB 截断改为：getline 本身动态分配，"超长"改为 maxctx 溢出拒绝，已在 Task 4 覆盖）；README→Task 6。缺口：spec 的"输入行 >64KB 截断"由 getline 动态分配天然支持，不再截断——按"上下文满拒绝"处理，Task 6 README 里注明。
- **占位符**：test_tok_moon.c 的 main 骨架标注了"完整代码由实现者写"——其行为（逐例比较、失败打印三元组、退出码）已完全指定，判定为可执行规格而非 TBD；其余无占位。
- **类型一致性**：`mtok_*` 签名在 Task 3 定义、Task 4/5 使用一致；`MoonEmit/emit_moon` 仅 Task 4 内部；usage() 的 flags 与 Task 5 解析表一致；`--top-p→NUCLEUS`、`--temp→TEMP` 与 floyd.c 实际 env 名核对过（558/560/2364 行）。

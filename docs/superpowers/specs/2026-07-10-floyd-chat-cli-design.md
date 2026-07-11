# floyd chat + CLI flags 设计

日期：2026-07-10
状态：已获用户批准（tokenizer 走原生 C tiktoken；用户追加要求：设计正规 CLI flags，不再滥用 env 变量）

## 目标

1. **`./floyd chat`**：交互式多轮对话——Moonlight 的 tiktoken tokenizer 移植到纯 C、moonshot 对话模板、流式输出、KV 复用与持久化；
2. **CLI flags**：子命令 + `--flag` 形式的正规命令行接口；env 变量保留为兼容层（老命令不破坏）与调试开关（不再是用户接口）。

非目标：OpenAI HTTP 网关、GLM/colibrì 兼容性变化（tok.h 不动）、Metal 性能调优、采样策略新增。

## 组件

### 1. Unicode 表扩展（`tools/gen_unicode.py` → `tok_unicode.h` 重新生成）

moonshot 分词正则需要的字符类超出现有 L/N/空白：新增 **Han**（`\p{Han}` script）、**Lu∪Lt**、**Ll**、**Lm**、**Lo**、**M**（Mn/Mc/Me）区间表与 `is_*` 查询函数。生成器扩展后重新生成并提交头文件（colibrì 惯例：生成物入库，生成器可复跑）。现有 uni_L/uni_N/空白表保持不变（tok.h 的 GLM 路径零影响）。Han 用 `unicodedata` 不够（script 属性需要 Scripts.txt）——生成器内嵌 CJK 统一表意文字的区间常量（U+4E00-9FFF、3400-4DBF、20000-2A6DF、2A700-2B73F、2B740-2B81F、2B820-2CEAF、2CEB0-2EBEF、30000-3134F、F900-FAFF、2F800-2FA1F），并在生成时用 Python `regex` 模块（如已安装）对拍抽样验证；未安装则跳过验证仅生成。

### 2. C tiktoken tokenizer（新头文件 `tok_moon.h`）

原生读模型目录，零离线转换步骤：

- **词表**：解析 `tiktoken.model`（每行 `base64(bytes) rank`，163,584 行）→ 复用 tok.h 的 hmap：字节序列→rank；`id→(bytes,len)` 数组用于 decode。
- **特殊 token**：解析 `tokenizer_config.json` 的 `added_tokens_decoder`（json.h 现成）→ id 163584..163841 的 content 映射（[BOS]/[EOS]/<|im_end|>/<|im_user|>/<|im_assistant|>/<|im_system|>/<|im_middle|>/[PAD]/[UNK] + reserved）。`moon_special_id(&T,"<|im_end|>")` 按内容查 id。
- **编码**：moonshot pre-tokenizer（8 个候选式按序尝试、每位置取第一个匹配，语义同 regex 交替；`&&[^\p{Han}]` 交集用"类内 && !is_Han"两次查表实现；`(?i:'s|'t|…)` 后缀、`\s+(?!\S)` 负向前瞻按现有 tok.h cl100k 匹配器的手写风格实现）→ 每个 chunk 走 **tiktoken rank 合并 BPE**（相邻对中拼接后 rank 最小者先合并，循环到无可合并；无需 merges 表）。**用户文本永不匹配特殊 token**（模板用显式 id 拼接，杜绝注入）。
- **解码**：id→字节拼接，特殊 token 输出字面内容；流式输出在 UTF-8 码点边界缓冲（沿用 serve 的 emit 模式）。

### 3. 对话模板与 `run_chat`（floyd.c）

模板忠实复刻 `tokenizer_config.json` 的 chat_template（token id 级拼接）：

- 首轮前置 `<|im_system|>` `system`(文本) `<|im_middle|>` {system 文本，默认 "You are a helpful assistant"，`--system` 覆盖} `<|im_end|>`；
- 每轮用户：`<|im_user|>` `user` `<|im_middle|>` {输入} `<|im_end|>` `<|im_assistant|>` `assistant` `<|im_middle|>`；
- 生成到 `<|im_end|>`（**停止 token 用 generation_config.json 的 eos_token_id=[163586]**，不是 config.json 的 163585；读取失败回退 `moon_special_id("<|im_end|>")`），生成的 ids + `<|im_end|>` 追加进 hist 供下一轮。

交互循环：readline（getline）、流式打印、`:reset` 清历史、`:exit`/EOF 退出；KV 历史复用与 `.coli_kv` 磁盘持久化直接沿用 run_serve 的现有机制。采样默认 temp 0.7 / top-p 0.90（colibrì 的 int4 现实调参）。

**对拍验证（方法论不变，两级门槛）**：
1. **tokenizer 门槛**：`tools/make_tok_cases.py`（AutoTokenizer）产出 `tok_cases.json`——覆盖英文缩写、中文连串、中英混排、1/2/3/4+ 位数字、各种空白/换行、emoji、代码片段、边界（空串/纯空格/超长词）共 ≥40 例的期望 ids，外加一个 3 轮对话的 `apply_chat_template` 期望序列；`tests/test_tok_moon.c` 全部逐 id 精确匹配 + decode 往返字节精确。**任何一例不匹配 = 门槛不过。**
2. **E2E**：`./floyd chat --model models/moonlight_i8` 问 capital of France 含 "Paris"、第二轮引用第一轮内容验证多轮记忆、`<|im_end|>` 处干净停止。

### 4. CLI（子命令 + flags）

```
floyd <command> [flags]

commands:
  chat     交互式对话（本设计交付）
  run      单条 prompt 生成（顺带获得：tokenizer 就位后即 run_text 换模板）
  tf       teacher-forcing 对拍       floyd tf  --model DIR --ref ref.json
  gen      greedy 生成对拍            floyd gen --model DIR --ref ref.json
  help     用法

全局 flags：--model DIR（必填，原 SNAP）· --cap N（默认 64）· --ebits {4,8,16}（默认 8）·
  --dbits {4,8,16}（默认 8）· --ram GB · --metal（原 FLOYD_METAL=1，CPU 构建给出 exit 2）
chat/run：--ngen N（chat 默认 512）· --ctx N（默认 4096）· --temp T（默认 0.7）·
  --top-p P（默认 0.90）· --system "..."（chat）· --prompt "..."（run 必填）·
  --draft N（MTP 草稿深度）· --no-kvsave
```

**实现方式：flags→env 薄适配层**（`main()` 顶部解析 argv，映射为对应 `setenv`，随后走既有代码路径）。理由：内部 20+ 处 `getenv` 零改动、零行为风险，flags 成为文档化接口，env 降级为兼容层。未知 flag / 缺 `--model` / 值越界 → 打印用法并 exit 2。

**兼容性**：无子命令时完全走老路径（`SNAP=… ./floyd cap ebits dbits`）——已提交的 parity-report、README 命令、CI 习惯全部继续有效。README 改为以新 CLI 为主、老 env 形式移入"兼容/调试"一节；调试开关（IDOT、DSA、MTP、TF/REF 之外的 PILOT/LOOKA/STATS/PIN 等）保持 env-only 并明确标注为开发者接口。

## 错误处理

- tiktoken.model / tokenizer_config.json 缺失或行格式非法：报错指明文件与行号，exit 1（chat/run 需要 tokenizer；tf/gen 不需要，不受影响）；
- base64 解码失败、rank 不连续、特殊 token 区间与词表 size 不一致：启动即报错；
- 输入行超长（>64 KB）截断并提示；上下文满：提示后拒绝该轮（不静默丢历史）。

## 里程碑

1. Unicode 表扩展 + 重新生成（含抽样对拍）；
2. `tools/make_tok_cases.py` + 期望用例；
3. `tok_moon.h`（loader + pre-tokenizer + BPE + decode）+ **tokenizer 精确门槛**；
4. 模板构造 + `run_chat` + 停止 token；
5. CLI flags 层 + help + README 更新；
6. E2E chat 冒烟（真 i8 容器）+ 收尾 review。

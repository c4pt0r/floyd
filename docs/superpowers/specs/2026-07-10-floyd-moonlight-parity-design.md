# floyd — Moonlight-16B-A3B 转换与对拍（pure C）设计

日期：2026-07-10
状态：已获用户批准

## 目标

在新 repo `floyd` 中，用 **Moonlight-16B-A3B**（`deepseek_v3` 架构）复刻 colibrì 的两条核心流程：

1. **转换**：HF BF16 safetensors → colibrì 式量化容器（int8 / packed int4，逐 shard、可断点续传）；
2. **对拍**：pure C 引擎 vs `transformers` oracle 的逐 token 验证（teacher-forcing + greedy）。

另含一个 **Metal 加速层**（用户 2026-07-10 追加）：对应 colibrì 的 CUDA tier，但针对 Apple Silicon 统一内存重新设计（见组件 5）。CPU pure C 路径始终是默认构建与对拍精确性的基准；Metal 是 `make METAL=1` 的可选加速件。

非目标（明确排除）：C 端 tokenizer / chat 交互、MTP 投机解码、DSA 稀疏注意力、CUDA、HTTP 服务。引擎运行时保持 pure C + 可选 Metal 后端（Python 仅用于离线转换与 oracle 生成，与 colibrì 边界一致）。

## 背景与依据

- colibrì（`~/colibri`）的 `c/glm.c` 已把架构维度参数化到运行时 `config.json`，且路由实现（sigmoid + `noaux_tc` + `norm_topk_prob` + `routed_scaling_factor`，假设 `n_group=1`）与 Moonlight 的 config **完全匹配**（Moonlight：`n_group=1, topk_group=1, scoring=sigmoid, topk_method=noaux_tc, routed_scaling=2.446`）。
- 已验证的架构差异收敛为一点：Moonlight `q_lora_rank: null` → 权重是 `self_attn.q_proj.weight`（直接投影），而 glm.c 无条件走 `q_a_proj`/`q_a_layernorm`/`q_b_proj` 的 q-LoRA 路径。
- Moonlight 无 DSA indexer 权重、`num_nextn_predict_layers=0`（无 MTP 头）；glm.c 对两者均按"权重存在才启用"检测，缺失时应走 dense/无投机路径（实现时验证）。
- 本机：Apple Silicon（arm64 NEON+SDOT 路径在 glm.c 中已存在）、32 核、512 GB RAM、~800 GB 空闲盘。Moonlight BF16 共 31.9 GB。

## 组件

仓库布局镜像 colibrì（扁平 C 核 + tools/）：

```
floyd/
├── floyd.c            引擎（自 colibri/c/glm.c 拷贝适配）
├── st.h json.h compat.h   原样拷自 colibrì
├── backend_metal.h/.m 可选 Metal 后端（C 可调用 API，Objective-C 实现）
├── kernels.metal      MSL 计算内核（q8/q4 反量化点积 matmul 等）
├── Makefile           macOS clang/OpenMP 与 Linux gcc 均可构建；METAL=1 启用 GPU
├── tools/
│   ├── convert_moonlight.py   BF16 → int8/int4 容器（自 convert_fp8_to_int4.py 改）
│   ├── make_oracle.py         transformers CPU oracle → ref.json（自 make_glm_oracle.py 改）
│   └── make_tiny_fixture.py   tiny-random DeepseekV3 fixture 生成（真架构、q_lora=null）
├── tests/             最小 C 单测（沿用 colibrì 的 st/json 测试）
├── docs/superpowers/specs/    本文档
└── README.md
```

### 1. `floyd.c`（引擎）

- 基线：`colibri/c/glm.c` 全量拷贝，改名，保留量化内核（int8/int4/int2）、NEON/SDOT 整数点积、流式专家 LRU、MLA 压缩 KV 与权重吸收、批量并集 MoE。
- 修改点：
  - **q 直投影分支**：`config.json` 的 `q_lora_rank` 为 `null`/缺失时（解析为 0），加载 `self_attn.q_proj.weight`（形状 `[H*qk_head, hidden]`），前向 `q = q_proj(x)`，跳过 `q_a` rmsnorm 与 `q_b`；`q_lora>0` 时维持原路径（保留 GLM-5.2 兼容）。
  - 确认 DSA / MTP 在权重缺失时干净禁用（不崩、不静默错误）。
  - 与 GLM-5.2 专属假设相关的硬编码（若发现）改为 config 驱动。

### 2. `tools/convert_moonlight.py`（转换器）

- 输入：HF repo（本地目录或 `moonshotai/Moonlight-16B-A3B-Instruct`，逐 shard 下载后即删，磁盘峰值受控）；
- 变更 vs colibrì 转换器：去掉 FP8 128×128 块反量化（源是 BF16，直接读为 f32 再 requantize）；张量名支持 `q_proj`；无 MTP / indexer 分支；
- 输出：colibrì 容器格式目录（`out-*.safetensors` + `config.json`），dense 与专家分文件；支持 `--bits 8|4`（分别产出，或先 int8 再打包 int4——遵循 colibrì "int4 打包与 int8 容器逐位一致"的路线）；断点续传。

### 3. `tools/make_oracle.py`（oracle）

- transformers CPU、float32（约 64 GB RAM，本机充足），加载方式优先 `trust_remote_code=True`（Moonlight 自带 modeling_deepseek.py，transformers 4.46 时代），失败则回退 transformers 原生 `DeepseekV3ForCausalLM`；
- 输出 `ref.json`：prompt token ids、teacher-forcing 每位置 argmax、greedy 续写 N token——格式与 colibrì 的 `ref_glm.json` 一致，引擎 `TF=1 REF=ref.json` 直接消费。

### 4. `tools/make_tiny_fixture.py`（fixture）

- 用 DeepseekV3Config 按 Moonlight 的形状比例缩小（hidden≈256、层数≈4、64 专家保留、q_lora=null、kv_lora/rope 维度按真实比例），随机权重、固定 seed，~数亿参数级；
- 同时生成该 fixture 的 oracle（同一脚本或调 make_oracle.py）。

### 5. Metal 后端（`backend_metal.m` + `kernels.metal`）

设计原则与 colibrì CUDA tier 相同（正确性优先的自写内核、可选构建、启用失败即报错不静默回退），但利用 Apple Silicon **统一内存**做了关键简化：

- **零拷贝**：所有量化权重（常驻 dense 与 LRU 缓存中的流式专家）都用 `newBufferWithBytesNoCopy` 把 GPU buffer 直接映射到引擎已有的页对齐内存上——**没有"上传"步骤**。colibrì CUDA 层"流式专家留在 CPU（PCIe 瓶颈）"的取舍在这里不存在：GPU 可以直接算刚从磁盘读进 LRU 的专家。
- **接口**：`backend_metal.h` 暴露 C API（`fm_init` / `fm_matmul_qt` / `fm_available`），Objective-C 实现编译进 `METAL=1` 构建；引擎侧调用点与 CPU `matmul_qt` 同签名，按 `FLOYD_METAL=1` 环境变量启用。
- **内核**：MSL compute shader 实现 q8 / packed q4 反量化点积 matmul（per-row scale，f32 累加）、可选 rmsnorm；先做批量（prefill / TF 验证）路径——GPU 收益最大处；单 token decode 若测不出收益则留在 CPU（按测量决定，colibrì 惯例）。
- **验证**：`make metal-test` 内核级单测（vs CPU 参考实现，逐元素容差断言）；端到端 A/B：同一容器同一 ref.json，CPU 与 METAL 各跑一遍 TF 报告。浮点归约顺序不同可能造成 argmax 平票翻转——**对拍精确性门槛以 CPU 路径为准**，Metal 报告如实附上。

## 数据流

```
HF BF16 权重 ──convert_moonlight.py──▶ int8/int4 容器目录
                                          │
tiny-random fixture ──(同一转换器)──▶ 容器 │
                                          ▼
make_oracle.py ──▶ ref.json ──▶ SNAP=<容器> TF=1 REF=ref.json ./floyd ──▶ 对拍报告
```

## 验收标准（两级）

1. **fixture 级（精确门槛，必须全过）**：tiny-random fixture 上 teacher-forcing **32/32** 位置一致、greedy **20/20** token 一致（int8 容器）；
2. **真模型级（诚实报数）**：Moonlight-16B-A3B-Instruct 转换成功（int8 与 int4 各一份或分次），真权重 TF 匹配率报告——int8 预期 ≈100%，int4 如实记录（README 记录数字，不设人为阈值）。

构建要求：`make` 在本机（macOS/clang）零警告出二进制；转换与 oracle 在专用 venv 中运行（torch CPU 版）。

## 错误处理

- 转换器：shard 校验（大小/张量清单），中断后重跑从缺失文件继续；维度与 config 不符即报错退出（沿用 colibrì PR #25 的防御式解析）。
- 引擎：config 维度越界检查沿用 glm.c 的 CKR；缺失张量报错并指出文件名；`q_lora` 与实际权重名不匹配时给出明确诊断。
- oracle：remote code 加载失败自动回退原生实现，并在 ref.json 里记录用了哪条路径。

## 风险与对策

| 风险 | 对策 |
|---|---|
| Moonlight 的 modeling_deepseek.py 与新 transformers 不兼容 | 回退 transformers 原生 DeepseekV3；必要时 pin transformers 版本到 venv |
| RoPE/数值细节差异导致对拍不过 | 在 fixture 上逐层 dump 中间激活对拍定位（colibrì 现成方法论） |
| int4 真权重匹配率不好看 | 如实报告；int8 容器作为精度上界参照 |
| 磁盘峰值 | 逐 shard 转换即删；总预算 ~45 GB（32 下载 + 容器），盘余 800 GB |

## 里程碑顺序

1. repo 骨架 + 头文件拷贝 + Makefile（macOS 构建通过）；
2. fixture 生成器 + 转换器 BF16 路径 + oracle 脚本（venv 就绪）；
3. floyd.c q_proj 分支 + fixture 对拍 32/32 & 20/20（**核心门槛**，CPU 路径）；
4. 真 Moonlight（下载已并行启动）→ 转换 → 真权重 TF 匹配率报告；
5. Metal 后端：metal-test 内核单测全过 + fixture/真权重 CPU vs METAL A/B 报告 + README。

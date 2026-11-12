# SFU-Sim

[English](README.md)

**NVIDIA MUFU 位级精确模型，已在 A100、Ada 与 Blackwell 显卡上完成全域验证。**

SFU-Sim 逐位复现 GPU 特殊函数指令 `ex2`、`lg2`、`rcp`、`rsqrt`、`sqrt`、`sin`、`cos`、`tanh` 的行为。仓库包含模型库、PTX 层源码和全部验证器；`make gpu-all` 在 NVIDIA 显卡上重跑全域逐位比对。

## 结果

| 检查 | 设备 | 覆盖范围 | 失配数 |
|---|---|---|---:|
| 模型 vs 硬件 | A100（sm_80） | 29 条 PTX 路径，全部输入域 | 0 |
| 模型 vs 硬件 | GeForce RTX 4060 Laptop（sm_89） | 29 条 PTX 路径，全部输入域 | 0 |
| 模型 vs 硬件 | RTX PRO 5000 Blackwell（sm_120） | 31 条 PTX 路径，全部输入域 | 0 |

模型描述裸 MUFU 指令，`ptx/` 补齐外围指令，两者合起来对应 29 条 PTX 路径（sm_120 另含 2 条 BF16，共 31 条）。

“全部输入域”指每个 32 位编码（f16 / BF16 为每个 16 位编码；f64 为高字的每个 32 位编码，低字固定为 0）。各组的比较总数和工具版本见 [`docs/results/SUMMARY.md`](docs/results/SUMMARY.md)。

支持的架构：`sm_80`（A100）、`sm_89`（Ada）、`sm_120`（Blackwell）。

支持的 MUFU 原语：`ex2`、`lg2`、`rcp`、`rsqrt`、`sqrt`、`sin`、`cos`、`tanh`、`rcp64h`、`rsq64h`，以及 `ex2` / `tanh` 的 f16 / BF16 变体。

支持的数据类型：FP32、FP64 高字、FP16、BF16。

### 零失配比较方式

每个输入位模式分别送入 GPU 与 CPU：GPU 由 `test/gpu/` 执行对应 PTX 指令，CPU 由 `ptx/` 外围处理与 `lib/` 模型库计算；`test/` 在主机侧对两路输出位模式逐位比较。

```text
输入位模式
  ├─→ GPU：test/gpu/ 执行 PTX 指令 ────→ 硬件输出位模式
  └─→ CPU：ptx/ 外围 + lib/ 模型库 ────→ 模型输出位模式
                      ↓
          test/ 在主机侧逐位比较
```

一路通过的判据为 `checked`（实际比较的输入数）等于该路输入域大小且 `mismatch` 为 0。比较逻辑与计数位于 `test/` 源码；分块调度与负对照记录于[验证文档](docs/verification.md)。

## 快速开始

以下为 Linux / WSL2 命令。Windows 原生使用 [docs/windows.md](docs/windows.md) 中的 `run_all.ps1`（不使用 `make`）。无 NVIDIA GPU 时执行第 1 步。

```bash
# 0. 校验模型库 sha256
cd lib && sha256sum -c libmufu_model.a.sha256 && cd ..     # Windows: certutil -hashfile lib\mufu_model.lib SHA256

# 1. 无需 GPU：自检 + 逐行重放 131430 条 A100 实测输出
make test
make eval && build/sfu_eval --list                            # 列出 29 条 PTX 路径名（sm_120 为 31 条）
build/sfu_eval sqrt.rn.f32 0x40000000   # -> sqrt.rn.f32 input=0x40000000 model=0x3fb504f3

# 2. GPU 全流程（32 核主机约 6 分钟）
make gpu-minimal-smoke ARCH=sm_89                           # ARCH 与显卡架构一致
make gpu-all TAG=mygpu                                      # 不写 ARCH 时自动探测
make help                                                   # 全部目标
```

`ARCH` 与显卡架构一致，否则验证器拒绝运行（外围 SASS 随架构变化）。架构由 `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` 给出，输出 `8.9` 对应 `sm_89`。常见值：`sm_80`（A100）、`sm_89`（RTX 40 系）、`sm_120`（Blackwell，另含 2 条 BF16）。单独运行某个 `gpu-*` 目标时不写 `ARCH` 使用默认值 `sm_80`；`make gpu-all` 自动探测。

模型使用全部 CPU 核，`SFU_THREADS=<n>` 限制线程数；完整运行时间随核数缩短。`TAG` 为结果目录名，日志和 `summary.txt` 写至 `build/results/<TAG>_<ARCH>/`。

环境：Linux x86_64、支持 C++17 的 g++ 和 GNU make（Windows 见 [docs/windows.md](docs/windows.md)）。GPU 检查需要 CUDA Toolkit 和 NVIDIA GPU（BF16 路径需要 sm_120 和 CUDA 12.8+）。

## 测试命令

| 命令 | 验证内容 | 需要 NVIDIA GPU |
|---|---|---|
| `make test` | 模型自检 + 逐行重放 131430 条 A100 实测输出 | 否 |
| `make eval` | 编译 `sfu_eval`，查询某条路径对某个输入的模型输出 | 否 |
| `make gpu-minimal-smoke ARCH=sm_XX` | 12 条路径（sm_120 为 14 条）的快速抽样检查 | 是 |
| `make gpu-all TAG=<名称>` | 全部 PTX 路径在全部输入域上与真卡逐位比较 | 是 |

各步的覆盖域、比较总数和验证方法见 [docs/verification.md](docs/verification.md) 与 [`docs/results/SUMMARY.md`](docs/results/SUMMARY.md)。

## 内容与目录

```text
lib/            模型库：libmufu_model.a = Linux / ELF，mufu_model.lib = Windows / COFF，
                旁边各有一份 .sha256
include/        mufu_model.h（库的 C 接口）、mufu_bits.hpp（位运算辅助）
ptx/            PTX 层源码：29 条 PTX 路径的外围指令模型
                （sm_120 为 31 条）
test/gpu/       硬件验证器源码：真卡执行 PTX，与模型逐位比较
test/offline/   A100 实测记录表（数据）+ 回放程序，不需要 GPU
test/unit/      模型自检
tools/          sfu_eval：查询某条路径对某个输入的模型输出
docs/           架构、验证方法、results/SUMMARY.md
.github/        CI：校验两个库哈希 + 跑不需要 GPU 的 make test
```

## 文档

以下文档为英文。

- [模型架构](docs/architecture.md)
- [验证](docs/verification.md)
- [在 Windows 上运行](docs/windows.md)

## 许可与引用

[MIT](LICENSE)。引用信息见 [`CITATION.cff`](CITATION.cff)。[arXiv:2611.13879](https://arxiv.org/abs/2611.13879) 给出逆向方法。

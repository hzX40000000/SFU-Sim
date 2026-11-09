# SFU-Sim

[中文说明](README.zh-CN.md)

**Bit-exact NVIDIA MUFU model, exhaustively verified on A100, Ada and Blackwell GPUs.**

SFU-Sim reproduces the bit-level behaviour of the GPU special-function instructions `ex2`, `lg2`, `rcp`, `rsqrt`, `sqrt`, `sin`, `cos` and `tanh`. The repository ships the model library, the PTX-layer sources and all verifiers; `make gpu-all` reruns the exhaustive bit-for-bit comparison on an NVIDIA GPU.

## Results

| Check | Device | Coverage | Mismatches |
|---|---|---|---:|
| Model vs hardware | A100 (sm_80) | 29 PTX paths, full input domain | 0 |
| Model vs hardware | GeForce RTX 4060 Laptop (sm_89) | 29 PTX paths, full input domain | 0 |
| Model vs hardware | RTX PRO 5000 Blackwell (sm_120) | 31 PTX paths, full input domain | 0 |

The model covers the bare MUFU instructions; `ptx/` adds the surrounding instructions, giving 29 PTX paths (31 on sm_120, which adds two BF16 paths).

"Full input domain" means every 32-bit input encoding (every 16-bit encoding for f16 / BF16; every high-word encoding with the low word fixed to zero for f64). Totals per group and the tool versions are in [`docs/results/SUMMARY.md`](docs/results/SUMMARY.md).

Supported architectures: sm_80 (A100), sm_89 (Ada), sm_120 (Blackwell).

Supported MUFU primitives: `ex2`, `lg2`, `rcp`, `rsqrt`, `sqrt`, `sin`, `cos`, `tanh`, `rcp64h`, `rsq64h`, and the f16 / BF16 variants of `ex2` and `tanh`.

Supported data types: FP32, FP64 high word, FP16, BF16.

### Comparison Method

Each input bit pattern is evaluated on both the GPU and the CPU: `test/gpu/` executes the corresponding PTX instruction on the GPU, while the `ptx/` wrapper and the `lib/` model compute the same input on the CPU. `test/` compares the two output bit patterns on the host.

```text
input bit pattern
  ├─→ GPU: test/gpu/ executes the PTX instruction ──→ hardware output bits
  └─→ CPU: ptx/ wrapper + lib/ model            ──→ model output bits
                      ↓
        test/ compares the two outputs on the host, bit for bit
```

A path passes when `checked` (inputs actually compared) equals the input-domain size and `mismatch` is 0. The comparison logic and the counters are in `test/`. Chunking and the negative control are described in the [verification guide](docs/verification.md).

## Quick start

Linux / WSL2 commands. Native Windows uses `run_all.ps1` from [docs/windows.md](docs/windows.md) (no `make`). Without an NVIDIA GPU, only step 1 applies.

```bash
# 0. Check the model library sha256
cd lib && sha256sum -c libmufu_model.a.sha256 && cd ..     # Windows: certutil -hashfile lib\mufu_model.lib SHA256

# 1. No GPU needed: self test + 131,430 recorded A100 outputs replayed through the model.
make test
make eval && build/sfu_eval --list                            # the 29 PTX path names (31 when built for sm_120)
build/sfu_eval sqrt.rn.f32 0x40000000   # -> sqrt.rn.f32 input=0x40000000 model=0x3fb504f3

# 2. GPU full run (~6 min on a 32-core host).
make gpu-minimal-smoke ARCH=sm_89                           # ARCH must match the GPU
make gpu-all TAG=mygpu                                      # ARCH is auto-detected when omitted
make help                                                   # all targets
```

`ARCH` must match the GPU or the verifier refuses to run (the surrounding SASS is architecture-specific). The compute capability is reported by `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`; `8.9` corresponds to `sm_89`. Common values: `sm_80` (A100), `sm_89` (RTX 40 series), `sm_120` (Blackwell, adds the two BF16 paths). Single `gpu-*` targets default to `sm_80` when `ARCH` is omitted; `make gpu-all` detects it automatically.

The model uses all CPU cores; `SFU_THREADS=<n>` limits the thread count and the full run scales with the core count. `TAG` names the result folder; logs and `summary.txt` are written to `build/results/<TAG>_<ARCH>/`.

Requirements: Linux x86_64, g++ with C++17 and GNU make (Windows: see [docs/windows.md](docs/windows.md)). GPU checks need the CUDA Toolkit and an NVIDIA GPU (BF16 paths need sm_120 and CUDA 12.8+).

## Test commands

| Command | What it checks | NVIDIA GPU needed |
|---|---|---|
| `make test` | Model self test + replay of 131,430 recorded A100 outputs | No |
| `make eval` | Build `sfu_eval` to query the model for one input | No |
| `make gpu-minimal-smoke ARCH=sm_XX` | Quick sampled check of 12 paths (14 on sm_120) | Yes |
| `make gpu-all TAG=<name>` | All PTX paths vs the GPU, bit for bit, over the full input domain | Yes |

Coverage, comparison totals and the method behind each step are in [docs/verification.md](docs/verification.md) and [`docs/results/SUMMARY.md`](docs/results/SUMMARY.md).

## What's in the repository

```text
lib/            model library: libmufu_model.a = Linux / ELF, mufu_model.lib = Windows / COFF,
                each with a .sha256 next to it
include/        mufu_model.h (C API of the library), mufu_bits.hpp (bit helpers)
ptx/            PTX-layer sources: the surrounding instructions for the 29 PTX paths
                (31 on sm_120)
test/gpu/       hardware verifier sources: run PTX on the GPU, compare with the model
test/offline/   recorded A100 outputs (data) + replay program, no GPU needed
test/unit/      model self test
tools/          sfu_eval: print the model output for one PTX path and input
docs/           architecture, verification, results/SUMMARY.md
.github/        CI: check both library hashes and run the CPU-only make test
```

## Documentation

- [Model architecture](docs/architecture.md)
- [Verification](docs/verification.md)
- [Running on Windows](docs/windows.md)

## License and citation

[MIT](LICENSE). Citation metadata is in [`CITATION.cff`](CITATION.cff).

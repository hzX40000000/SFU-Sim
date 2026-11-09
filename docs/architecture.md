# Model Architecture

SFU-Sim has two layers: the bit-level model of the bare MUFU instructions, provided by `lib/` and `include/`, and the PTX-layer sources in `ptx/`.

```text
PTX input bit pattern
  → ptx/*.hpp              pre-processing (SIN/COS pre-multiply, non-.ftz scaling, f16x2 unpack ...)
  → model library (lib/)   bare MUFU bit-level behaviour
  → ptx/*.hpp              post-processing (result scaling, f64 low-word clear, f16x2 pack ...)
  → PTX output bit pattern
```

- The library describes the output of the bare MUFU instructions; GPU verification covers the combined result of the PTX wrapper and the bare MUFU.
- The verification method is in [verification.md](verification.md).

## 1. Model Library (`lib/`, `include/`)

`include/mufu_model.h` declares the C API (`sfu_mufu_*`). All functions are pure and thread-safe. Linux links `lib/libmufu_model.a`; Windows links `lib/mufu_model.lib`.

| Function | Input / output | Description |
|---|---|---|
| `sfu_mufu_ex2/lg2/rcp/rsq/sqrt/tanh` | 32 / 32 bits | f32 bit pattern in, bit pattern out |
| `sfu_mufu_sin/cos` | 32 / 32 bits | Argument in turns; the PTX layer multiplies by 1/2π first |
| `sfu_mufu_rcp64h/rsq64h` | 32 / 32 bits | Input and output are the high 32 bits of an f64 |
| `sfu_mufu_ex2_f16/tanh_f16` | 16 / 16 bits | f16 |
| `sfu_mufu_ex2_bf16/tanh_bf16` | 16 / 16 bits | BF16 (sm_120) |
| `sfu_model_version` | — | Library version string |

`include/mufu_bits.hpp` provides the bit operations used by `ptx/` (`mul_rn`, `add_rn`, ...).

## 2. `ptx/`: PTX Wrapper

| File | PTX paths | Surrounding instructions (A100 `sm_80` SASS observations) |
|---|---|---|
| `minimal.hpp` | `ex2/lg2/rcp/rsqrt/sqrt.approx.ftz.f32`, `tanh.approx.f32` | None: `LDG → MUFU.xxx → STG` |
| | `sin/cos.approx.ftz.f32` | Leading `FMUL.RZ x, 0x3e22f983` (multiply by 1/2π) |
| | `rcp/rsqrt.approx.ftz.f64` | Only the input high word reaches `MUFU.RCP64H/RSQ64H`; the output low word is cleared |
| | `ex2/tanh.approx.f16` | Direct `MUFU.EX2.F16 / TANH.F16` |
| `bf16.hpp` (sm_120 only) | `ex2.approx.ftz.bf16`, `tanh.approx.bf16` | Direct `MUFU.EX2.BF16 / TANH.BF16` |
| `extended_f32.hpp` | `ex2.approx.f32` | For `x < -126`: input ×0.5, then square the MUFU result |
| | `lg2.approx.f32` | For \|x\| below the smallest normal: input ×2^24, result −24 |
| | `rcp.approx.f32` | Scale ∈ {2^24, 2^-2, 1} selected by whether \|x\| is subnormal or large; applied once to the input and once to the output |
| | `rsqrt.approx.f32` | Subnormal: input ×2^24, result ×2^12 |
| | `sqrt.approx.f32` | Subnormal: input ×2^24, result ×2^-12 |
| | `sin/cos.approx.f32` | The same `FMUL.RZ` pre-multiply as minimal |
| `f16x2.hpp` | `ex2/tanh.approx.f16x2` | One `MUFU.*.F16` per half, then `PRMT` to pack |
| `round_f32.hpp` | `rcp/sqrt.rn/rz/rm/rp.f32` (non-`.ftz`) | SASS takes an initial estimate from `MUFU.RCP` (`MUFU.RSQ` for sqrt), refines it with FFMA residual iterations and directional corrections, with a slow path; the model computes the final result from the exact quotient / remainder or integer square root and the rounding semantics, independently of the library |

The table lists the surrounding instructions observed on A100 (sm_80). The same model passed full-domain verification on sm_80, sm_89 and sm_120 (see [verification.md](verification.md)).

## 3. Path Mapping

| Bare MUFU (SASS) | Library function | PTX that triggers it (minimal) |
|---|---|---|
| `MUFU.EX2` | `sfu_mufu_ex2` | `ex2.approx.ftz.f32` |
| `MUFU.LG2` | `sfu_mufu_lg2` | `lg2.approx.ftz.f32` |
| `MUFU.RCP` | `sfu_mufu_rcp` | `rcp.approx.ftz.f32` |
| `MUFU.RSQ` | `sfu_mufu_rsq` | `rsqrt.approx.ftz.f32` |
| `MUFU.SQRT` | `sfu_mufu_sqrt` | `sqrt.approx.ftz.f32` |
| `MUFU.TANH` | `sfu_mufu_tanh` | `tanh.approx.f32` |
| `MUFU.SIN` | `sfu_mufu_sin` | `sin.approx.ftz.f32` (via `FMUL.RZ`) |
| `MUFU.COS` | `sfu_mufu_cos` | `cos.approx.ftz.f32` (via `FMUL.RZ`) |
| `MUFU.RCP64H` | `sfu_mufu_rcp64h` | `rcp.approx.ftz.f64` |
| `MUFU.RSQ64H` | `sfu_mufu_rsq64h` | `rsqrt.approx.ftz.f64` |
| `MUFU.EX2.F16` | `sfu_mufu_ex2_f16` | `ex2.approx.f16` |
| `MUFU.TANH.F16` | `sfu_mufu_tanh_f16` | `tanh.approx.f16` |
| `MUFU.EX2.BF16` | `sfu_mufu_ex2_bf16` | `ex2.approx.ftz.bf16` (sm_120) |
| `MUFU.TANH.BF16` | `sfu_mufu_tanh_bf16` | `tanh.approx.bf16` (sm_120) |

## 4. SASS Disassembly

```bash
make build/gpu_verify_sm_80                         # compile only
cuobjdump --dump-sass build/gpu_verify_sm_80 | grep MUFU
```

The verifier's GPU kernel (`hw_kernel`) contains only the PTX instruction under test; the model is evaluated on the CPU. The surrounding instructions are taken from the disassembly of that kernel.

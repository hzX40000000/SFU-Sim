# Verification Summary

Every number below is the sum of `checked` over the paths of one group; `mismatch` is 0 in all cases. Each row is reproduced by the command in the last column (see [verification.md](../verification.md)).

## Hardware: Model vs Real GPU

For every input the GPU executes the PTX instruction under test and the model (`lib/libmufu_model.a` on Linux, `lib/mufu_model.lib` on Windows) computes the same input on the CPU; the two results are compared bit for bit.

| Device | Arch | Driver | CUDA toolkit |
|---|---|---|---|
| NVIDIA A100-SXM4-40GB | sm_80 | 595.71.05 | 13.2 |
| GeForce RTX 4060 Laptop GPU | sm_89 | 560.94 | 12.2 (Windows, MSVC 19.44) |
| NVIDIA RTX PRO 5000 Blackwell | sm_120 | 580.105.08 | 12.8 |

| Group | PTX paths | Inputs compared per device | Mismatch | Command |
|---|---:|---:|---:|---|
| minimal | 12 (sm_120: 14) | 42,949,804,032 (sm_120: 42,949,935,104) | 0 | `make gpu-minimal` |
| extended (non-`.ftz` approx f32) | 7 | 30,064,771,072 | 0 | `make gpu-extended` |
| f16x2 | 2 | 8,589,934,592 | 0 | `make gpu-f16x2` |
| round (`rcp`/`sqrt` `.rn .rz .rm .rp`) | 8 | 34,359,738,368 | 0 | `make gpu-round` |
| offline (recorded A100 outputs) | 12 | 131,430 | 0 | `make offline` |
| negative control (comparator detects an injected error) | 1 (sm_120: 2) | 65,536 per path, 1 detection | n/a | `make gpu-negative` |
| **total** | **29** (sm_120: **31**) | **115,964,248,064** (sm_120: 115,964,379,136) | **0** | `make gpu-all` |

The total is the sum of the four GPU groups; the offline replay and the negative control are listed separately. All three devices reported zero mismatches on every path; each path was compared over its entire input domain (every 32-bit encoding, every 16-bit encoding for f16 / BF16, and every high-word encoding with the low word fixed to zero for f64).

The `make` commands in the last column are the Linux / WSL2 flow. Single-group targets default to `ARCH=sm_80`; other cards require `ARCH=sm_XX` (`make gpu-all` detects it automatically). On native Windows the equivalent is `test\gpu\run_all.ps1 -Tag <name> -Arch <sm_XX>` (see [windows.md](../windows.md)); `-Arch` defaults to `sm_89` there and is not auto-detected.

The RTX 4060 run used `lib/mufu_model.lib` (sha256 in `lib/mufu_model.lib.sha256`).

## Example Output

`make gpu-all TAG=a100 ARCH=sm_80` prints, on the A100:

```text
selftest        PASS MODEL SELF TEST PASS
offline         PASS PASS
negative        PASS NEGATIVE CONTROL PASS
minimal         PASS PASS
extended_smoke  PASS EXTENDED F32 SMOKE PASS
extended        PASS EXTENDED F32 PASS
f16x2_smoke     PASS F16X2 SMOKE PASS
f16x2           PASS F16X2 PASS
round_smoke     PASS ROUND F32 SMOKE PASS
round           PASS ROUND F32 PASS
```

and `make gpu-minimal` ends with, per path, `OK <path> checked=4294967296 mismatch=0` followed by `PASS`.

# Running on Windows

The repository provides two model libraries: `lib/libmufu_model.a` (Linux x86_64, ELF) and `lib/mufu_model.lib` (Windows x64, COFF).

| | Option A: WSL2 | Option B: Native Windows |
|---|---|---|
| Model library | `lib/libmufu_model.a` | `lib/mufu_model.lib` |
| Command | Same as Linux | `test\gpu\run_all.ps1` |

Option C covers the CPU-only checks for hosts without an NVIDIA GPU or without the CUDA Toolkit.

The sha256 of both libraries is in `lib/`:

```bat
certutil -hashfile lib\mufu_model.lib SHA256
type lib\mufu_model.lib.sha256
```
```bash
cd lib && sha256sum -c libmufu_model.a.sha256     # WSL2 / Linux
```

## Option A: WSL2

1. In an administrator PowerShell, run `wsl --install -d Ubuntu-24.04`; create a user after the restart.
2. WSL2 uses the Windows NVIDIA driver, so no driver is installed inside WSL; `nvidia-smi` in Ubuntu lists the GPU.
3. Install `build-essential` and the CUDA Toolkit (Linux / x86_64 / WSL-Ubuntu) in Ubuntu.
4. Place the repository on the WSL filesystem, for example `~/SFU-Sim`; compilation and file I/O under `/mnt/c/` are slower.
5. Run:
   ```bash
   make test
   make gpu-all TAG=rtx4060 ARCH=sm_89
   ```

WSL assigns half of the host memory by default. The model is evaluated on the CPU; `memory=` in `%UserProfile%\.wslconfig` raises that limit, and `SFU_THREADS=<n>` limits the thread count.

## Option B: Native Windows

Requirements: the C++ workload of Visual Studio (or Build Tools), a compatible CUDA Toolkit and an NVIDIA driver.

1. In the **x64 Native Tools Command Prompt for VS**, confirm that the following commands are available:
   ```bat
   cl
   lib
   nvcc --version
   ```
2. From the repository directory, run:
   ```bat
   powershell -ExecutionPolicy Bypass -File test\gpu\run_all.ps1 -Tag rtx4060 -Arch sm_89
   ```
   The script uses `lib\mufu_model.lib`, builds the self test, the offline comparison and four GPU verifiers, runs the same 10 steps in the Linux order, writes logs to `build\results\<Tag>_<Arch>\` and prints `summary.txt`.
3. With all steps passing, `summary.txt` contains 10 lines and the comparison total matches [`results/SUMMARY.md`](results/SUMMARY.md).

`-Steps "minimal f16x2"` runs a subset of the steps; `set SFU_THREADS=8` limits the CPU thread count.

> Verifiers are bound to the architecture used at compile time: a binary built with `-Arch sm_89` is rejected on sm_80 because the surrounding SASS is architecture-specific; an architecture change requires a rebuild.
>
> The Windows script does not auto-detect `-Arch` and defaults to `sm_89`; Linux `make gpu-all` detects it automatically. The compute capability is reported by `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`; `8.9` corresponds to `-Arch sm_89`.

## Option C: CPU-only Checks

Option C requires only the **C++ workload** of Visual Studio (`cl`); no CUDA Toolkit and no GPU. In the **x64 Native Tools Command Prompt for VS**:

```bat
certutil -hashfile lib\mufu_model.lib SHA256
type lib\mufu_model.lib.sha256
if not exist build mkdir build
cl /nologo /O2 /std:c++17 /MD /EHsc /fp:strict /W3 /I. /Iinclude test\unit\model_selftest.cpp /Fobuild\ /Febuild\model_selftest.exe lib\mufu_model.lib
build\model_selftest.exe
cl /nologo /O2 /std:c++17 /MD /EHsc /fp:strict /W3 /I. /Iinclude test\offline\offline_verify.cpp /Fobuild\ /Febuild\offline_verify.exe lib\mufu_model.lib
build\offline_verify.exe test\offline\a100_sm80.tsv
cl /nologo /O2 /std:c++17 /MD /EHsc /fp:strict /W3 /I. /Iinclude tools\sfu_eval.cpp /Fobuild\ /Febuild\sfu_eval.exe lib\mufu_model.lib
build\sfu_eval.exe --list
```

The three checks print `MODEL SELF TEST PASS`, `PASS` and the PTX path names, corresponding to Linux `make test` and `make eval`.

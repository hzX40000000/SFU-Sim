# Windows-native counterpart of run_all.sh: builds everything with MSVC (cl) + nvcc and runs every
# hardware verification on one GPU. Logs go to build\results\<Tag>_<Arch>\.
#
# Run from an "x64 Native Tools Command Prompt for VS" (so that cl, lib and nvcc are on PATH):
#   powershell -ExecutionPolicy Bypass -File test\gpu\run_all.ps1 -Tag rtx4060 -Arch sm_89
#
# The published lib\mufu_model.lib is used as-is.
# Environment: SFU_THREADS limits the CPU threads used for the model; CUDA_VISIBLE_DEVICES selects the GPU.
param(
    [Parameter(Mandatory = $true)][string]$Tag,
    [string]$Arch = "sm_89",
    [string]$Steps = ""
)
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

$sm   = $Arch -replace '^sm_', ''
$out  = "build\results\${Tag}_${Arch}"
New-Item -ItemType Directory -Force -Path "build", "lib", $out | Out-Null

function Need($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: '$name' not found. Start this script from the x64 Native Tools Command Prompt for VS."
        exit 2
    }
}
Need cl; Need lib; Need nvcc

$cl = @("/nologo", "/O2", "/std:c++17", "/MD", "/EHsc", "/fp:strict", "/W3")

function Run-Checked($label, [scriptblock]$cmd) {
    Write-Host "--- $label"
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: $label failed (exit $LASTEXITCODE)"; exit 3 }
}

# ---- model library
if (-not (Test-Path "lib\mufu_model.lib")) {
    Write-Host "ERROR: lib\mufu_model.lib is missing; re-clone the repository."; exit 2
}

# ---- environment record (GPU, driver, toolchain versions only)

$envLines = @("arch: $Arch", "--- gpu")
$envLines += (& nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv 2>&1 | ForEach-Object { "$_" })
$envLines += "--- nvcc"
$envLines += (& nvcc --version 2>&1 | ForEach-Object { "$_" })
$envLines += "--- cl"
$envLines += (& cl 2>&1 | Select-Object -First 1 | ForEach-Object { "$_" })
$envLines | Set-Content -Encoding ascii "$out\env.txt"

# ---- build the checkers
$nv = @("-O3", "-std=c++17", "-arch=$Arch", "-DSFU_TARGET_SM=$sm", "-I.", "-Iinclude")
$bf = @(); if ($Arch -eq "sm_120") { $bf = @("-DSFU_ENABLE_BF16=1") }
function Build-Cpu($src, $exe) {
    Run-Checked "cl $src" { cl @cl /I. /Iinclude $src "/Fobuild\" "/Fe$exe" lib\mufu_model.lib }
}
function Build-Gpu($src, $exe, $extra) {
    Run-Checked "nvcc $src" { nvcc @nv @extra $src -o $exe lib\mufu_model.lib }
}
Build-Cpu "test\unit\model_selftest.cpp"  "build\model_selftest.exe"
Build-Cpu "test\offline\offline_verify.cpp" "build\offline_verify.exe"
Build-Gpu "test\gpu\gpu_verify.cu"                "build\gpu_verify_$Arch.exe" $bf
Build-Gpu "test\gpu\gpu_verify_extended_f32.cu"   "build\gpu_verify_extended_f32_$Arch.exe" @()
Build-Gpu "test\gpu\gpu_verify_f16x2.cu"          "build\gpu_verify_f16x2_$Arch.exe" @()
Build-Gpu "test\gpu\gpu_verify_round_f32.cu"      "build\gpu_verify_round_f32_$Arch.exe" @()

# name | executable | arguments | PASS line
$min = "build\gpu_verify_$Arch.exe"
$ext = "build\gpu_verify_extended_f32_$Arch.exe"
$f16 = "build\gpu_verify_f16x2_$Arch.exe"
$rnd = "build\gpu_verify_round_f32_$Arch.exe"
$table = @(
    @("selftest",       "build\model_selftest.exe",  @(),                                   "MODEL SELF TEST PASS"),
    @("offline",        "build\offline_verify.exe",  @("test\offline\a100_sm80.tsv"),       "PASS"),
    @("negative",       $min, @("negative"), "NEGATIVE CONTROL PASS"),
    @("minimal",        $min, @("full"),     "PASS"),
    @("extended_smoke", $ext, @("smoke"),    "EXTENDED F32 SMOKE PASS"),
    @("extended",       $ext, @("full"),     "EXTENDED F32 PASS"),
    @("f16x2_smoke",    $f16, @("smoke"),    "F16X2 SMOKE PASS"),
    @("f16x2",          $f16, @("full"),     "F16X2 PASS"),
    @("round_smoke",    $rnd, @("smoke"),    "ROUND F32 SMOKE PASS"),
    @("round",          $rnd, @("full"),     "ROUND F32 PASS")
)

$summary = @()
$idx = 0
foreach ($row in $table) {
    $idx++
    $name = $row[0]
    if ($Steps -and (" $Steps " -notlike "* $name *")) { continue }
    $log = "{0}\{1:00}_{2}.log" -f $out, $idx, $name
    Write-Host "=== ${name}: $($row[1]) $($row[2] -join ' ')"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    Set-Content -Encoding ascii $log ""
    $exeArgs = @($row[2])
    & $row[1] @exeArgs 2>&1 | ForEach-Object { $s = "$_"; Write-Host $s; Add-Content -Encoding ascii $log $s }
    Add-Content -Encoding ascii $log ("# elapsed_s {0}" -f [int]$sw.Elapsed.TotalSeconds)
    $last = (Get-Content $log | Where-Object { $_ -and -not $_.StartsWith("#") } | Select-Object -Last 1)
    $status = if ($last -eq $row[3]) { "PASS" } else { "FAIL" }
    $summary += ("{0,-15} {1,-4} {2}" -f $name, $status, $last)
}
$summary | Set-Content -Encoding ascii "$out\summary.txt"
Write-Host "--- $out\summary.txt"
$summary | ForEach-Object { Write-Host $_ }
if ($summary -match ' FAIL ') { exit 1 }

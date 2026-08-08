# Build JamePeng llama-cpp-python with HIP against an existing ComfyUI-rocm python_env.
# No venv Activate.ps1 - uses python.exe + Lib\site-packages next to it.
#
# Usage (ComfyUI CLOSED):
#   powershell -ExecutionPolicy Bypass -File .\scripts\build-jamepeng-hip.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\build-jamepeng-hip.ps1 -Gfx gfx1201

param(
    [string]$PythonExe = "D:\ComfyUI-rocm\python_env\python.exe",
    [string]$Gfx = "",
    [string]$SrcDir = "C:\temp\lpy"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path $PythonExe)) {
    throw "Python not found: $PythonExe"
}

$Py = (Resolve-Path $PythonExe).Path
Write-Host "[build] python: $Py"

# ComfyUI-rocm python_env: <python_env>\Lib\site-packages
$PyRoot = Split-Path -Parent $Py
$SitePackages = Join-Path $PyRoot "Lib\site-packages"
if (-not (Test-Path $SitePackages)) {
    throw "Expected site-packages at $SitePackages (next to python.exe)."
}
Write-Host "[build] site-packages: $SitePackages"

$RocmDevel = Join-Path $SitePackages "_rocm_sdk_devel"
$RocmCore  = Join-Path $SitePackages "_rocm_sdk_core"
$RocmLibs  = Join-Path $SitePackages "_rocm_sdk_libraries"

foreach ($p in @($RocmDevel, $RocmCore, $RocmLibs)) {
    if (-not (Test-Path $p)) {
        Write-Warning "Missing ROCm dir: $p"
    } else {
        Write-Host "[build] found $($p.Substring($SitePackages.Length + 1))"
    }
}
if (-not (Test-Path $RocmDevel)) {
    throw "Need _rocm_sdk_devel under site-packages (ComfyUI ROCm install incomplete for compiling)."
}

function Get-DetectedGfx {
    $fromPip = & $Py -c "import re,importlib.metadata as md; found=[]; [found.append(m.group(1)) for d in md.distributions() for m in [re.search(r'(?:device-|libraries-)(gfx[0-9a-f]+)', ((d.metadata.get('Name') or '').lower().replace('_','-')))] if m]; print('\n'.join(sorted(set(found))))"
    if ($fromPip) {
        $cands = @($fromPip -split "`r?`n" | Where-Object { $_ -match '^gfx' })
        if ($cands.Count -ge 1) {
            Write-Host "[build] device packages: $($cands -join ', ')"
            $full = @($cands | Where-Object { $_ -match '^gfx\d{4}' })
            if ($full.Count -ge 1) { return ($full | Sort-Object Length -Descending | Select-Object -First 1) }
            return ($cands | Sort-Object Length -Descending | Select-Object -First 1)
        }
    }

    $fromTorch = & $Py -c "import torch; p=torch.cuda.get_device_properties(0); print(getattr(p,'gcnArchName',None) or p.name); print(getattr(torch.version,'hip',None))" 2>$null
    if ($fromTorch -match '(gfx[0-9a-f]+)') { return $Matches[1] }

    foreach ($binDir in @(
        (Join-Path $RocmLibs "bin"),
        (Join-Path $RocmDevel "bin"),
        (Join-Path $RocmCore "bin")
    )) {
        foreach ($exe in @("hipInfo.exe", "hipinfo.exe", "rocminfo.exe")) {
            $p = Join-Path $binDir $exe
            if (Test-Path $p) {
                $out = & $p 2>&1 | Out-String
                if ($out -match '(gfx[0-9a-f]+)') { return $Matches[1] }
            }
        }
    }

    if (Test-Path $RocmLibs) {
        $hit = Get-ChildItem $RocmLibs -Recurse -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^gfx\d{4}$' } |
            Select-Object -First 1 -ExpandProperty Name
        if ($hit) { return $hit }
    }

    return $null
}

if (-not $Gfx) { $Gfx = Get-DetectedGfx }
if (-not $Gfx) {
    throw "Could not auto-detect GPU_TARGETS. Pass -Gfx gfx1201 (or your ISA)."
}
Write-Host "[build] GPU_TARGETS=$Gfx"

$env:HIP_PATH = $RocmDevel
$env:ROCM_PATH = $RocmDevel
$env:ROCM_HOME = $RocmDevel
$bitcode = Join-Path $RocmCore "lib\llvm\amdgcn\bitcode"
if (-not (Test-Path $bitcode)) {
    $bitcode = Join-Path $RocmDevel "lib\llvm\amdgcn\bitcode"
}
$env:HIP_DEVICE_LIB_PATH = $bitcode
$env:DEVICE_LIB_PATH = $bitcode

# Clang (HIP) still needs MSVC/Windows SDK tools for rc.exe / link on Windows.
function Import-VsDevEnv {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { return $false }
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) {
        $vsPath = & $vswhere -latest -products * -property installationPath
    }
    if (-not $vsPath) { return $false }
    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) { return $false }
    Write-Host "[build] importing VS env: $vcvars"
    $envDump = & cmd.exe /c "`"$vcvars`" >nul 2>&1 && set"
    foreach ($line in $envDump) {
        if ($line -match '^([^=]+)=(.*)$') {
            Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
        }
    }
    return $true
}

function Find-RcCompiler {
    $fromPath = Get-Command rc.exe -ErrorAction SilentlyContinue
    if ($fromPath) { return $fromPath.Source }
    $kitRoot = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"
    if (-not (Test-Path $kitRoot)) { return $null }
    $hit = Get-ChildItem $kitRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName "x64\rc.exe" } |
        Where-Object { Test-Path $_ } |
        Select-Object -First 1
    return $hit
}

if (-not (Import-VsDevEnv)) {
    Write-Warning "VS Build Tools / vcvars64.bat not found. Install 'Desktop development with C++' (incl. Windows SDK)."
}

$RcCompiler = Find-RcCompiler
if (-not $RcCompiler) {
    throw "No CMAKE_RC_COMPILER (rc.exe). Install Visual Studio Build Tools + Windows 10/11 SDK, then re-run."
}
$env:RC = $RcCompiler
Write-Host "[build] RC=$RcCompiler"

$clangBin = Join-Path $RocmDevel "lib\llvm\bin"
$env:HIP_CLANG_PATH = $clangBin
$env:HIP_PLATFORM = "amd"
$env:CMAKE_GENERATOR = "Ninja"
$env:CC  = Join-Path $clangBin "clang.exe"
$env:CXX = Join-Path $clangBin "clang++.exe"
$env:FORCE_CMAKE = "1"

$pathBits = @(
    (Join-Path $RocmLibs "bin"),
    (Join-Path $RocmDevel "bin"),
    $clangBin,
    (Join-Path $RocmCore "bin"),
    (Split-Path -Parent $RcCompiler)
) | Where-Object { Test-Path $_ }
$env:PATH = ($pathBits + $env:PATH) -join ";"

$R = ($RocmDevel -replace '\\', '/')
$RcCMake = ($RcCompiler -replace '\\', '/')
$hipLib = "$R/lib/amdhip64.lib"
if (-not (Test-Path ($hipLib -replace '/', '\'))) {
    $alt = Join-Path $RocmLibs "lib\amdhip64.lib"
    if (Test-Path $alt) { $hipLib = ($alt -replace '\\', '/') }
}

$env:CMAKE_ARGS = "-DGGML_HIP=ON -DGGML_HIPBLAS=on -DGPU_TARGETS=$Gfx -DCMAKE_HIP_ARCHITECTURES=$Gfx -DCMAKE_C_COMPILER=`"$R/lib/llvm/bin/clang.exe`" -DCMAKE_CXX_COMPILER=`"$R/lib/llvm/bin/clang++.exe`" -DCMAKE_RC_COMPILER=`"$RcCMake`" -DHIP_LIBRARIES=`"$hipLib`" -DCMAKE_PREFIX_PATH=`"$R`""
Write-Host "[build] CMAKE_ARGS=$env:CMAKE_ARGS"

Write-Host "[build] ensuring cmake>=3.21 + ninja in python_env ..."
& $Py -m pip install "cmake>=3.21" ninja
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install cmake/ninja into python_env."
}
# Prefer pip-provided cmake/ninja on PATH for scikit-build-core
$scriptsDir = Join-Path $PyRoot "Scripts"
if (Test-Path $scriptsDir) {
    $env:PATH = "$scriptsDir;$env:PATH"
}
$cmakeExe = Join-Path $scriptsDir "cmake.exe"
if (-not (Test-Path $cmakeExe)) {
    $cmakeExe = (& $Py -c "import cmake; import os; print(os.path.join(os.path.dirname(cmake.__file__), 'data', 'bin', 'cmake.exe'))" 2>$null)
}
if ($cmakeExe -and (Test-Path $cmakeExe)) {
    $env:PATH = "$(Split-Path -Parent $cmakeExe);$env:PATH"
    Write-Host "[build] cmake: $(& $cmakeExe --version | Select-Object -First 1)"
} else {
    Write-Warning "cmake.exe not found after pip install - scikit-build may still fail."
}

Write-Host "[build] enabling git longpaths + cloning to $SrcDir ..."
git config --global core.longpaths true
try {
    New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
        -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force -ErrorAction Stop | Out-Null
    Write-Host "[build] Windows LongPathsEnabled=1"
} catch {
    Write-Warning "Could not set LongPathsEnabled (admin). Relying on short SrcDir + git longpaths."
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $SrcDir) | Out-Null
if (Test-Path $SrcDir) {
    Write-Host "[build] removing old $SrcDir ..."
    Remove-Item -Recurse -Force $SrcDir
}

git clone --recursive --depth 1 https://github.com/JamePeng/llama-cpp-python.git $SrcDir
if ($LASTEXITCODE -ne 0) {
    if (Test-Path $SrcDir) { Remove-Item -Recurse -Force $SrcDir -ErrorAction Continue }
    throw "git clone failed (exit $LASTEXITCODE). Enable LongPathsEnabled + git core.longpaths true, reboot, re-run."
}

$verifyPy = Join-Path $env:TEMP "llm_prompter_verify_llama.py"
@'
from llama_cpp.llama_chat_format import Qwen35ChatHandler
import llama_cpp, os
lib = os.path.join(os.path.dirname(llama_cpp.__file__), "lib")
print("version", llama_cpp.__version__)
print("Qwen35ChatHandler OK")
print("libs", [f for f in os.listdir(lib) if "hip" in f.lower() or "vulkan" in f.lower()][:20])
'@ | Set-Content -Path $verifyPy -Encoding ASCII

try {
    Write-Host "[build] uninstalling existing llama-cpp-python ..."
    & $Py -m pip uninstall -y llama-cpp-python

    Write-Host "[build] compiling from $SrcDir (long) ..."
    & $Py -m pip install "$SrcDir" --force-reinstall --no-cache-dir --no-build-isolation
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed (exit $LASTEXITCODE). llama_cpp is NOT installed - restore the hip-radeon wheel if needed."
    }

    Write-Host "[build] verify handlers + HIP backend ..."
    & $Py $verifyPy
    if ($LASTEXITCODE -ne 0) {
        throw "Verify failed - build may have produced a broken install."
    }

    Write-Host "[build] done. Restart ComfyUI; set chat_handler to Qwen3.5 / Qwen3.5-Thinking."
}
finally {
    if (Test-Path $SrcDir) {
        Write-Host "[build] cleaning up $SrcDir ..."
        Remove-Item -Recurse -Force $SrcDir -ErrorAction Continue
    }
    if (Test-Path $verifyPy) {
        Remove-Item -Force $verifyPy -ErrorAction Continue
    }
}

# -*- coding: utf-8 -*-
<#
  AI 翻唱工坊 v1.0 — 环境安装脚本（Windows / PowerShell）

  它只做四件事，全部在本目录内，不乱写系统：
    1. 建一个 venv（.venv），装 torch cu121 + 依赖
    2. 下载 ffmpeg 到 bin\ffmpeg.exe
    3. 生成 settings.json（指向本目录）
    4. 打印还差什么（RVC runtime、模型）

  用法（在本文件所在目录的上一级运行）：
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

  可用参数：
    -PythonPath "C:\Python311\python.exe"   指定 Python 3.11（默认自动找 py -3.11 / python）
    -SkipTorch                              跳过 torch 安装（已经装过）
    -NoVenv                                 直接装到当前 Python（不建 venv）
#>
param(
    [string]$PythonPath = "",
    [switch]$SkipTorch,
    [switch]$NoVenv
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Say($msg) { Write-Host "  $msg" }
function Step($msg) { Write-Host ""; Write-Host "== $msg ==" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "  ! $msg" -ForegroundColor Yellow }

Write-Host ""
Write-Host "AI 翻唱工坊 v1.0 - 环境安装" -ForegroundColor Green
Say "工作目录：$Root"

# ---------------------------------------------------------------- 1. Python
Step "1/5 找 Python 3.11"

function Find-Python {
    if ($PythonPath -and (Test-Path $PythonPath)) { return $PythonPath }
    foreach ($cand in @("py -3.11", "python3.11", "python")) {
        $parts = $cand.Split(" ")
        $exe = $parts[0]
        $cmd = Get-Command $exe -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        $argList = @()
        if ($parts.Length -gt 1) { $argList += $parts[1] }
        $argList += @("-c", "import sys;print('%d.%d' % sys.version_info[:2])")
        try {
            $ver = & $exe @argList 2>$null
        } catch { continue }
        if ($ver -eq "3.11") {
            if ($parts.Length -gt 1) {
                # py -3.11：返回 exe 与固定参数
                return @{ Exe = $exe; Args = @($parts[1]) }
            }
            return @{ Exe = (Get-Command $exe).Source; Args = @() }
        }
    }
    return $null
}

$py = Find-Python
if (-not $py) {
    Warn "没找到 Python 3.11。请先从 https://www.python.org/downloads/release/python-3119/ 安装（勾 Add to PATH），或用 -PythonPath 指定。"
    exit 1
}
$pyExe = $py.Exe; $pyArgs = $py.Args
Say "使用：$pyExe $($pyArgs -join ' ')"

# ---------------------------------------------------------------- 2. venv
Step "2/5 建虚拟环境"
$pyDir = $pyExe
if ($NoVenv) {
    Say "跳过（-NoVenv）"
} else {
    if (-not (Test-Path ".venv\Scripts\python.exe")) {
        & $pyExe @pyArgs -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "venv 创建失败" }
        Say "已创建 .venv"
    } else {
        Say "已存在 .venv，复用"
    }
    $pyDir = Join-Path $Root ".venv\Scripts\python.exe"
}

& $pyDir -m pip install --upgrade pip setuptools wheel | Out-Null

# ---------------------------------------------------------------- 3. torch
Step "3/5 装 PyTorch（CUDA 12.1）"
if ($SkipTorch) {
    Say "跳过（-SkipTorch）"
} else {
    & $pyDir -m pip install "torch==2.4.1+cu121" "torchaudio==2.4.1+cu121" `
        --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) { throw "torch 安装失败（网络/代理？）" }
    & $pyDir -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
}

Step "4/5 装其余依赖"
& $pyDir -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "依赖安装失败" }
& $pyDir -m pip install "audio-separator[cpu]==0.48.0" | Out-Null
& $pyDir -m pip install matchering==2.0.6 | Out-Null

# ---------------------------------------------------------------- 4. ffmpeg
Step "5/5 准备 ffmpeg"
$ffDir = Join-Path $Root "bin"
$ffExe = Join-Path $ffDir "ffmpeg.exe"
if (Test-Path $ffExe) {
    Say "已有 $ffExe"
} else {
    New-Item -ItemType Directory -Force -Path $ffDir | Out-Null
    $zip = Join-Path $env:TEMP "ffmpeg-essentials.zip"
    $url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    Say "下载 ffmpeg（约 90 MB）…"
    try {
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
        $tmp = Join-Path $env:TEMP "ffmpeg-extract"
        if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
        Expand-Archive -Path $zip -DestinationPath $tmp -Force
        $found = Get-ChildItem $tmp -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
        Copy-Item $found.FullName $ffExe -Force
        Say "已放好：$ffExe"
    } catch {
        Warn "ffmpeg 下载失败（$($_.Exception.Message)）。请手动下载解压到 bin\ffmpeg.exe，或把已有的写进 settings.json 的 ffmpeg。"
    }
}

# ---------------------------------------------------------------- 5. settings.json
$settingsPath = Join-Path $Root "settings.json"
if (-not (Test-Path $settingsPath)) {
    $cfg = [ordered]@{
        rvc_python    = ""
        ffmpeg        = ($ffExe -replace "\\", "/")
        rvc_core      = (Join-Path $Root "rvc_core") -replace "\\", "/"
        sep_model_dir = (Join-Path $Root "models\mdxc") -replace "\\", "/"
        model_roots   = @(
            @{ path = (Join-Path $Root "rvc_models") -replace "\\", "/"; label = "本地模型"; writable = $true }
        )
    }
    $cfg | ConvertTo-Json -Depth 5 | Out-File -Encoding UTF8 $settingsPath
    Say "已写 settings.json"
} else {
    Say "settings.json 已存在，未覆盖"
}

# ---------------------------------------------------------------- 完成提示
Write-Host ""
Write-Host "环境装好了。还差三样（都不是脚本能替你做的）：" -ForegroundColor Green
Say "1) RVC runtime（随便一个 RVC 整合包里的 runtime\python.exe）"
Say "   → 填进 settings.json 的 rvc_python；并把 infer/ configs/ assets/ 复制成 rvc_core\"
Say "   → 没有也行，第 2 步会灰掉，分离和混音照常能用"
Say "2) 分离模型：第一次跑分离会自动下载到 models\mdxc（约 1.5 GB）"
Say "3) RVC 音色模型：把 .pth 和 .index 放进 rvc_models\<名字>\"
Write-Host ""
Say "启动：双击 start.bat（或 python tools\webui\app.py --port 8765 --open）"
Say "文档：docs\MANUAL.md、docs\LESSONS.md、docs\CHANGELOG.md"
Write-Host ""
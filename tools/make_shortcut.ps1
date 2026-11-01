# 在桌面创建「启动 / 停止」快捷方式（PowerShell 5.1，脚本需保存为 UTF-8 with BOM）
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$desktop = [Environment]::GetFolderPath('Desktop')
$ws = New-Object -ComObject WScript.Shell

$lnk = $ws.CreateShortcut((Join-Path $desktop 'AI Cover Studio.lnk'))
$lnk.TargetPath = Join-Path $root 'start.bat'
$lnk.WorkingDirectory = $root
$lnk.Description = 'AI Cover Studio: separate -> convert -> mix'
$lnk.WindowStyle = 7
$lnk.Save()

$lnk2 = $ws.CreateShortcut((Join-Path $desktop 'AI Cover Studio (stop).lnk'))
$lnk2.TargetPath = Join-Path $root 'stop.bat'
$lnk2.WorkingDirectory = $root
$lnk2.Description = 'Stop the local AI Cover Studio service'
$lnk2.WindowStyle = 7
$lnk2.Save()

Write-Output ('shortcuts created in ' + $desktop)

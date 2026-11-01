# Desktop shortcuts for the AI cover studio (UTF-8 with BOM required).

$ErrorActionPreference = 'Stop'

$root = '<REPO>'

$pyw  = '<REPO>/py311/pythonw.exe'

$app  = '<REPO>/tools/webui/app.py'

$desktop = [Environment]::GetFolderPath('Desktop')

$ws = New-Object -ComObject WScript.Shell



$lnk = $ws.CreateShortcut((Join-Path $desktop 'AI 翻唱工坊.lnk'))

$lnk.TargetPath = $pyw

$lnk.Arguments = '"' + $app + '" --port 8765 --open'

$lnk.WorkingDirectory = $root

$lnk.IconLocation = "$pyw,0"

$lnk.Description = 'AI 翻唱工坊：分离 → 转换 → 合成'

$lnk.WindowStyle = 7

$lnk.Save()



$lnk2 = $ws.CreateShortcut((Join-Path $desktop 'AI 翻唱工坊（停止）.lnk'))

$lnk2.TargetPath = '<REPO>/停止AI翻唱工坊.bat'

$lnk2.WorkingDirectory = $root

$lnk2.IconLocation = "$pyw,0"

$lnk2.Description = '停止 AI 翻唱工坊本地服务'

$lnk2.WindowStyle = 7

$lnk2.Save()



Write-Output ('ARGS ' + $lnk.Arguments)


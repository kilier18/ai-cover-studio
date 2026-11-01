$out = Join-Path $PSScriptRoot '..\tmp\win_roots.pem'
if (Test-Path $out) { Remove-Item $out -Force }
$sb = New-Object System.Text.StringBuilder
$certs = @()
$certs += Get-ChildItem Cert:\LocalMachine\Root -ErrorAction SilentlyContinue
$certs += Get-ChildItem Cert:\CurrentUser\Root -ErrorAction SilentlyContinue
foreach ($c in $certs) {
    [void]$sb.AppendLine('-----BEGIN CERTIFICATE-----')
    [void]$sb.AppendLine([Convert]::ToBase64String($c.RawData, 'InsertLineBreaks'))
    [void]$sb.AppendLine('-----END CERTIFICATE-----')
}
[IO.File]::WriteAllText($out, $sb.ToString(), (New-Object System.Text.UTF8Encoding($false)))
Write-Output ("certs=" + $certs.Count + " bytes=" + (Get-Item $out).Length)
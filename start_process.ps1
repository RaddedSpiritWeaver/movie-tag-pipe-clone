param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

if (-not $Command -or $Command.Count -eq 0) {
    Write-Error "Usage: .\start_process.ps1 <command> [args...]"
    exit 1
}

$exe = $Command[0]
$args = $Command[1..($Command.Count - 1)]

$outFile = "process.out.log"
$errFile = "process.err.log"

Write-Host "Starting process:"
Write-Host "  Executable: $exe"
Write-Host "  Arguments : $($args -join ' ')"
Write-Host "  StdOut    : $outFile"
Write-Host "  StdErr    : $errFile"

Start-Process $exe `
    -ArgumentList $args `
    -RedirectStandardOutput $outFile `
    -RedirectStandardError $errFile `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Process started in background."
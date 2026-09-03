# The seed sweep (#361). Nine seeds, one horizon, both quantities per run.
#
# #274's nine seeds are reused so this reading sits beside that one, but at a
# horizon held FIXED across seeds -- #274's own seeds ran to 100k, 30k and 2k,
# and dwell is cumulative, so a mixed horizon would compare a long run's dwell
# against a short one's by construction.
#
# One torch thread per process so nine runs share the box rather than each
# claiming six threads and thrashing.
#
#   pwsh prototypes/admissible-band-361/sweep.ps1 -Ticks 30000

param([int]$Ticks = 30000, [int[]]$Seeds = @(42,43,44,45,46,47,48,49,50))

$env:PYTHONPATH = "src"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

$jobs = foreach ($seed in $Seeds) {
    Start-Process -FilePath "python" -PassThru -NoNewWindow `
        -ArgumentList "prototypes/admissible-band-361/read.py","--ticks",$Ticks,"--seed",$seed `
        -RedirectStandardOutput "prototypes/admissible-band-361/log-seed$seed.txt" `
        -RedirectStandardError "prototypes/admissible-band-361/err-seed$seed.txt"
}
Write-Host "launched $($jobs.Count) runs at $Ticks ticks"
$jobs | Wait-Process
Write-Host "all runs finished"

<#
.SYNOPSIS
    Kick off ChessZero training in WSL from a Windows PowerShell session.

.DESCRIPTION
    Launches scripts/supervise_train.sh inside WSL, detached from this
    PowerShell window. The supervisor + training keep running after this
    window closes. Survives terminal disconnect / SSH death. Does NOT
    survive WSL VM restart or host reboot — for that, register this
    script as a Scheduled Task to run on logon.

    The supervisor itself handles auto-restart on python-side crashes
    (max 10 retries, max 3 consecutive no-progress attempts).

.PARAMETER RunId
    Run identifier. Used as the directory under runs/chess/<RunId>/ and
    checkpoints/chess/<RunId>/. Required.

.PARAMETER TrainLog
    Filename (relative to repo root) for the training log. Default:
    chess_train.log.run0003.

.PARAMETER StockfishGames
    Number of Stockfish warmstart games to inject. 0 = cold start (no
    warmstart pool). Default: 0.

.PARAMETER WslDistro
    Optional WSL distro name. Empty string uses the default distro.

.PARAMETER RepoPath
    Absolute path to the chessZero repo inside WSL. Default:
    /home/jacky/code/chessZero.

.EXAMPLE
    .\train-chess.ps1 -RunId 2026_05_07_test
    Cold-start run named "2026_05_07_test".

.EXAMPLE
    .\train-chess.ps1 -RunId 2026_05_07_warmstart -StockfishGames 32000
    Warmstart from 32k Stockfish games.
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$RunId,

    [string]$TrainLog = "chess_train.log.run0003",

    [int]$StockfishGames = 0,

    [string]$WslDistro = "",

    [string]$RepoPath = "/home/jacky/code/chessZero"
)

$ErrorActionPreference = "Stop"

# Bash command run inside WSL. The PowerShell here-string interpolates
# $RunId, $TrainLog, $RepoPath, $StockfishGames at this layer; backtick-$
# (`$) escapes pass literal $ through to bash.
#
# nohup + setsid + redirected stdio (</dev/null >file 2>&1) fully detach
# the supervisor from this PowerShell session, so closing the window does
# not propagate SIGHUP to the training. The supervisor then auto-restarts
# python on crash, while we still get a clean log via tail -f.
#
# OMP/MKL thread caps reduce PyTorch's autograd worker thread count, which
# shrinks the surface for the WSL2 pt_autograd_0 SIGSEGV pattern (futex/
# tgkill races). Costs ~5-15% on CPU work but training is GPU-bound.
$bashCmd = @"
set -e
cd '$RepoPath'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup setsid bash scripts/supervise_train.sh \
  --game chess \
  --run-id '$RunId' \
  --train-log '$TrainLog' \
  --stockfish-injection-games $StockfishGames \
  --stockfish-injection-interval 0 \
  --warmstart-sample-frac 0 \
  --use-gpu-chess \
  --use-tensor-mcts \
  --use-gpu-resident-self-play \
  --tensor-mcts-select-backend triton \
  --tensor-mcts-subtree-reuse \
  --tensor-mcts-hidden-dtype float16 \
  </dev/null >/tmp/supervisor_$RunId.log 2>&1 &
SUP_PID=`$!
echo started: supervisor PID=`$SUP_PID
"@

$wslArgs = @()
if ($WslDistro -ne "") {
    $wslArgs += "-d", $WslDistro
}
$wslArgs += "--", "bash", "-c", $bashCmd

# Run wsl. Output (the supervisor PID) prints to this PowerShell window
# so the user sees the launch confirmation.
& wsl @wslArgs

Write-Host ""
Write-Host "Run '$RunId' launched." -ForegroundColor Green
Write-Host ""
Write-Host "Useful commands (run from any WSL terminal or via 'wsl <cmd>' from PowerShell):"
Write-Host "  Tail supervisor stdout:  " -NoNewline
Write-Host "wsl tail -f /tmp/supervisor_$RunId.log" -ForegroundColor Cyan
Write-Host "  Tail training log:       " -NoNewline
Write-Host "wsl tail -f $RepoPath/$TrainLog" -ForegroundColor Cyan
Write-Host "  Stop cleanly (no restart):" -NoNewline
Write-Host " wsl touch $RepoPath/runs/chess/$RunId/STOP" -ForegroundColor Cyan
Write-Host "  Check process is alive:  " -NoNewline
Write-Host "wsl pgrep -af supervise_train" -ForegroundColor Cyan
Write-Host "  TensorBoard URL:         " -NoNewline
Write-Host "http://localhost:6006" -ForegroundColor Cyan

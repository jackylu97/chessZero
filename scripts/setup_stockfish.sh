#!/usr/bin/env bash
# Install a Stockfish binary into the repo (tools/stockfish/stockfish) so probes
# that need a reference engine work. Idempotent: skips if already present and
# runnable. The binary lives in the repo so it persists with the /workspace
# volume; on a fresh git-clone pod (no binary), re-run this script.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=tools/stockfish/stockfish
mkdir -p tools/stockfish

if [[ -x "$DEST" ]] && printf 'uci\nquit\n' | "$DEST" 2>/dev/null | grep -q uciok; then
  echo "stockfish already installed and runnable: $DEST"
  exit 0
fi

# Pick a build matched to CPU capability.
if grep -q avx2 /proc/cpuinfo; then BUILD=avx2; else BUILD=sse41-popcnt; fi
URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_17/stockfish-ubuntu-x86-64-${BUILD}.tar"
echo "downloading $URL"
TAR=tools/stockfish/_sf.tar
TMP=tools/stockfish/_extract
curl -sL "$URL" -o "$TAR"
rm -rf "$TMP"; mkdir -p "$TMP"
tar -xf "$TAR" -C "$TMP" 2>/dev/null || true   # ignore chown warnings
BIN=$(find "$TMP" -name "stockfish-ubuntu-*" -type f | head -1)
cp "$BIN" "$DEST"
chmod +x "$DEST"
rm -f "$TAR"; rm -rf "$TMP"

printf 'uci\nquit\n' | "$DEST" 2>/dev/null | grep -q uciok \
  && echo "installed and verified: $DEST" \
  || { echo "ERROR: stockfish not runnable after install"; exit 1; }

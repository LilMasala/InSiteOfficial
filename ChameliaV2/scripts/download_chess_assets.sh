#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/data/curriculum/stage3/chess/lichess"
ENGINE_DIR="$ROOT/external/stockfish"

mkdir -p "$DATA_DIR" "$ENGINE_DIR" "$ROOT/logs/downloads"

fetch_if_missing() {
  local url="$1"
  local out="$2"
  if [[ -s "$out" ]]; then
    echo "exists $out"
    return 0
  fi
  echo "downloading $url -> $out"
  wget -c --tries=20 --timeout=30 -O "$out" "$url"
}

fetch_if_missing \
  "https://database.lichess.org/lichess_db_puzzle.csv.zst" \
  "$DATA_DIR/lichess_db_puzzle.csv.zst"

fetch_if_missing \
  "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst" \
  "$DATA_DIR/lichess_db_standard_rated_2013-01.pgn.zst"

fetch_if_missing \
  "https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-x86-64.tar" \
  "$ENGINE_DIR/stockfish-ubuntu-x86-64.tar"

if [[ ! -d "$ENGINE_DIR/stockfish" ]]; then
  tar -xf "$ENGINE_DIR/stockfish-ubuntu-x86-64.tar" -C "$ENGINE_DIR"
fi

if [[ ! -x "$ENGINE_DIR/stockfish14-cluster-x86-64" ]]; then
  fetch_if_missing \
    "https://github.com/official-stockfish/Stockfish/archive/refs/tags/sf_14.tar.gz" \
    "$ENGINE_DIR/stockfish-sf_14.tar.gz"
  rm -rf "$ENGINE_DIR/Stockfish-sf_14"
  tar -xzf "$ENGINE_DIR/stockfish-sf_14.tar.gz" -C "$ENGINE_DIR"
  make -C "$ENGINE_DIR/Stockfish-sf_14/src" -j"${CHESS_STOCKFISH_BUILD_JOBS:-2}" build ARCH=x86-64
  cp "$ENGINE_DIR/Stockfish-sf_14/src/stockfish" "$ENGINE_DIR/stockfish14-cluster-x86-64"
  chmod +x "$ENGINE_DIR/stockfish14-cluster-x86-64"
fi

find "$DATA_DIR" -maxdepth 1 -type f -printf "%f %s bytes\n" | sort
find "$ENGINE_DIR" -maxdepth 2 -type f -name "stockfish*" -printf "%p\n" | sort

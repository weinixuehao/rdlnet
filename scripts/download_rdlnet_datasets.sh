#!/usr/bin/env bash
# Download open datasets for RDLNet stage-2 conversion (证件 / A4版面 / 票据).
# None of these are in repo JSON format yet — convert separately to ann.json + masks.
#
# Usage:
#   bash scripts/download_rdlnet_datasets.sh [DEST]
#   DEST defaults to data/raw/rdlnet_sources
#
# Optional env:
#   SKIP_MIDV=1          Skip MIDV-500 (pip package + large download)
#   SKIP_PUBLAYNET=1     Skip PubLayNet val+labels (~3.4GB + labels)
#   SKIP_CORD=1          Skip CORD v2 from Hugging Face
#   SKIP_SROIE=1         Skip SROIE zip
#   SKIP_SMARTDOC=1      Skip all SmartDoc 2015 (Zenodo)
#   SKIP_SMARTDOC_CH1=1  Skip SmartDoc Challenge 1 (document quad in video frames)
#   SMARTDOC_CH1_NO_TEST=1  Ch1: only sampleDataset.tar.gz (~21MB), skip testDataset.tar.gz (~1.5GB)
#   SMARTDOC_CH2=1       Also download SmartDoc Challenge 2 sampleDataset.zip (~12GB, mobile OCR images)
#   SMARTDOC_CH2_TEST=1  Also download Ch2 testDataset.zip (~26GB; requires SMARTDOC_CH2=1)
#
# Requires: curl, unzip, tar. For MIDV: Python + pip install midv500.
# For CORD: pip install huggingface_hub
#
# SmartDoc: ICDAR 2015 — Ch1 Zenodo 1230218, Ch2 Zenodo 2572929. CC-BY; cite competition paper.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${ROOT}/data/raw/rdlnet_sources}"
mkdir -p "$DEST"

echo "Download root: $DEST"

# --- 票据: SROIE (ICDAR 2019 scanned receipts, ~whole receipt images) ---
download_sroie() {
  if [[ "${SKIP_SROIE:-0}" == "1" ]]; then
    echo "[SROIE] skipped (SKIP_SROIE=1)"
    return 0
  fi
  local url="https://github.com/BlackStar1313/ICDAR-2019-RRC-SROIE/releases/download/v2.0/SROIE2019.zip"
  local d="$DEST/sroie2019"
  mkdir -p "$d"
  if [[ -f "$d/SROIE2019.zip" ]] || [[ -d "$d/SROIE2019" ]]; then
    echo "[SROIE] already present under $d"
    return 0
  fi
  echo "[SROIE] downloading..."
  curl -fL --retry 5 --retry-delay 10 -o "$d/SROIE2019.zip" "$url"
  unzip -q "$d/SROIE2019.zip" -d "$d" && rm -f "$d/SROIE2019.zip" || true
  echo "[SROIE] done -> $d"
}

# --- 证件: MIDV-500 (identity docs in video frames; use pip helper) ---
download_midv() {
  if [[ "${SKIP_MIDV:-0}" == "1" ]]; then
    echo "[MIDV-500] skipped (SKIP_MIDV=1)"
    return 0
  fi
  local d="$DEST/midv500"
  mkdir -p "$d"
  if [[ -n "$(find "$d" -maxdepth 3 -type f 2>/dev/null | head -1)" ]]; then
    echo "[MIDV-500] directory non-empty, skip: $d"
    return 0
  fi
  echo "[MIDV-500] using PyPI package midv500 (pip install if needed)..."
  export MIDV_DEST="$d"
  python3 - << 'PY'
import os, sys
try:
    import midv500
except ImportError:
    print("Install: pip install midv500", file=sys.stderr)
    sys.exit(2)
d = os.environ["MIDV_DEST"]
midv500.download_dataset(d, "midv500")
print("MIDV-500 download finished.")
PY
  echo "[MIDV-500] done -> $d"
}

# --- 通用A4/论文版面: PubLayNet val + COCO-style labels (IBM DAX; train split is ~100GB) ---
download_publaynet_val() {
  if [[ "${SKIP_PUBLAYNET:-0}" == "1" ]]; then
    echo "[PubLayNet] skipped (SKIP_PUBLAYNET=1)"
    return 0
  fi
  local base="https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0"
  local d="$DEST/publaynet"
  mkdir -p "$d"
  for f in labels.tar.gz val.tar.gz; do
    if [[ ! -f "$d/$f" ]] && [[ ! -d "$d/${f%.tar.gz}" ]]; then
      echo "[PubLayNet] downloading $f ..."
      curl -fL -C - --retry 5 --retry-delay 15 -o "$d/$f" "$base/$f" || {
        echo "[PubLayNet] failed: $base/$f — open https://github.com/ibm-aur-nlp/PubLayNet and download manually." >&2
        return 1
      }
    fi
  done
  echo "[PubLayNet] extracting..."
  ( cd "$d" && tar -xzf labels.tar.gz )
  ( cd "$d" && tar -xzf val.tar.gz )
  echo "[PubLayNet] done -> $d (val + labels; train-* not downloaded)"
}

# --- 票据: CORD v2 (HF dataset snapshot) ---
download_cord_hf() {
  if [[ "${SKIP_CORD:-0}" == "1" ]]; then
    echo "[CORD] skipped (SKIP_CORD=1)"
    return 0
  fi
  local d="$DEST/cord_v2"
  if [[ -d "$d" ]] && [[ -n "$(find "$d" -maxdepth 2 -type f 2>/dev/null | head -1)" ]]; then
    echo "[CORD] already present: $d"
    return 0
  fi
  echo "[CORD] Hugging Face: naver-clova-ix/cord-v2 ..."
  export CORD_DEST="$d"
  python3 - << 'PY'
import os, sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Install: pip install huggingface_hub", file=sys.stderr)
    sys.exit(2)
d = os.environ["CORD_DEST"]
snapshot_download(
    repo_id="naver-clova-ix/cord-v2",
    repo_type="dataset",
    local_dir=d,
)
print("CORD v2 snapshot done.")
PY
  echo "[CORD] done -> $d"
}

# --- SmartDoc 2015 (ICDAR): smartphone-captured documents; Zenodo direct links ---
# Ch1: preview frames + XML quads (document localization). Ch2: single images + OCR GT (large zips).
download_smartdoc() {
  if [[ "${SKIP_SMARTDOC:-0}" == "1" ]]; then
    echo "[SmartDoc] skipped (SKIP_SMARTDOC=1)"
    return 0
  fi

  if [[ "${SKIP_SMARTDOC_CH1:-0}" != "1" ]]; then
    local d1="$DEST/smartdoc2015/challenge1"
    mkdir -p "$d1"
    local base1="https://zenodo.org/records/1230218/files"
    if [[ ! -f "$d1/sampleDataset.tar.gz" ]] && [[ ! -d "$d1/sampleDataset" ]]; then
      echo "[SmartDoc Ch1] downloading sampleDataset.tar.gz (~21MB)..."
      curl -fL -C - --retry 5 --retry-delay 15 -o "$d1/sampleDataset.tar.gz" "$base1/sampleDataset.tar.gz?download=1"
    else
      echo "[SmartDoc Ch1] sample archive or folder already present"
    fi
    if [[ "${SMARTDOC_CH1_NO_TEST:-0}" != "1" ]]; then
      if [[ ! -f "$d1/testDataset.tar.gz" ]] && [[ ! -d "$d1/testDataset" ]]; then
        echo "[SmartDoc Ch1] downloading testDataset.tar.gz (~1.5GB)..."
        curl -fL -C - --retry 5 --retry-delay 15 -o "$d1/testDataset.tar.gz" "$base1/testDataset.tar.gz?download=1"
      else
        echo "[SmartDoc Ch1] test archive or folder already present"
      fi
    else
      echo "[SmartDoc Ch1] skipping test split (SMARTDOC_CH1_NO_TEST=1)"
    fi
    echo "[SmartDoc Ch1] extracting..."
    ( cd "$d1" && [[ -f sampleDataset.tar.gz ]] && tar -xzf sampleDataset.tar.gz )
    ( cd "$d1" && [[ -f testDataset.tar.gz ]] && tar -xzf testDataset.tar.gz )
    echo "[SmartDoc Ch1] done -> $d1"
  fi

  if [[ "${SMARTDOC_CH2:-0}" == "1" ]]; then
    local d2="$DEST/smartdoc2015/challenge2"
    mkdir -p "$d2"
    local base2="https://zenodo.org/records/2572929/files"
    if [[ ! -f "$d2/sampleDataset.zip" ]]; then
      echo "[SmartDoc Ch2] downloading sampleDataset.zip (~12GB)..."
      curl -fL -C - --retry 5 --retry-delay 15 -o "$d2/sampleDataset.zip" "$base2/sampleDataset.zip?download=1"
    else
      echo "[SmartDoc Ch2] sample zip or folder already present"
    fi
    if [[ "${SMARTDOC_CH2_TEST:-0}" == "1" ]]; then
      if [[ ! -f "$d2/testDataset.zip" ]]; then
        echo "[SmartDoc Ch2] downloading testDataset.zip (~26GB)..."
        curl -fL -C - --retry 5 --retry-delay 15 -o "$d2/testDataset.zip" "$base2/testDataset.zip?download=1"
      fi
    fi
    echo "[SmartDoc Ch2] extracting sample (may take a while)..."
    ( cd "$d2" && [[ -f sampleDataset.zip ]] && unzip -q -o sampleDataset.zip )
    if [[ "${SMARTDOC_CH2_TEST:-0}" == "1" ]] && [[ -f "$d2/testDataset.zip" ]]; then
      ( cd "$d2" && unzip -q -o testDataset.zip )
    fi
    echo "[SmartDoc Ch2] done -> $d2"
  else
    echo "[SmartDoc Ch2] skipped (set SMARTDOC_CH2=1 to fetch ~12GB sample; Ch2 test ~26GB with SMARTDOC_CH2_TEST=1)"
  fi
}

download_sroie || true
download_midv || echo "[MIDV-500] failed — install: pip install midv500 && re-run"
download_publaynet_val || echo "[PubLayNet] skipped or failed — use IBM DAX or Hugging Face PubLayNet mirror"
download_cord_hf || echo "[CORD] failed — pip install huggingface_hub && re-run"
download_smartdoc || echo "[SmartDoc] failed — check Zenodo / disk space"

cat << EOF

Next steps (manual):
  - Convert each dataset to RDLNet format: JSON list + mask PNGs + normalized corner points
    (see rdlnet/data/doc_json.py).
  - PubLayNet: COCO JSON in labels/; map e.g. "text" class polygon -> document mask for whole page if needed.
  - SROIE / CORD: receipt quads in annotations -> mask + corners.
  - SmartDoc Ch1: XML quads per frame -> mask + corners. Ch2: OCR text GT + captured page images.

EOF

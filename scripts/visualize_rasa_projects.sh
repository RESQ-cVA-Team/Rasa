#!/usr/bin/env bash

set -euo pipefail

OUT_FILE="build/graph.html"
WORK_DIR="build/visualize"

while [[ "$#" -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --out)
      OUT_FILE="$2"; shift 2 ;;
    --out=*)
      OUT_FILE="${1#*=}"; shift ;;
    --work-dir)
      WORK_DIR="$2"; shift 2 ;;
    --work-dir=*)
      WORK_DIR="${1#*=}"; shift ;;
    --)
      shift; break ;;
    *)
      echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 [--out <graph.html>] [--work-dir <dir>] <project_path_1> [project_path_2 ...]" >&2
  exit 1
fi

PROJECTS=("$@")
BASE_PATH="${PROJECTS[0]}"
OVERLAY_PATHS=("${PROJECTS[@]:1}")

BASE_DOMAIN="$BASE_PATH/domain"
BASE_DATA="$BASE_PATH/data"

OVERLAY_DOMAIN_ARR=()
OVERLAY_NLU_ARR=()
OVERLAY_STORIES_ARR=()

for overlay_path in "${OVERLAY_PATHS[@]}"; do
  [[ -d "$overlay_path/domain" ]] && OVERLAY_DOMAIN_ARR+=("$overlay_path/domain")
  [[ -d "$overlay_path/data/nlu" ]] && OVERLAY_NLU_ARR+=("$overlay_path/data/nlu")
  [[ -d "$overlay_path/data" ]] && OVERLAY_STORIES_ARR+=("$overlay_path/data")
done

OVERLAY_DOMAIN_STR=$(IFS=,; echo "${OVERLAY_DOMAIN_ARR[*]}")
OVERLAY_NLU_STR=$(IFS=,; echo "${OVERLAY_NLU_ARR[*]}")
OVERLAY_STORIES_STR=$(IFS=,; echo "${OVERLAY_STORIES_ARR[*]}")

export OVERLAY_BASE_DOMAIN="$BASE_DOMAIN"
export OVERLAY_DOMAIN="$OVERLAY_DOMAIN_STR"
export OVERLAY_NLU="$OVERLAY_NLU_STR"
export OVERLAY_STORIES="$OVERLAY_STORIES_STR"
export PYTHONPATH="${PYTHONPATH:-$PWD}"

MERGED_DIR="$WORK_DIR/merged"
STORIES_DIR="$WORK_DIR/stories"
MERGED_DOMAIN="$MERGED_DIR/merged-domain.yml"
MERGED_NLU="$MERGED_DIR/merged-nlu.yml"

rm -rf "$WORK_DIR"
mkdir -p "$MERGED_DIR" "$STORIES_DIR"

bash "$(dirname "$0")/layer_rasa_projects.sh" --dry-run=files --dump-dir "$MERGED_DIR" "${PROJECTS[@]}"

if [[ -d "$BASE_DATA" ]]; then
  cp -R "$BASE_DATA/." "$STORIES_DIR/"
fi

for story_root in "${OVERLAY_STORIES_ARR[@]}"; do
  if [[ -d "$story_root" ]]; then
    cp -R "$story_root/." "$STORIES_DIR/"
  fi
done

mkdir -p "$(dirname "$OUT_FILE")"

rasa visualize \
  --domain "$MERGED_DOMAIN" \
  --stories "$STORIES_DIR" \
  --nlu "$MERGED_NLU" \
  --out "$OUT_FILE"

echo "Graph written to $OUT_FILE"

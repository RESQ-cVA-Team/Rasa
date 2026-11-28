#!/bin/bash

# Usage:
#   ./layer_rasa_projects.sh [--dry-run[=stdout|files]] [--dump-dir <dir>] /path/to/project1 [/path/to/project2 ...]
# Layers Rasa projects in the order provided (first is the base layer). When --dry-run is supplied,
# it will dump the merged domain and NLU without running 'rasa train'.

set -euo pipefail

DRY_RUN_MODE=""
DUMP_DIR="build/overlay-dump"

while [[ "$#" -gt 0 && "$1" == --* ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN_MODE="files"; shift ;;
    --dry-run=stdout)
      DRY_RUN_MODE="stdout"; shift ;;
    --dry-run=files)
      DRY_RUN_MODE="files"; shift ;;
    --dump-dir)
      DUMP_DIR="$2"; shift 2 ;;
    --dump-dir=*)
      DUMP_DIR="${1#*=}"; shift ;;
    --)
      shift; break ;;
    *)
      echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 [--dry-run[=stdout|files]] [--dump-dir <dir>] <project_path_1> [project_path_2 ...]"
  exit 1
fi

PROJECTS=("$@")

BASE_PATH="${PROJECTS[0]}"
OVERLAY_PATHS=("${PROJECTS[@]:1}")

BASE_DOMAIN="$BASE_PATH/domain"
BASE_CONFIG="src/core/config.yml"
BASE_STORIES="$BASE_PATH/data"

OVERLAY_DOMAIN_ARR=()
OVERLAY_NLU_ARR=()
OVERLAY_STORIES_ARR=()

for overlay_path in "${OVERLAY_PATHS[@]}"; do
  [ -d "$overlay_path/domain" ] && OVERLAY_DOMAIN_ARR+=("$overlay_path/domain")
  [ -d "$overlay_path/data/nlu" ] && OVERLAY_NLU_ARR+=("$overlay_path/data/nlu")
  [ -d "$overlay_path/data" ] && OVERLAY_STORIES_ARR+=("$overlay_path/data")
done

OVERLAY_DOMAIN_STR=$(IFS=,; echo "${OVERLAY_DOMAIN_ARR[*]}")
OVERLAY_NLU_STR=$(IFS=,; echo "${OVERLAY_NLU_ARR[*]}")
OVERLAY_STORIES_STR=$(IFS=,; echo "${OVERLAY_STORIES_ARR[*]}")

export OVERLAY_BASE_DOMAIN="$BASE_DOMAIN"
export OVERLAY_DOMAIN="$OVERLAY_DOMAIN_STR"
export OVERLAY_NLU="$OVERLAY_NLU_STR"
export OVERLAY_STORIES="$OVERLAY_STORIES_STR"

export PYTHONPATH="${PYTHONPATH:-$PWD}"

# Persist overlay info for runtime inspection/debugging
mkdir -p build
{
  echo "BASE_DOMAIN=$BASE_DOMAIN"
  echo "OVERLAY_DOMAIN=$OVERLAY_DOMAIN_STR"
  echo "OVERLAY_NLU=$OVERLAY_NLU_STR"
  echo -n "PROJECTS="; printf '%s ' "${PROJECTS[@]}"; echo
} > build/overlay-info.txt

if [[ -n "$DRY_RUN_MODE" ]]; then
  if [[ "$DRY_RUN_MODE" == "stdout" ]]; then
    export OVERLAY_DUMP_DOMAIN=stdout
    export OVERLAY_DUMP_NLU=stdout
  else
    mkdir -p "$DUMP_DIR"
    export OVERLAY_DUMP_DOMAIN="$DUMP_DIR/merged-domain.yml"
    export OVERLAY_DUMP_NLU="$DUMP_DIR/merged-nlu.yml"
    echo "Dry run: writing merged files to $DUMP_DIR" >&2
  fi

  python - <<'PY' "$BASE_DOMAIN" "${OVERLAY_DOMAIN_ARR[@]}"
import os, sys
from src.components.layered_importer import OverlayImporter

base_domain = [sys.argv[1]]
overlay_domain = sys.argv[2:]
overlay_nlu = os.environ.get('OVERLAY_NLU')
imp = OverlayImporter(base_domain=base_domain, overlay_domain=overlay_domain, overlay_nlu=overlay_nlu)

# Trigger merges; dumps are controlled by env vars OVERLAY_DUMP_DOMAIN / OVERLAY_DUMP_NLU
imp.get_domain()
imp.get_nlu_data()
PY
  exit 0
fi

python - <<'PY'
import sys
print('sys.path=', sys.path)
try:
  import src.components.layered_importer as li
  from src.components.layered_importer import OverlayImporter
  print('OverlayImporter import OK:', li.__file__)
except Exception as e:
  print('Failed to import OverlayImporter:', e)
  raise
try:
  import src.components.entity_consolidator as ec
  print('EntityConsolidator import OK:', ec.__file__)
except Exception as e:
  print('Failed to import EntityConsolidator:', e)
  raise
PY

# The trained model depends on TensorFlow's thread counts as well as the seeds in the config.
export TF_INTRA_OP_PARALLELISM_THREADS=4 TF_INTER_OP_PARALLELISM_THREADS=4
rasa train -d "$BASE_DOMAIN" --config "$BASE_CONFIG" --data "$BASE_STORIES"

#!/usr/bin/env bash

set -euo pipefail

OUT_FILE="build/graph.html"
WORK_DIR="build/visualize"
NO_US_FALLBACK="false"
ARGS_TO_FORWARD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out|--work-dir)
      ARGS_TO_FORWARD+=("$1" "$2"); shift 2 ;;
    --out=*|--work-dir=*)
      ARGS_TO_FORWARD+=("$1"); shift ;;
    --no-us-fallback)
      NO_US_FALLBACK="true"; shift ;;
    --)
      shift; break ;;
    --*)
      ARGS_TO_FORWARD+=("$1"); shift ;;
    *)
      break ;;
  esac
done

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [--out <graph.html>] [--work-dir <dir>] [--no-us-fallback] <lang_spec>" >&2
  exit 2
fi

LANG_SPEC="$1"
LANG_CODE="$LANG_SPEC"
REGION=""

if [[ "$LANG_SPEC" == */* ]]; then
  REGION="${LANG_SPEC#*/}"
  LANG_CODE="${LANG_SPEC%%/*}"
fi

LANG_CODE_CANON="${LANG_CODE,,}"
REGION_CANON=""
if [[ -n "$REGION" ]]; then
  if [[ "$REGION" =~ ^[0-9]+$ ]]; then
    REGION_CANON="$REGION"
  elif [[ ${#REGION} -eq 4 && ${REGION:0:1} =~ [A-Z] && ${REGION:1} =~ [a-z][a-z][a-z] ]]; then
    REGION_CANON="$REGION"
  else
    REGION_CANON="${REGION^^}"
  fi
fi

LAYERS=("src/core")
if [[ "$NO_US_FALLBACK" != "true" ]]; then
  LAYERS+=("src/locales/en/US")
fi
if [[ "$LANG_CODE_CANON" != "en" ]]; then
  LAYERS+=("src/locales/$LANG_CODE_CANON")
fi
if [[ -n "$REGION_CANON" ]]; then
  LAYERS+=("src/locales/$LANG_CODE_CANON/$REGION_CANON")
fi

EXISTING=()
for path in "${LAYERS[@]}"; do
  if [[ -d "$path" ]]; then
    skip="false"
    for existing in "${EXISTING[@]}"; do
      if [[ "$existing" == "$path" ]]; then
        skip="true"
        break
      fi
    done
    if [[ "$skip" != "true" ]]; then
      EXISTING+=("$path")
    fi
  fi
done

if [[ ${#EXISTING[@]} -eq 0 ]]; then
  echo "No valid layer directories found for spec '$LANG_SPEC' (checked: ${LAYERS[*]})." >&2
  exit 3
fi

echo "Using layers: ${EXISTING[*]}"

CMD=("bash" "$(dirname "$0")/visualize_rasa_projects.sh")
CMD+=("${ARGS_TO_FORWARD[@]}")
CMD+=("${EXISTING[@]}")

"${CMD[@]}"

#!/bin/bash
set -euo pipefail
# set -x

OVERWRITE=true
# optional first flag to disable overwriting
if [[ ${1-} == "-n" || ${1-} == "--no-overwrite" ]]; then
  OVERWRITE=false
  shift
fi

# expect one positional arg: the campaign directory
if [[ $# -lt 1 ]]; then
  echo "usage: $0 [-n|--no-overwrite] <campaign-directory>" >&2
  exit 2
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 <campaign-directory>"
    exit 1
fi

for file in "$1"/*.json; do
  if [ -f "$file" ]; then
    python3 postprocess.py "$1" "$file" "$OVERWRITE"
  else
    echo "No .json files found in the directory."
    exit 1
  fi
done

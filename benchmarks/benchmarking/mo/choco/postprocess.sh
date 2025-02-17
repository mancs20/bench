#!/bin/bash

# set -x

# Check if a directory was provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <campaign-directory>"
    exit 1
fi

for file in "$1"/*.json; do
  if [ -f "$file" ]; then
    python3 postprocess.py "$1" "$file"
  else
    echo "No .json files found in the directory."
    exit 1
  fi
done

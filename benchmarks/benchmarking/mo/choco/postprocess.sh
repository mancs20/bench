#!/bin/bash

set -x

# Check if a directory was provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <campaign-directory>"
    exit 1
fi

# Create a temporary directory with all the yaml files for mzn-bench.
#MOBENCH_TMP=$1/tmp
#mkdir -p $MOBENCH_TMP
#MOBENCH_TMP=$(realpath $MOBENCH_TMP)
for file in "$1"/*.json; do
  if [ -f "$file" ]; then
#    echo "$1 $file"
    python3 postprocess.py "$1" "$file"
  else
    echo "No .json files found in the directory."
    exit 1
  fi
done
#
#for file in $1/*.json; do
#  python3 postprocess.py "$1" "$file"
#done

#mzn-bench collect-objectives $MOBENCH_TMP $1/../$(basename $1)-objectives.csv
#mzn-bench collect-statistics $MOBENCH_TMP $1/../$(basename $1).csv
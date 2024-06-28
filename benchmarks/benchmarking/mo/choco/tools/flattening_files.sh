#!/bin/bash

#set -x

# Check if a directory was provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dzn-files-directory>"
    exit 1
fi

MZN_LIB_PATH="/Users/manuel.combarrosimon/choco-solver/choco-solver-4.10.14_pareto_unsatisfaction/parsers/src/main/minizinc/mzn_lib"
MODEL_PATH="$1/sims_cost_resolution.mzn"

for file in "$1"/*.dzn; do
  if [ -f "$file" ]; then
    output_file="${file%.dzn}.fzn"
    echo "Start flattening file $file"
    minizinc -c --solver org.minizinc.mzn-fzn -I "$MZN_LIB_PATH" "$MODEL_PATH" -d "$file" -o "$output_file"
    echo "Flattened file $file"
  else
    echo "No .dzn files found in the directory."
    exit 1
  fi
done
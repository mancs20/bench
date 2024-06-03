from sys import stdin
from pathlib import Path
import sys
import os
import json
import datetime

if os.environ.get("MZN_DEBUG", "OFF") == "ON":
  import logging
  logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

if __name__ == "__main__":
  output_dir = sys.argv[1]
  benchmark = sys.argv[2]
  problem = sys.argv[3]
  instance_name = sys.argv[4]
  front_generator = sys.argv[5]
  solver = sys.argv[6]
  solver_version = sys.argv[7]
  cores = sys.argv[8]
  threads = sys.argv[9]
  timeout = sys.argv[10]
  mem = sys.argv[11]

  # data = Path(sys.argv[4])
  # solver = sys.argv[5]
  # cores = sys.argv[6]
  # threads = sys.argv[7]
  extras = []
  for i in range(10, len(sys.argv)):
    arg = sys.argv[i].strip().replace(' ', '-')
    if arg != "" and arg != "-s": # we use "-s" when there are "no special options to be used".
      extras.append(arg)
      # Remove leading "-" from extras (these are used for specifying options)
      if extras[-1].startswith("-"):
        extras[-1] = extras[-1][1:]

  uid = (solver.replace('.', '-') + solver_version.replace('.', '-') + "_" + problem
         + "_" + benchmark + "_" + instance_name + "_" + front_generator)
  if cores != "1" or threads != "1":
    uid += f"_{cores}cores_{threads}threads"
  if len(extras) > 0:
    uid += "_"
    uid += "_".join(extras)

  if(output_dir[-1] == "/"):
    output_dir = output_dir[:-1]
  if(Path(output_dir).exists() == False):
    os.mkdir(output_dir)
  log_filename = Path(output_dir + "/" + uid + ".json")

  stat_base = {
    "benchmark": benchmark,
    "problem": problem,
    "instance": instance_name,
    "solver": solver,
    "solver_version": solver_version,
    "datetime": datetime.datetime.now().isoformat(),
    "front_generator": front_generator,
    "timeout": timeout,
    "cores": cores,
    "threads": threads,
    "mem": mem,
  }

  # If the file exists, we do not delete what is already inside but append new content.
  # We start all benchmarks with a special line {"lattice-land/bench": "start"}.
  print("Writing to file: ", log_filename)
  with open(log_filename, "a") as file:
    header = {"type": "lattice-land", "lattice-land": "start"}
    json.dump(header, file)
    file.write("\n")
    msg = {"type": "statistics", "statistics": stat_base}
    json.dump(msg, file)
    file.write("\n")
    for line in stdin:
      file.write(line)
      file.flush()

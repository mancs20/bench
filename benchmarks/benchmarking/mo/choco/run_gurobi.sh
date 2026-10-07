#!/bin/bash -l
#SBATCH --time=01:45:00
#SBATCH --nodes=68
#SBATCH --partition=batch
#SBATCH --ntasks-per-node=8 # when benchmarking sequential solver, perform at most 8 experiments per node in aion. One experiment per socket/cpu.
#SBATCH --ntasks-per-socket=1
#SBATCH -c 16
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --export=ALL
#SBATCH --output=slurm-mo-gurobi.out

# Exits when an error occurs.
set -e
set -o pipefail
set -x # useful for debugging.

# This is the Gurobi counterpart of run.sh: same campaign/CSV mechanics (dump.py + postprocess.py are reused
# unchanged), but instead of calling the Choco jar it calls mosaic_image_combination's own main.py, which now
# prints the same JSON-lines protocol dump.py expects (see print_bench_json_protocol() in model/mo/main.py).
# main.py only gets the 4 CSV fields (benchmark_name, problem, instance, instance_path); the model, the data and
# the problem type are derived from the instance path by mosaic's Config (bench mode).
# For now this only runs the Saugmecon front generator, as requested -- no new front-generator strategy added.

# Shortcuts of paths to benchmarking directories.
CHOCO_WORKFLOW_PATH=$(dirname "$(realpath "run_gurobi.sh")")
BENCHMARKING_DIR_PATH="$CHOCO_WORKFLOW_PATH/../.."
BENCHMARKS_DIR_PATH="$CHOCO_WORKFLOW_PATH/../../.."

# TODO: set this to wherever mosaic_image_combination is checked out on this machine.
MOSAIC_REPO_PATH="$HOME/mosaic_image_combination"
MAIN_PY_PATH="$MOSAIC_REPO_PATH/model/mo/main.py"

# Configure the environment.
if [ -z "$1" ]; then
  echo "Usage: $0 <machine.sh>"
  echo "  Name of the machine running the experiments with the configuration of the environment."
  exit 1
fi
source $1
source "${BENCHMARKS_DIR_PATH}"/../pybench/bin/activate

# If it has an argument, we retry the jobs that failed on a previous run.
RESUME_FLAG="--resume"
if [ "$2" = "retry" ]; then
  RESUME_FLAG="--resume-failed"
fi

# I. Define the campaign to run.

TIMEOUT=5400            # keep in sync with the Choco run.sh timeout you want to compare against.
CORES=${SLURM_CPUS_PER_TASK:-16}
REAL_USED_CORES=1       # actual thread count passed to Gurobi -- kept at 1 to match Choco's single-threaded search
                        # (run.sh's THREADS=1) and mosaic's own gnu_parallel_launcher_experiments.sh convention.
THREADS=1
TASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-16}
TASKS_PER_SOCKET=${SLURM_NTASKS_PER_SOCKET:-1}

MACHINE=$(basename "$1" ".sh")
# Same 4-column CSV format as Choco's (benchmark_name,problem,instance,instance_path) -- reuse the same file
# (or a copy of it) so both solvers run on the exact same instance set.
INSTANCES_PATH="$BENCHMARKS_DIR_PATH/benchmarking/mo_choco.csv"
FRONT_STRATEGY="saugmecon"   # the only mosaic front-generator name matching one of Choco's: model/mo/FrontGenerators/Saugmecon.py

MEM_GB_PER_XP=32
# II. Prepare the command lines and output directory.
VERSION="12.0.1"
SOLVER="gurobi"
SUMMARY_FILE="$BENCHMARKS_DIR_PATH/benchmarking/mo_gurobi_summary_$MACHINE.csv"  # mosaic's own CSV, kept as a side record; the campaign CSV below is what you compare against Choco.
OUTPUT_DIR="$BENCHMARKS_DIR_PATH/campaign/$MACHINE/mo/$SOLVER-$VERSION"
mkdir -p "$OUTPUT_DIR"

# If we are on the HPC, we encapsulate the command in a srun command to reserve the resources needed.
if [ -n "${SLURM_JOB_NODELIST}" ]; then
  SRUN_COMMAND="srun --exclusive --cpus-per-task=$CORES --nodes=1 --ntasks=1 --cpu-bind=verbose"
  TOTAL_PARALLEL_TASKS=$(( SLURM_JOB_NUM_NODES * TASKS_PER_NODE ))
  NUM_PARALLEL_EXPERIMENTS=$TOTAL_PARALLEL_TASKS
else
  NUM_PARALLEL_EXPERIMENTS=1
fi

DUMP_PY_PATH="$CHOCO_WORKFLOW_PATH/dump.py"   # reused unchanged from the Choco workflow.

# For replicability.
cp -r "$CHOCO_WORKFLOW_PATH" "$OUTPUT_DIR/"
cp "$INSTANCES_PATH" "$OUTPUT_DIR/$(basename "$CHOCO_WORKFLOW_PATH")/"

lshw -json > "$OUTPUT_DIR/$(basename "$CHOCO_WORKFLOW_PATH")/hardware-$MACHINE".json 2> /dev/null

# III. Run the experiments in parallel. One `parallel` task per instance (no front-generator cross-product
# this time, since we're only running Saugmecon).

COMMANDS_LOG="$OUTPUT_DIR/$(basename "$CHOCO_WORKFLOW_PATH")/jobs-gurobi.log"
parallel --verbose --no-run-if-empty --rpl '{} uq()' -k --colsep ',' --skip-first-line -j "$NUM_PARALLEL_EXPERIMENTS" $RESUME_FLAG --joblog "$COMMANDS_LOG" " $SRUN_COMMAND python3 \"$MAIN_PY_PATH\" --compute_hypervolume_per_solution 0 --benchmark_name {1} --problem_name {2} --instance_name {3} --instance_path \"$BENCHMARKING_DIR_PATH\"/{4} --solver_name $SOLVER --front_strategy $FRONT_STRATEGY --solver_timeout_sec $TIMEOUT --summary \"$SUMMARY_FILE\" --cores $REAL_USED_CORES --solver_search_strategy free --fzn_optimisation_level 1 2>&1 | python3 \"$DUMP_PY_PATH\" \"$OUTPUT_DIR\" {1} {2} {3} $FRONT_STRATEGY $SOLVER $VERSION $CORES $THREADS $TIMEOUT $MEM_GB_PER_XP " :::: "$INSTANCES_PATH"

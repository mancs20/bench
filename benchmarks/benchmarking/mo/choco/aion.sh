#!/bin/bash

# See installation instructions for Choco.
module use $HOME/.local/easybuild/${ULHPC_CLUSTER}/turbo/${RESIF_ARCH}/modules/all

module load env/legacy/2020b
module load lang/Python/3.8.6-GCCcore-10.2.0
module load lang/Java/21.0.2 # for Choco
module load env/development/2024a
module load math/Gurobi/12.0.1-GCCcore-13.3.0 # for the Gurobi runs -- configured to use the SIU-managed license server
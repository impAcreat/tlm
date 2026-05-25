#!/usr/bin/env bash
set -euo pipefail

# Creates the user-requested Conda environment name: agent.
# Keep this script tiny so it remains auditable on shared servers.
/sdc/ninghan/miniforge3/bin/conda env create -f environment.yml

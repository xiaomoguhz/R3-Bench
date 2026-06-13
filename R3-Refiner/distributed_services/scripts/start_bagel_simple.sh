#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"

if [[ -n "${CONDA_SH:-}" ]]; then
    source "${CONDA_SH}"
    conda activate "${EDIT_CONDA_ENV:-bagelfast}"
fi

export INSTANCES_PER_GPU="${INSTANCES_PER_GPU:-1}"
model_path="${EDIT_MODEL_PATH:?Set EDIT_MODEL_PATH to the BAGEL checkpoint directory.}"

bash distributed_services/scripts/deploy_services.sh edit_server "${model_path}" bagel

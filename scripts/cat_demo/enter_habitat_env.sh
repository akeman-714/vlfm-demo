#!/usr/bin/env bash
# Start a small browser-based Habitat viewer for the merged cat scene.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_pip

pick_idle_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F',' '
      {
        gsub(/ /, "", $1);
        gsub(/ /, "", $2);
        if (best == "" || $2 < best_mem) {
          best = $1;
          best_mem = $2;
        }
      }
      END {
        if (best != "") print best;
      }'
}

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU="$(pick_idle_gpu)"
  if [ -z "${GPU}" ]; then
    echo "Could not query an idle GPU with nvidia-smi." >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi

export HABITAT_VIEWER_HOST="${HABITAT_VIEWER_HOST:-127.0.0.1}"
export HABITAT_VIEWER_PORT="${HABITAT_VIEWER_PORT:-7862}"

echo ">>> Habitat Cat Scene Viewer"
echo ">>> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo ">>> server URL : http://${HABITAT_VIEWER_HOST}:${HABITAT_VIEWER_PORT}"
echo ">>> laptop URL : http://127.0.0.1:17862"
echo ">>> SSH tunnel : ssh -N -L 17862:127.0.0.1:${HABITAT_VIEWER_PORT} <user>@<server>"
echo ">>> Controls   : W/S forward/back, A/D turn, Q/E strafe, Z/C look, R reset"
echo

exec python scripts/cat_demo/enter_habitat_env.py "$@"

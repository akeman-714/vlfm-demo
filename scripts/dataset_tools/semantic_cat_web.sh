#!/usr/bin/env bash
# Start the end-to-end semantic cat finder web UI.
set -u

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

PRIVATE_ENV="${SEMANTIC_CAT_WEB_ENV:-${HOME}/.config/vlfm-demo/semantic_cat_web.env}"
if [ -f "${PRIVATE_ENV}" ]; then
  # Private convenience file for BAILIAN_API_KEY and local GPU/port overrides.
  # Keep it outside the repo.
  set -a
  # shellcheck disable=SC1090
  source "${PRIVATE_ENV}"
  set +a
fi

source /data/jinsong.yuan/miniconda3/etc/profile.d/conda.sh
conda activate vlfm_pip

export BAILIAN_BASE_URL="${BAILIAN_BASE_URL:-https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1}"
export BAILIAN_MODEL="${BAILIAN_MODEL:-qwen3.6-flash}"
export SEMANTIC_CAT_WEB_HOST="${SEMANTIC_CAT_WEB_HOST:-127.0.0.1}"
export SEMANTIC_CAT_WEB_PORT="${SEMANTIC_CAT_WEB_PORT:-7861}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,0}"
export VLFM_POINTNAV_GPU_ID="${VLFM_POINTNAV_GPU_ID:-1}"

echo ">>> ObjectNav Run Web"
echo ">>> URL on cloud : http://${SEMANTIC_CAT_WEB_HOST}:${SEMANTIC_CAT_WEB_PORT}"
echo ">>> SSH tunnel   : ssh -N -L 17861:127.0.0.1:${SEMANTIC_CAT_WEB_PORT} -p 20755 jinsong.yuan@120.133.130.214"
echo ">>> Laptop URL   : http://127.0.0.1:17861"
echo ">>> CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  (sim/EGL=cuda:0, nav=cuda:${VLFM_POINTNAV_GPU_ID})"
echo ">>> Run modes    : semantic_query, find_cat, global_home_40, global_home_100, object_memory_cat, persistent_memory_cat_pair"

PIDS_ON_PORT="$(ss -ltnp 2>/dev/null | awk -v port=":${SEMANTIC_CAT_WEB_PORT}" '
  $4 ~ port"$" {
    line=$0
    while (match(line, /pid=[0-9]+/)) {
      print substr(line, RSTART + 4, RLENGTH - 4)
      line = substr(line, RSTART + RLENGTH)
    }
  }
' | sort -u)"
if [ -n "${PIDS_ON_PORT}" ]; then
  for pid in ${PIDS_ON_PORT}; do
    cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    if printf '%s' "${cmd}" | grep -q 'scripts/dataset_tools/semantic_cat_web.py'; then
      echo ">>> Replacing old semantic_cat_web.py on port ${SEMANTIC_CAT_WEB_PORT}: pid ${pid}"
      kill "${pid}" 2>/dev/null || true
    else
      echo "Port ${SEMANTIC_CAT_WEB_PORT} is occupied by pid ${pid}: ${cmd}" >&2
      echo "Refusing to kill a non-semantic-cat-web process." >&2
      exit 98
    fi
  done
  sleep 1
fi

python scripts/dataset_tools/semantic_cat_web.py

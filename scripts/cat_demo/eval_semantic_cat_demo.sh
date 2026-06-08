#!/usr/bin/env bash
# Resolve a natural-language request with Bailian, then run the existing cat demo.
#
# Example:
#   export BAILIAN_API_KEY=...
#   bash scripts/cat_demo/eval_semantic_cat_demo.sh "咪咪你在哪"
#
# Optional:
#   export BAILIAN_BASE_URL=https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
#   export BAILIAN_MODEL=qwen3.6-flash   # or qwen3.6-plus
set -u

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

REQUEST_TEXT="${*:-${GOAL_TEXT:-咪咪你在哪}}"

echo ">>> Semantic ObjectNav request: ${REQUEST_TEXT}"
LABEL="$(python scripts/cat_demo/semantic_goal_head.py --text "${REQUEST_TEXT}")"
STATUS=$?
if [ "${STATUS}" -ne 0 ]; then
  echo "semantic_goal_head failed with exit code ${STATUS}" >&2
  exit "${STATUS}"
fi

echo ">>> Resolved target label: ${LABEL}"
if [ "${LABEL}" != "cat" ]; then
  echo "Only label 'cat' is wired to a demo split right now; got '${LABEL}'." >&2
  exit 4
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo ">>> DRY_RUN=1, semantic routing is ready; skipping eval_cat_demo.sh"
  exit 0
fi

export SPLIT="${SPLIT:-cat_demo}"
bash scripts/cat_demo/eval_cat_demo.sh

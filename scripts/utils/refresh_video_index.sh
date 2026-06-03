#!/usr/bin/env bash
# Re-build the index.html for the most recent 100ep_parallel run (or a
# user-specified dir). Re-run this any time you want the page to pick up new
# mp4s. The HTTP server keeps serving — just hit reload in the browser.
#
# Usage:
#   bash scripts/refresh_video_index.sh                # auto-pick latest run
#   bash scripts/refresh_video_index.sh outputs/<dir>  # explicit dir
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ $# -ge 1 ]; then
  ROOT="$1"
else
  ROOT="$(ls -dt outputs/100ep_parallel_*/ 2>/dev/null | head -1)"
fi
ROOT="${ROOT%/}"

if [ -z "${ROOT}" ] || [ ! -d "${ROOT}" ]; then
  echo "no run dir found (looked for outputs/100ep_parallel_*/)" >&2
  exit 2
fi

python3 scripts/utils/build_video_index.py "${ROOT}"

# Repoint outputs/latest -> this run so the permanent bookmark
# http://<host>:7777/latest/index.html always shows the freshest run.
TARGET_NAME="$(basename "${ROOT}")"
ln -sfn "${TARGET_NAME}" outputs/latest
echo "outputs/latest -> ${TARGET_NAME}"
echo "open: http://120.133.130.214:7777/latest/index.html"

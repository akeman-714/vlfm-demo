#!/usr/bin/env bash
# Compact status snapshot for the 23h demo-harvest run.
# Reads outputs/.last_demo_stamp for ROOT_OUT / wrapper PID, then prints a
# one-screen health summary: process tree alive, GPU 4/5 usage, current scene,
# attempts/steps, success/failure counters from filenames, recent log activity.
#
# Designed to be cheap (no heavy parsing) so we can poll every 2h without
# perturbing the run.

set -u

cd "$(dirname "$0")/../.."

STAMP_FILE="outputs/.last_demo_stamp"
if [ ! -f "${STAMP_FILE}" ]; then
  echo "[check] missing ${STAMP_FILE} -- no run registered."
  exit 1
fi

STAMP="$(grep '^STAMP_SAVED=' "${STAMP_FILE}" | cut -d= -f2-)"
ROOT_OUT="$(grep '^ROOT_OUT_SAVED=' "${STAMP_FILE}" | cut -d= -f2-)"
STATE_LINE="$(cat outputs/100ep_watcher.state 2>/dev/null || echo IDLE)"
WRAP_PID="$(echo "${STATE_LINE}" | awk '{print $2}')"

NOW="$(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo "[check ${NOW}] 23h demo-harvest status"
echo "  stamp     = ${STAMP}"
echo "  out_dir   = ${ROOT_OUT}"
echo "  wrap_pid  = ${WRAP_PID}   state_line=\"${STATE_LINE}\""
echo "================================================================"

# --- process tree ---------------------------------------------------------
echo "-- process tree --"
if [ -n "${WRAP_PID}" ] && kill -0 "${WRAP_PID}" 2>/dev/null; then
  ps -p "${WRAP_PID}" -o pid,etime,stat,cmd | sed 's/  */ /g'
else
  echo "  wrap_pid=${WRAP_PID} NOT ALIVE"
fi
ps -ef | grep -E "run_100ep_parallel|eval_itm_policy_split|vlfm\.run" \
  | grep -v grep | awk '{printf "  %s %s %s %s\n", $2, $3, $7, $8}'

# --- GPU snapshot ---------------------------------------------------------
echo "-- gpu 4/5 usage --"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader | awk -F, '$1+0==4 || $1+0==5 {print "  gpu" $0}'

# --- VLM health probes ----------------------------------------------------
echo "-- vlm probes --"
for p in 12181 12182 12183 12184; do
  code="$(curl -s -o /dev/null --max-time 2 -w '%{http_code}' "http://127.0.0.1:${p}/" || true)"
  printf "  port %s: HTTP %s\n" "${p}" "${code:-down}"
done

# --- run progress ---------------------------------------------------------
echo "-- run progress --"
if [ -f "${ROOT_OUT}/procA.log" ]; then
  echo "  procA.log tail:"
  tail -n 5 "${ROOT_OUT}/procA.log" | sed 's/^/    /'
fi

# Per-scene CSV: 1 row per finished scene.
if [ -f "${ROOT_OUT}/procA_scenes.csv" ]; then
  finished_scenes="$(awk 'NR>1' "${ROOT_OUT}/procA_scenes.csv" | wc -l)"
  echo "  finished scenes (csv rows) = ${finished_scenes}"
  if [ "${finished_scenes}" -gt 0 ]; then
    echo "  scene,exit,started,success,failure,rate%,wall_s"
    awk -F, 'NR>1 {printf "    %s,%s,%s,%s,%s,%s,%s\n", $1,$3,$4,$5,$6,$7,$8}' \
      "${ROOT_OUT}/procA_scenes.csv"
  fi
fi

# --- current scene attempt log -------------------------------------------
echo "-- current scene attempt --"
latest_attempt="$(ls -t "${ROOT_OUT}"/procA_logs/*.attempt*.raw 2>/dev/null | head -1)"
if [ -n "${latest_attempt}" ]; then
  echo "  log: ${latest_attempt}"
  cur_step="$(grep -oE '^Step: [0-9]+' "${latest_attempt}" | tail -1)"
  cur_ep="$(grep -cE '^Step: 0 \| Mode: initialize' "${latest_attempt}")"
  succ_so_far="$(grep -oE 'Success rate: [0-9.]+% \([0-9]+ out of [0-9]+\)' "${latest_attempt}" | tail -1)"
  echo "  attempted episodes: ${cur_ep}   ${cur_step}"
  [ -n "${succ_so_far}" ] && echo "  ${succ_so_far}"
fi

# --- video accounting (success vs failure) -------------------------------
echo "-- videos written so far --"
if [ -d "${ROOT_OUT}/procA_video" ]; then
  succ_n="$(find "${ROOT_OUT}/procA_video" -name '*success=1.00*.mp4' | wc -l)"
  fail_n="$(find "${ROOT_OUT}/procA_video" -name '*success=0.00*.mp4' | wc -l)"
  tot_n="$(find "${ROOT_OUT}/procA_video" -type f -name '*.mp4' | wc -l)"
  size_h="$(du -sh "${ROOT_OUT}/procA_video" 2>/dev/null | awk '{print $1}')"
  echo "  procA_video/  total=${tot_n}  success=${succ_n}  failure=${fail_n}  size=${size_h}"
  echo "  failure_montage/ count=$(ls "${ROOT_OUT}/failure_montage" 2>/dev/null | wc -l)"
fi

# --- disk -----------------------------------------------------------------
echo "-- /data disk free --"
df -h /data | awk 'NR<=2 {print "  " $0}'

# --- exit code summary ----------------------------------------------------
if [ -f "${ROOT_OUT}/procA.exit" ]; then
  echo "-- run terminated --"
  echo "  procA.exit = $(cat "${ROOT_OUT}/procA.exit")"
fi

echo "================================================================"

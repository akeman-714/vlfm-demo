#!/usr/bin/env bash
# 2h GPU watcher: scans all available GPUs, finds any 4 simultaneously idle
# ones, and launches scripts/legacy/run_100ep_parallel.sh with the discovered GPU
# pairs (first idle pair -> procA sim+torch, second idle pair -> procB).
# When the in-flight run finishes, emits a DONE sentinel.
#
# State file: outputs/100ep_watcher.state
#   IDLE
#   RUNNING <pid> <out_dir>
#   COMPLETED <out_dir> <exit_a> <exit_b>
#
# Heartbeat: outputs/100ep_watcher.heartbeat.log
#
# Stdout sentinels (only on state change):
#   AGENT_LOOP_TICK_VLFM_100EP_LAUNCHED {...}
#   AGENT_LOOP_TICK_VLFM_100EP_DONE     {...}
#
# Tunables (env):
#   GPU_IDLE_MEM_MIB     max memory.used [MiB] to count a GPU as idle  (default 2000)
#   GPU_IDLE_UTIL_PCT    max utilization.gpu [%] to count a GPU as idle (default 10)
#   VLM_PORTS            comma list of VLM probe ports  (default 12181-12184)
#   REQUIRE_VLM          1 => skip launch if any VLM port down          (default 1)
#   RUN_SCRIPT           path to run_100ep_parallel.sh
#
# GPU-pair discovery (two-tier):
#   Tier 1: if ≥ 4 idle GPUs found → dual-proc mode (procA + procB in parallel)
#   Tier 2: if ≥ 2 idle GPUs found → single-proc fallback (procA only, 2 GPUs)
#   < 2 idle GPUs          → stay IDLE, retry next tick

set -u
cd "$(dirname "$0")/../.."

STATE_FILE="${STATE_FILE:-outputs/100ep_watcher.state}"
HEARTBEAT_LOG="${HEARTBEAT_LOG:-outputs/100ep_watcher.heartbeat.log}"
RUN_SCRIPT="${RUN_SCRIPT:-./scripts/legacy/run_100ep_parallel.sh}"

GPU_IDLE_MEM_MIB="${GPU_IDLE_MEM_MIB:-2000}"
GPU_IDLE_UTIL_PCT="${GPU_IDLE_UTIL_PCT:-10}"
VLM_PORTS="${VLM_PORTS:-12181,12182,12183,12184}"
REQUIRE_VLM="${REQUIRE_VLM:-1}"

mkdir -p "$(dirname "${STATE_FILE}")"
[ -f "${STATE_FILE}" ] || echo "IDLE" > "${STATE_FILE}"

NOW="$(date '+%Y-%m-%d %H:%M:%S')"
log() {
  printf '[%s] %s\n' "${NOW}" "$*" >> "${HEARTBEAT_LOG}"
}

read_state() {
  cat "${STATE_FILE}" 2>/dev/null || echo "IDLE"
}

write_state() {
  printf '%s\n' "$*" > "${STATE_FILE}"
}

# Two-tier GPU discovery.  Sets PROC_A_GPUS, PROC_B_GPUS, SINGLE_PROC and
# exports them.  Returns 0 on success, 1 if fewer than 2 idle GPUs exist.
#
# Tier 1 (≥4 idle): dual-proc mode  — SINGLE_PROC=0, both pairs set.
# Tier 2 (≥2 idle): single-proc fallback — SINGLE_PROC=1, PROC_B_GPUS="".
find_idle_gpu_pairs() {
  local q
  q="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
       --format=csv,noheader,nounits 2>/dev/null || true)"
  if [ -z "${q}" ]; then
    echo "nvidia-smi returned nothing" >&2
    return 1
  fi

  local idle_gpus=()
  while IFS=',' read -r idx mem util; do
    idx="$(echo "${idx}" | tr -d ' ')"
    mem="$(echo "${mem}"  | tr -d ' ')"
    util="$(echo "${util}" | tr -d ' ')"
    [ -z "${idx}" ] && continue
    if [ "${mem:-999999}" -le "${GPU_IDLE_MEM_MIB}" ] && \
       [ "${util:-100}"   -le "${GPU_IDLE_UTIL_PCT}" ]; then
      idle_gpus+=("${idx}")
    fi
  done <<< "${q}"

  local n="${#idle_gpus[@]}"

  if [ "${n}" -lt 2 ]; then
    echo "only ${n} idle GPU(s) found (need ≥2); idle=[${idle_gpus[*]:-}]" >&2
    return 1
  fi

  PROC_A_GPUS="${idle_gpus[0]},${idle_gpus[1]}"
  export PROC_A_GPUS

  if [ "${n}" -ge 4 ]; then
    PROC_B_GPUS="${idle_gpus[2]},${idle_gpus[3]}"
    SINGLE_PROC=0
    echo "DUAL-PROC: ${n} idle GPUs [${idle_gpus[*]}]  procA=${PROC_A_GPUS}  procB=${PROC_B_GPUS}" >&2
  else
    PROC_B_GPUS=""
    SINGLE_PROC=1
    echo "SINGLE-PROC fallback: only ${n} idle GPUs [${idle_gpus[*]}]  procA=${PROC_A_GPUS}  procB=none" >&2
  fi
  export PROC_B_GPUS SINGLE_PROC
  return 0
}

check_vlm_up() {
  [ "${REQUIRE_VLM}" = "1" ] || return 0
  local IFS=','
  read -ra PORTS <<<"${VLM_PORTS}"
  unset IFS
  local p code
  for p in "${PORTS[@]}"; do
    p="$(echo "${p}" | tr -d '[:space:]')"
    code="$(curl -s -o /dev/null --max-time 2 -w '%{http_code}' "http://127.0.0.1:${p}/" || true)"
    if [ "${code}" = "000" ] || [ -z "${code}" ]; then
      echo "VLM port ${p} down (HTTP ${code:-down})" >&2
      return 1
    fi
  done
  return 0
}

STATE="$(read_state)"
WORD="$(echo "${STATE}" | awk '{print $1}')"

case "${WORD}" in
  RUNNING)
    PID="$(echo "${STATE}" | awk '{print $2}')"
    OUT="$(echo "${STATE}" | awk '{print $3}')"
    if [ -n "${PID}" ] && kill -0 "${PID}" 2>/dev/null; then
      log "RUNNING pid=${PID} out=${OUT}  (still alive)"
      exit 0
    fi
    EXIT_A="$(cat "${OUT}/procA.exit" 2>/dev/null || echo '?')"
    EXIT_B="$(cat "${OUT}/procB.exit" 2>/dev/null || echo '?')"
    write_state "COMPLETED ${OUT} ${EXIT_A} ${EXIT_B}"
    log "DONE  out=${OUT}  exit_a=${EXIT_A}  exit_b=${EXIT_B}"
    printf 'AGENT_LOOP_TICK_VLFM_100EP_DONE {"out_dir":"%s","exit_a":"%s","exit_b":"%s","prompt":"VLFM 100ep run finished at %s; read summary.json/summary.csv, build Chinese report + failure montage at outputs/100ep_watcher_report.md, then kill the 2h loop."}\n' \
      "${OUT}" "${EXIT_A}" "${EXIT_B}" "${OUT}"
    exit 0
    ;;
  COMPLETED)
    log "already COMPLETED  ${STATE#COMPLETED }  (no-op)"
    exit 0
    ;;
  *)
    : # IDLE / unknown -> fall through and consider launching.
    ;;
esac

REASON=""
if ! find_idle_gpu_pairs 2>/tmp/.watch_reason.$$; then
  REASON="$(cat /tmp/.watch_reason.$$ 2>/dev/null)"
  rm -f /tmp/.watch_reason.$$
  log "BUSY  ${REASON}"
  exit 0
fi
REASON="$(cat /tmp/.watch_reason.$$ 2>/dev/null)"
rm -f /tmp/.watch_reason.$$
log "IDLE_FOUND  ${REASON}  single_proc=${SINGLE_PROC}"

if ! check_vlm_up 2>/tmp/.watch_reason.$$; then
  REASON="$(cat /tmp/.watch_reason.$$ 2>/dev/null)"
  rm -f /tmp/.watch_reason.$$
  log "VLM_DOWN  ${REASON}  (will retry next tick)"
  exit 0
fi
rm -f /tmp/.watch_reason.$$

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="outputs/100ep_parallel_${STAMP}"
WRAP_LOG="outputs/run_100ep_parallel_wrapper_${STAMP}.out"
mkdir -p "${OUT}" outputs

# Launch detached.  Pass STAMP/ROOT_OUT through env so run_100ep_parallel.sh
# writes into the directory the watcher already chose (and recorded in state).
# nohup + setsid + disown so the wrapper survives this loop iteration and
# any future agent restart.
(
  export STAMP ROOT_OUT="${OUT}" PROC_A_GPUS PROC_B_GPUS SINGLE_PROC
  setsid nohup bash "${RUN_SCRIPT}" > "${WRAP_LOG}" 2>&1 < /dev/null &
  echo "$!"
  disown "$!" 2>/dev/null || true
) > /tmp/.watch_pid.$$ 2>/dev/null
PID="$(cat /tmp/.watch_pid.$$ 2>/dev/null)"
rm -f /tmp/.watch_pid.$$

echo "${PID} ${WRAP_LOG} ${STAMP}" > "${OUT}/launch_info.txt"
write_state "RUNNING ${PID} ${OUT}"
log "LAUNCHED pid=${PID} out=${OUT} wrap_log=${WRAP_LOG}"
printf 'AGENT_LOOP_TICK_VLFM_100EP_LAUNCHED {"pid":%s,"out_dir":"%s","wrap_log":"%s","prompt":"VLFM 100ep run just launched at %s (pid %s); no agent action needed until AGENT_LOOP_TICK_VLFM_100EP_DONE arrives."}\n' \
  "${PID}" "${OUT}" "${WRAP_LOG}" "${OUT}" "${PID}"

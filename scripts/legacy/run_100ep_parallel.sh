#!/usr/bin/env bash
# Orchestrator: run 100 HM3D ObjectNav val episodes total, split across 2
# parallel split-GPU processes.  GPU pairs are discovered dynamically by the
# watcher (PROC_A_GPUS / PROC_B_GPUS) or default to 4,5 / 1,2.
#
# Normal mode:
#   10 scenes per process, EP_PER_SCENE episodes each => 50 ep/proc, 100 total.
#   One inner-script call per scene; native-crash triggers up to MAX_NATIVE_RETRIES.
#
# Re-run mode (RERUN_EPISODE_IDS=<id,...>):
#   Split the given episode IDs evenly across procA / procB, run without scene
#   restriction so habitat resolves scenes automatically.
#   Set RERUN_N_EPISODES to override the count passed to the inner script.
#
# Key env vars:
#   PROC_A_GPUS           CUDA_VISIBLE_DEVICES for procA    (default 4,5)
#   PROC_B_GPUS           CUDA_VISIBLE_DEVICES for procB    (default 1,2)
#   RERUN_EPISODE_IDS     comma list of episode IDs to re-run (empty = normal mode)
#   EP_PER_SCENE          episodes per scene in normal mode  (default 5)
#   TIMEOUT_S             per-proc wall budget [s]           (default 9000)
#   PER_SCENE_TIMEOUT_S   per-scene wall budget [s]          (default 900)
#   MAX_NATIVE_RETRIES    retries on native crash             (default 3)
#   STAMP / ROOT_OUT      output directory override (set by watcher)
#
# Outputs (under outputs/100ep_parallel_<stamp>/):
#   procA.log / procB.log              orchestrator-level log per proc
#   procA.exit / procB.exit            worst-case exit per proc
#   procA_video/<scene>/*.mp4          failure videos (success skipped)
#   procA_logs/<scene>.log             inner-script log per scene
#   procA_scenes.csv                   per-scene metrics (scene-level)
#   procB_... (mirror of A)
#   failure_montage/                   all failure mp4s prefixed proc__scene__
#   episodes.csv                       per-episode row (parsed from filenames)
#   summary.csv / summary.json / summary.txt
#
# Pre-reqs:
#   * 4 VLM Flask servers up (run scripts/launch_vlm_servers_jy.sh 0 first)
#   * At least 4 idle GPUs (check with nvidia-smi) — watcher finds them auto
#   * conda env vlfm_cuda_sim

set -u

cd "$(dirname "$0")/../.."
SCRIPT_DIR="$(pwd)/scripts"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT_OUT="${ROOT_OUT:-outputs/100ep_parallel_${STAMP}}"
mkdir -p "${ROOT_OUT}" "${ROOT_OUT}/failure_montage"

# 20 alpha-sorted val scenes, split as first 10 / last 10.
SCENES_A_DEFAULT="4ok3usBNeis,5cdEh9F2hJL,6s7QHgap2fW,bxsVRursffK,cvZr5TUy5C5,Dd4bFSTQ8gi,DYehNKdT76V,mL8ThkuaVTM,mv2HUxq3B53,Nfvxx8J5NCo"
SCENES_B_DEFAULT="p53SfW6mjZe,q3zU7Yy5E5s,QaLdnwvtxbs,qyAac8rV8Zk,svBbv1Pavdk,TEEsavR23oF,wcojb4TFT35,XB4GS9ShBRE,ziup5kvtCCR,zt1RVoi7PcG"
SCENES_A="${SCENES_A:-${SCENES_A_DEFAULT}}"
SCENES_B="${SCENES_B:-${SCENES_B_DEFAULT}}"

# GPU pairs: set by watcher (two-tier discovery) or explicit override.
# SINGLE_PROC=1 means only procA runs (watcher found 2-3 idle GPUs).
PROC_A_GPUS="${PROC_A_GPUS:-4,5}"
PROC_B_GPUS="${PROC_B_GPUS:-1,2}"
SINGLE_PROC="${SINGLE_PROC:-0}"

EP_PER_SCENE="${EP_PER_SCENE:-5}"
TIMEOUT_S="${TIMEOUT_S:-9000}"
PER_SCENE_TIMEOUT_S="${PER_SCENE_TIMEOUT_S:-900}"
MAX_NATIVE_RETRIES="${MAX_NATIVE_RETRIES:-3}"
SPLIT="${SPLIT:-val}"

# When 1, the inner script skips success videos and writes only failure mp4s
# (saves disk).  When 0, every episode's mp4 is kept — used to harvest demo
# candidates from a long run.
VLFM_SKIP_SUCCESS_VIDEOS="${VLFM_SKIP_SUCCESS_VIDEOS:-1}"
export VLFM_SKIP_SUCCESS_VIDEOS

# Re-run mode: comma list of global episode IDs to re-run.  Empty = normal.
RERUN_EPISODE_IDS="${RERUN_EPISODE_IDS:-}"

# Optional explicit per-scene CSV columns; kept here for grep-friendliness.
SCENE_CSV_HEADER="scene,attempts,exit_final,n_started,n_success,n_failure,success_rate_pct,wall_s,fail_videos"

# Inner script path (sibling).
INNER_SCRIPT="${SCRIPT_DIR}/legacy/eval_itm_policy_split_gpu.sh"
if [ ! -x "${INNER_SCRIPT}" ] && [ ! -f "${INNER_SCRIPT}" ]; then
  echo "ERROR: inner script not found at ${INNER_SCRIPT}" >&2
  exit 2
fi

echo "================================================================"
echo "100-ep parallel runner   stamp=${STAMP}"
if [ -n "${RERUN_EPISODE_IDS}" ]; then
  echo "MODE: re-run  episode_ids=${RERUN_EPISODE_IDS}"
else
  echo "MODE: normal"
  echo "  scenes_A=${SCENES_A}"
  echo "  scenes_B=${SCENES_B}"
  echo "  ep/scene=${EP_PER_SCENE}"
fi
if [ "${SINGLE_PROC}" = "1" ]; then
  echo "process A: GPUs ${PROC_A_GPUS}   (SINGLE-PROC fallback — procB skipped)"
else
  echo "process A: GPUs ${PROC_A_GPUS}   process B: GPUs ${PROC_B_GPUS}"
fi
echo "per-proc wall budget: ${TIMEOUT_S}s  /  per-scene budget: ${PER_SCENE_TIMEOUT_S}s"
echo "native-crash retries: up to ${MAX_NATIVE_RETRIES} (exits not in {0,124})"
if [ "${VLFM_SKIP_SUCCESS_VIDEOS}" = "1" ]; then
  echo "split: ${SPLIT}    videos: failures only (VLFM_SKIP_SUCCESS_VIDEOS=1)"
else
  echo "split: ${SPLIT}    videos: ALL episodes (VLFM_SKIP_SUCCESS_VIDEOS=0, demo-harvest mode)"
fi
echo "out dir: ${ROOT_OUT}/"
echo "================================================================"
echo "-- pre-flight: VLM health --"
for p in 12181 12182 12183 12184; do
  code=$(curl -s -o /dev/null --max-time 2 -w '%{http_code}' "http://127.0.0.1:${p}/" || true)
  echo "  port ${p}: HTTP ${code:-down}"
done
echo "-- pre-flight: GPU usage --"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv | head -10
echo "================================================================"

# --------------------------------------------------------------------------
# run_one_proc <tag> <cuda_visible_devices> <scenes_csv>
#
# Iterates the scenes, calls the inner script once per scene with N_EPISODES,
# implements native-crash retry, captures per-scene CSV row, and copies
# failure mp4s into ROOT_OUT/failure_montage/ with proc+scene prefix.
# Writes worst-case exit (max scene exit) into ${ROOT_OUT}/${tag}.exit.
# --------------------------------------------------------------------------
run_one_proc() {
  local tag="$1"
  local cuda_dev="$2"
  local scenes_csv="$3"

  local proc_log="${ROOT_OUT}/${tag}.log"
  local proc_csv="${ROOT_OUT}/${tag}_scenes.csv"
  local proc_video_root="${ROOT_OUT}/${tag}_video"
  local proc_tb_root="${ROOT_OUT}/${tag}_tb"
  local proc_logs_root="${ROOT_OUT}/${tag}_logs"
  mkdir -p "${proc_video_root}" "${proc_tb_root}" "${proc_logs_root}"

  echo "${SCENE_CSV_HEADER}" > "${proc_csv}"

  exec >"${proc_log}" 2>&1
  echo "[${tag}] starting on CUDA_VISIBLE_DEVICES=${cuda_dev}"
  echo "[${tag}] scenes: ${scenes_csv}"
  echo "[${tag}] ep/scene=${EP_PER_SCENE}  per-scene budget=${PER_SCENE_TIMEOUT_S}s"
  echo "[${tag}] per-proc budget=${TIMEOUT_S}s  max retries=${MAX_NATIVE_RETRIES}"

  local proc_t_start
  proc_t_start=$(date +%s)
  local worst_exit=0

  # Iterate the CSV scenes list.
  local IFS=','
  read -ra SCENE_ARR <<<"${scenes_csv}"
  unset IFS

  local scene
  for scene in "${SCENE_ARR[@]}"; do
    # Cheap defense against trailing whitespace/blank tokens.
    scene="$(echo "${scene}" | tr -d '[:space:]')"
    [ -z "${scene}" ] && continue

    local now elapsed remaining
    now=$(date +%s)
    elapsed=$((now - proc_t_start))
    remaining=$((TIMEOUT_S - elapsed))
    if [ "${remaining}" -le 60 ]; then
      echo "[${tag}] global budget exhausted (elapsed=${elapsed}s / ${TIMEOUT_S}s), skipping scene=${scene}"
      printf '%s,%d,%s,%d,%d,%d,%.2f,%d,%s\n' \
        "${scene}" 0 "skipped" 0 0 0 0 0 "" >> "${proc_csv}"
      continue
    fi

    local scene_t_start scene_t_end
    scene_t_start=$(date +%s)
    local scene_log="${proc_logs_root}/${scene}.log"
    local scene_video="${proc_video_root}/${scene}"
    local scene_tb="${proc_tb_root}/${scene}"
    mkdir -p "${scene_video}" "${scene_tb}"

    local attempt=0
    local exit_final=255
    while [ "${attempt}" -lt "${MAX_NATIVE_RETRIES}" ]; do
      attempt=$((attempt + 1))
      now=$(date +%s)
      elapsed=$((now - proc_t_start))
      if [ "$((TIMEOUT_S - elapsed))" -le 60 ]; then
        echo "[${tag}/${scene}] no time left for attempt ${attempt}, breaking"
        break
      fi
      echo "[${tag}/${scene}] attempt ${attempt}/${MAX_NATIVE_RETRIES}  (proc_elapsed=${elapsed}s)"

      # Clamp per-scene budget to whatever's left of the proc budget.
      local scene_budget="${PER_SCENE_TIMEOUT_S}"
      if [ "$((TIMEOUT_S - elapsed))" -lt "${scene_budget}" ]; then
        scene_budget=$((TIMEOUT_S - elapsed - 30))
        echo "[${tag}/${scene}] clamping scene budget to ${scene_budget}s (proc budget tight)"
      fi

      # Inner script env per attempt.  Append to scene_log so retries
      # accumulate evidence; the parser tolerates multiple "Success rate"
      # banners and just takes the last one.
      (
        export CUDA_VISIBLE_DEVICES="${cuda_dev}"
        export VLFM_POINTNAV_GPU_ID=1
        # Inherit VLFM_SKIP_SUCCESS_VIDEOS from outer env (default 1 above).
        export CONTENT_SCENES="${scene}"
        export N_EPISODES="${EP_PER_SCENE}"
        export SPLIT="${SPLIT}"
        export TIMEOUT_S="${scene_budget}"
        export VIDEO_DIR="${scene_video}"
        export TB_DIR="${scene_tb}"
        export LOG="${scene_log}.attempt${attempt}.raw"
        bash "${INNER_SCRIPT}"
      )
      exit_final=$?
      echo "[${tag}/${scene}] attempt ${attempt} exit=${exit_final}"
      # Concat the raw inner log into the scene log for one-stop grep.
      {
        echo "===== attempt ${attempt} exit=${exit_final} ====="
        cat "${scene_log}.attempt${attempt}.raw" 2>/dev/null || true
      } >> "${scene_log}"

      # 0 = clean,  124 = inner timeout (script did its job, no point retry).
      if [ "${exit_final}" -eq 0 ] || [ "${exit_final}" -eq 124 ]; then
        break
      fi
      echo "[${tag}/${scene}] native-side failure (${exit_final}), retrying after 5s..."
      sleep 5
    done

    scene_t_end=$(date +%s)
    local scene_wall=$((scene_t_end - scene_t_start))

    # Parse per-scene metrics from the accumulated scene_log.
    # n_started: count of "Step: 0 | Mode: initialize" lines (one per ep).
    # success/failure: the most recent "Success rate: X.XX% (Y out of Z)"
    #                  line gives Y=success count, Z=total finished.
    # n_failure derived as Z - Y; failure videos confirm it.
    local n_started n_total n_success n_failure rate_pct fail_videos
    n_started=$(grep -cE '^Step: 0 \| Mode: initialize' "${scene_log}" 2>/dev/null || echo 0)
    # Last "Success rate" banner of the scene log:
    local last_rate
    last_rate=$(grep -oE 'Success rate: [0-9.]+% \([0-9]+ out of [0-9]+\)' "${scene_log}" 2>/dev/null | tail -n 1)
    if [ -n "${last_rate}" ]; then
      rate_pct=$(echo "${last_rate}"  | sed -nE 's/Success rate: ([0-9.]+)% .*/\1/p')
      n_success=$(echo "${last_rate}" | sed -nE 's/.*\(([0-9]+) out of ([0-9]+)\)/\1/p')
      n_total=$(echo "${last_rate}"   | sed -nE 's/.*\(([0-9]+) out of ([0-9]+)\)/\2/p')
    else
      rate_pct="0.00"
      n_success=0
      n_total=0
    fi
    n_failure=$(( n_total - n_success ))
    [ "${n_failure}" -lt 0 ] && n_failure=0

    # Failure-only videos -> failure_montage/<tag>__<scene>__<orig>.mp4
    local copied=0
    if [ -d "${scene_video}" ]; then
      while IFS= read -r -d '' f; do
        local base
        base=$(basename "${f}")
        cp -f "${f}" "${ROOT_OUT}/failure_montage/${tag}__${scene}__${base}"
        copied=$((copied + 1))
      done < <(find "${scene_video}" -maxdepth 1 -type f -name '*.mp4' -print0)
    fi
    fail_videos="${copied}"

    printf '%s,%d,%d,%d,%d,%d,%s,%d,%d\n' \
      "${scene}" "${attempt}" "${exit_final}" "${n_started}" \
      "${n_success}" "${n_failure}" "${rate_pct}" "${scene_wall}" "${fail_videos}" \
      >> "${proc_csv}"

    echo "[${tag}/${scene}] DONE  attempts=${attempt}  exit=${exit_final}  started=${n_started}  ok=${n_success}  fail=${n_failure}  rate=${rate_pct}%  wall=${scene_wall}s  fail_videos=${fail_videos}"

    # Track worst exit at proc level (124 < native-crash codes; rank
    # native crashes above clean and above timeout, retain whatever
    # final attempt produced).  Simple: keep max if > current.
    if [ "${exit_final}" -gt "${worst_exit}" ]; then
      worst_exit="${exit_final}"
    fi
  done

  local proc_t_end
  proc_t_end=$(date +%s)
  local proc_wall=$((proc_t_end - proc_t_start))
  echo "[${tag}] all scenes done  wall=${proc_wall}s  worst_exit=${worst_exit}"

  # Final proc exit file (the orchestrator's view of severity).
  echo "${worst_exit}" > "${ROOT_OUT}/${tag}.exit"
}

T_START=$(date +%s)

# In re-run mode, split the requested episode IDs evenly across the two procs.
# Each proc gets half the IDs, no CONTENT_SCENES restriction, one call to the
# inner script (rather than per-scene loop) with EPISODE_IDS set.
if [ -n "${RERUN_EPISODE_IDS}" ]; then
  # Convert to array, split in half.
  IFS=',' read -ra _ALL_IDS <<<"${RERUN_EPISODE_IDS}"
  unset IFS
  _HALF=$(( (${#_ALL_IDS[@]} + 1) / 2 ))
  _IDS_A="$(printf '%s,' "${_ALL_IDS[@]:0:${_HALF}}" | sed 's/,$//')"
  _IDS_B="$(printf '%s,' "${_ALL_IDS[@]:${_HALF}}" | sed 's/,$//')"
  # Fallback: if one half is empty (odd count), give all to A.
  [ -z "${_IDS_B}" ] && _IDS_B="${_IDS_A}"

  _rerun_one_proc() {
    local tag="$1" cuda_dev="$2" ep_ids="$3"
    local proc_log="${ROOT_OUT}/${tag}.log"
    local scene_video="${ROOT_OUT}/${tag}_video/rerun"
    local scene_tb="${ROOT_OUT}/${tag}_tb/rerun"
    local raw_log="${ROOT_OUT}/${tag}_logs/rerun.raw"
    mkdir -p "${scene_video}" "${scene_tb}" "$(dirname "${raw_log}")"
    exec >"${proc_log}" 2>&1
    echo "[${tag}] RERUN mode  gpus=${cuda_dev}  ids=${ep_ids}"
    local attempt=0 exit_final=255
    local n_ids
    n_ids="$(echo "${ep_ids}" | tr ',' '\n' | wc -l)"
    while [ "${attempt}" -lt "${MAX_NATIVE_RETRIES}" ]; do
      attempt=$((attempt + 1))
      echo "[${tag}] attempt ${attempt}/${MAX_NATIVE_RETRIES}"
      (
        export CUDA_VISIBLE_DEVICES="${cuda_dev}"
        export VLFM_POINTNAV_GPU_ID=1
        export VLFM_SKIP_SUCCESS_VIDEOS=0
        export EPISODE_IDS="${ep_ids}"
        export N_EPISODES="${n_ids}"
        export SPLIT="${SPLIT}"
        export TIMEOUT_S="${TIMEOUT_S}"
        export VIDEO_DIR="${scene_video}"
        export TB_DIR="${scene_tb}"
        export LOG="${raw_log}.attempt${attempt}"
        bash "${INNER_SCRIPT}"
      )
      exit_final=$?
      echo "[${tag}] attempt ${attempt} exit=${exit_final}"
      if [ "${exit_final}" -eq 0 ] || [ "${exit_final}" -eq 124 ]; then break; fi
      echo "[${tag}] native crash, retrying in 5s..."
      sleep 5
    done
    echo "${exit_final}" > "${ROOT_OUT}/${tag}.exit"
    # Copy videos to montage (all videos, not just failures, in rerun mode).
    while IFS= read -r -d '' f; do
      local base; base=$(basename "${f}")
      cp -f "${f}" "${ROOT_OUT}/failure_montage/${tag}__rerun__${base}"
    done < <(find "${scene_video}" -maxdepth 1 -type f -name '*.mp4' -print0)
  }

  _rerun_one_proc procA "${PROC_A_GPUS}" "${_IDS_A}" &
  PID_A=$!
  sleep 5
  [ "${_IDS_B}" != "${_IDS_A}" ] && {
    _rerun_one_proc procB "${PROC_B_GPUS}" "${_IDS_B}" &
    PID_B=$!
  } || PID_B="${PID_A}"

else
  # Normal scene-rotation mode.
  run_one_proc procA "${PROC_A_GPUS}" "${SCENES_A}" &
  PID_A=$!
  if [ "${SINGLE_PROC}" = "1" ]; then
    echo "single-proc fallback: procB skipped (SINGLE_PROC=1)"
    PID_B="${PID_A}"
  else
    sleep 5
    run_one_proc procB "${PROC_B_GPUS}" "${SCENES_B}" &
    PID_B=$!
  fi
fi

echo "launched: procA pid=${PID_A}  procB pid=${PID_B}"
echo "to follow:"
echo "  tail -f ${ROOT_OUT}/procA.log ${ROOT_OUT}/procB.log"
echo "  tail -f ${ROOT_OUT}/procA_logs/<scene>.log"

wait "${PID_A}" "${PID_B}"
T_END=$(date +%s)
WALL=$((T_END - T_START))

EXIT_A=$(cat "${ROOT_OUT}/procA.exit" 2>/dev/null || echo "?")
EXIT_B=$(cat "${ROOT_OUT}/procB.exit" 2>/dev/null || echo "?")

# Combined summary CSV.
SUMMARY_CSV="${ROOT_OUT}/summary.csv"
echo "proc,${SCENE_CSV_HEADER}" > "${SUMMARY_CSV}"
for tag in procA procB; do
  if [ -f "${ROOT_OUT}/${tag}_scenes.csv" ]; then
    tail -n +2 "${ROOT_OUT}/${tag}_scenes.csv" | awk -v t="${tag}" '{print t "," $0}' >> "${SUMMARY_CSV}"
  fi
done

# Generate episodes.csv and summary.json.
# Write Python to a temp file so there are no heredoc-within-heredoc issues.
_PY="${ROOT_OUT}/.gen_summary.py"
cat >"${_PY}" <<'PYEOF'
"""Parse video filenames -> episodes.csv; aggregate -> summary.json."""
import csv, json, os, re, glob, sys

ROOT = sys.argv[1]

EP_FIELDS = [
    "proc", "scene", "episode_id",
    "success", "spl", "soft_spl",
    "distance_to_goal", "distance_to_goal_reward",
    "traveled_stairs", "target_detected", "stop_called",
    "yaw", "start_yaw", "video_file",
]

_kv_re = re.compile(r'([a-z_]+)=(-?[0-9]+(?:\.[0-9]+)?)')

def parse_mp4(path, proc, scene):
    stem = os.path.basename(path)
    if stem.endswith('.mp4'): stem = stem[:-4]
    kv = dict(_kv_re.findall(stem))
    return {
        "proc": proc, "scene": scene,
        "episode_id":            kv.get("episode", ""),
        "success":               kv.get("success", ""),
        "spl":                   kv.get("spl", ""),
        "soft_spl":              kv.get("soft_spl", ""),
        "distance_to_goal":      kv.get("distance_to_goal", ""),
        "distance_to_goal_reward": kv.get("distance_to_goal_reward", ""),
        "traveled_stairs":       kv.get("traveled_stairs", ""),
        "target_detected":       kv.get("target_detected", ""),
        "stop_called":           kv.get("stop_called", ""),
        "yaw":                   kv.get("yaw", ""),
        "start_yaw":             kv.get("start_yaw", ""),
        "video_file":            path,
    }

episode_rows = []
for proc in ("procA", "procB"):
    vid_root = os.path.join(ROOT, f"{proc}_video")
    if not os.path.isdir(vid_root):
        continue
    for mp4 in sorted(glob.glob(os.path.join(vid_root, "**", "*.mp4"), recursive=True)):
        rel = os.path.relpath(mp4, vid_root)
        parts = rel.split(os.sep)
        scene = parts[0] if len(parts) > 1 else "unknown"
        episode_rows.append(parse_mp4(mp4, proc, scene))

ep_csv = os.path.join(ROOT, "episodes.csv")
with open(ep_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=EP_FIELDS)
    w.writeheader()
    w.writerows(episode_rows)

wall_s = 0
ws_file = os.path.join(ROOT, ".wall_s")
if os.path.exists(ws_file):
    try: wall_s = int(open(ws_file).read().strip())
    except: pass

out = {
    "run_id": os.path.basename(ROOT),
    "wall_s": wall_s,
    "exit": {},
    "procs": {},
    "totals": {"n_started": 0, "n_success": 0, "n_failure": 0, "fail_videos": 0},
    "episodes_csv": ep_csv,
    "failure_episode_ids": sorted(
        set(r["episode_id"] for r in episode_rows if r["episode_id"]),
        key=lambda x: int(x) if x.isdigit() else x,
    ),
}

for tag in ("procA", "procB"):
    ef = os.path.join(ROOT, f"{tag}.exit")
    out["exit"][tag] = open(ef).read().strip() if os.path.exists(ef) else "?"
    csv_path = os.path.join(ROOT, f"{tag}_scenes.csv")
    scenes, agg = [], {"n_started": 0, "n_success": 0, "n_failure": 0,
                       "fail_videos": 0, "wall_s": 0}
    if os.path.exists(csv_path):
        with open(csv_path) as fh:
            for row in csv.DictReader(fh):
                for k in ("n_started", "n_success", "n_failure", "wall_s", "fail_videos"):
                    row[k] = int(row.get(k, 0) or 0)
                row["success_rate_pct"] = float(row.get("success_rate_pct", 0) or 0)
                scenes.append(row)
                for k in agg:
                    if k in row: agg[k] += row[k]
    agg["success_rate_pct"] = round(
        (agg["n_success"] / agg["n_started"] * 100.0) if agg["n_started"] else 0.0, 2)
    out["procs"][tag] = {"scenes": scenes, "aggregate": agg}
    for k in ("n_started", "n_success", "n_failure", "fail_videos"):
        out["totals"][k] += agg[k]

n_s = out["totals"]["n_started"]
out["totals"]["success_rate_pct"] = round(
    (out["totals"]["n_success"] / n_s * 100.0) if n_s else 0.0, 2)
out["totals"]["fail_videos_in_montage"] = len(
    glob.glob(os.path.join(ROOT, "failure_montage", "*.mp4")))
out["totals"]["failure_episode_count"] = len(out["failure_episode_ids"])
print(json.dumps(out, indent=2, ensure_ascii=False))
PYEOF
echo "${WALL}" > "${ROOT_OUT}/.wall_s"
python3 "${_PY}" "${ROOT_OUT}" >"${ROOT_OUT}/summary.json" 2>"${ROOT_OUT}/summary.json.err" || true
rm -f "${_PY}"

# Human-readable summary banner.
{
  echo "================================================================"
  echo "wall total: ${WALL}s = $((WALL / 60))min $((WALL % 60))s"
  echo "procA exit=${EXIT_A}    procB exit=${EXIT_B}    (0=clean, 124=timeout, 134/139=native crash)"
  echo "================================================================"
  for tag in procA procB; do
    echo "-- ${tag} per-scene CSV --"
    column -t -s, "${ROOT_OUT}/${tag}_scenes.csv" 2>/dev/null || cat "${ROOT_OUT}/${tag}_scenes.csv"
    echo
  done
  echo "-- failure_montage/ --"
  ls "${ROOT_OUT}/failure_montage/" 2>/dev/null | wc -l | awk '{print "  total failure mp4 files = " $1}'
  ls -lh "${ROOT_OUT}/failure_montage/" 2>/dev/null | head -10
  echo
  echo "-- combined episode count from per-scene CSVs --"
  if [ -f "${SUMMARY_CSV}" ]; then
    python3 - <<PYEOF
import csv
tot_s = tot_ok = tot_fail = 0
with open("${SUMMARY_CSV}") as f:
    for r in csv.DictReader(f):
        tot_s    += int(r.get("n_started", 0) or 0)
        tot_ok   += int(r.get("n_success", 0) or 0)
        tot_fail += int(r.get("n_failure", 0) or 0)
rate = (tot_ok / tot_s * 100.0) if tot_s else 0.0
print(f"  total ep started: {tot_s}")
print(f"  total ep success: {tot_ok}")
print(f"  total ep failure: {tot_fail}")
print(f"  overall success rate: {rate:.2f}%")
PYEOF
  fi
  echo "DONE_PARALLEL out=${ROOT_OUT}"
} | tee "${ROOT_OUT}/summary.txt"

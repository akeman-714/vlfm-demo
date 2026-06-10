const form = document.getElementById("requestForm");
const input = document.getElementById("requestText");
const runModeInput = document.getElementById("runMode");
const apiKeyInput = document.getElementById("apiKey");
const button = document.getElementById("runButton");
const runLine = document.getElementById("runLine");
const stagePill = document.getElementById("stagePill");
const labelPill = document.getElementById("labelPill");
const elapsed = document.getElementById("elapsed");
const liveFrame = document.getElementById("liveFrame");
const liveBevFrame = document.getElementById("liveBevFrame");
const frameEmpty = document.getElementById("frameEmpty");
const bevFrameEmpty = document.getElementById("bevFrameEmpty");
const statusText = document.getElementById("statusText");
const semanticText = document.getElementById("semanticText");
const videoText = document.getElementById("videoText");
const logText = document.getElementById("logText");
const videoSection = document.getElementById("videoSection");
const resultVideo = document.getElementById("resultVideo");
const videoLink = document.getElementById("videoLink");

let currentRunId = null;
let pollTimer = null;

runModeInput.value = localStorage.getItem("semanticCatRunMode") || "semantic_query";
apiKeyInput.value = localStorage.getItem("semanticCatApiKey") || "";

const defaultTextByMode = {
  semantic_query: "咪咪在哪里",
  find_cat: "cat_demo",
  global_home_40: "return_home_after_40",
  global_home_100: "return_home_after_100",
  object_memory_cat: "object_memory_cat",
  persistent_memory_cat_pair: "persistent_memory_cat_pair",
};

function syncModeFields() {
  const needsApiKey = runModeInput.value === "semantic_query";
  apiKeyInput.classList.toggle("hidden", !needsApiKey);
  input.value = defaultTextByMode[runModeInput.value] || "";
}

syncModeFields();

function setBusy(isBusy) {
  button.disabled = isBusy;
  input.disabled = isBusy;
  runModeInput.disabled = isBusy;
  apiKeyInput.disabled = isBusy;
}

function setFrame(img, empty, url, token) {
  if (!url) return;
  const tok = token || "";
  // Only refetch when the frame actually changed. Polling (0.8s) is faster than
  // generation (~1.5s), so without this guard the same frame is re-downloaded ~2x.
  if (img.dataset.frameToken === tok && img.getAttribute("src")) return;
  img.dataset.frameToken = tok;
  img.src = `${url}?v=${encodeURIComponent(tok || Date.now())}`;
  img.classList.add("visible");
  empty.classList.add("hidden");
}

function setVideo(url, label = "saved") {
  if (!url) return;
  videoSection.classList.remove("hidden");
  resultVideo.src = url;
  videoLink.href = url;
  videoText.textContent = label;
}

function renderRun(run) {
  runLine.textContent = run.run_id ? `run ${run.run_id}` : "idle";
  stagePill.textContent = run.stage || "ready";
  labelPill.textContent = `mode: ${run.label || "-"}`;
  elapsed.textContent = `${run.elapsed_sec || 0}s`;
  statusText.textContent = run.error ? `${run.status}: ${run.error}` : run.status;
  semanticText.textContent = run.reason
    ? `${run.reason}`
    : "-";
  logText.textContent = run.log_tail || "";
  logText.scrollTop = logText.scrollHeight;

  if (run.latest_frame_url) setFrame(liveFrame, frameEmpty, run.latest_frame_url, run.latest_frame_token);
  if (run.latest_bev_frame_url) setFrame(liveBevFrame, bevFrameEmpty, run.latest_bev_frame_url, run.latest_bev_frame_token);
  if (run.live_video_url) {
    setVideo(run.live_video_url, run.live_video_path || "live rgb saved");
  } else if (run.video_url) {
    setVideo(run.video_url, run.video_path || "composite saved");
  }

  document.body.dataset.status = run.status || "idle";

  if (run.status === "complete" || run.status === "error") {
    setBusy(false);
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function poll() {
  if (!currentRunId) return;
  const response = await fetch(`/api/runs/${currentRunId}`, { cache: "no-store" });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "status request failed");
  renderRun(data);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  const runMode = runModeInput.value;
  const apiKey = apiKeyInput.value.trim();
  localStorage.setItem("semanticCatRunMode", runMode);
  if (apiKey) localStorage.setItem("semanticCatApiKey", apiKey);

  setBusy(true);
  videoSection.classList.add("hidden");
  resultVideo.removeAttribute("src");
  liveFrame.removeAttribute("src");
  liveFrame.classList.remove("visible");
  delete liveFrame.dataset.frameToken;
  liveBevFrame.removeAttribute("src");
  liveBevFrame.classList.remove("visible");
  delete liveBevFrame.dataset.frameToken;
  frameEmpty.classList.remove("hidden");
  bevFrameEmpty.classList.remove("hidden");
  frameEmpty.textContent = "starting";
  bevFrameEmpty.textContent = "starting";
  logText.textContent = "";
  videoText.textContent = "-";

  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, run_mode: runMode, api_key: apiKey }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "run request failed");
    currentRunId = data.run_id;
    renderRun(data);
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => {
      poll().catch((error) => {
        statusText.textContent = error.message;
        setBusy(false);
      });
    }, 800);
  } catch (error) {
    statusText.textContent = error.message;
    stagePill.textContent = "error";
    setBusy(false);
  }
});

runModeInput.addEventListener("change", syncModeFields);

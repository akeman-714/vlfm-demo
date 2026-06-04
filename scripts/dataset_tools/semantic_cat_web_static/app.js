const form = document.getElementById("requestForm");
const input = document.getElementById("requestText");
const apiKeyInput = document.getElementById("apiKey");
const button = document.getElementById("runButton");
const runLine = document.getElementById("runLine");
const stagePill = document.getElementById("stagePill");
const labelPill = document.getElementById("labelPill");
const elapsed = document.getElementById("elapsed");
const liveFrame = document.getElementById("liveFrame");
const frameEmpty = document.getElementById("frameEmpty");
const statusText = document.getElementById("statusText");
const semanticText = document.getElementById("semanticText");
const videoText = document.getElementById("videoText");
const logText = document.getElementById("logText");
const videoSection = document.getElementById("videoSection");
const resultVideo = document.getElementById("resultVideo");
const videoLink = document.getElementById("videoLink");

let currentRunId = null;
let pollTimer = null;

apiKeyInput.value = localStorage.getItem("semanticCatApiKey") || "";

function setBusy(isBusy) {
  button.disabled = isBusy;
  input.disabled = isBusy;
  apiKeyInput.disabled = isBusy;
}

function setFrame(url) {
  if (!url) return;
  liveFrame.src = `${url}?t=${Date.now()}`;
  liveFrame.classList.add("visible");
  frameEmpty.classList.add("hidden");
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
  labelPill.textContent = `label: ${run.label || "-"}`;
  elapsed.textContent = `${run.elapsed_sec || 0}s`;
  statusText.textContent = run.error ? `${run.status}: ${run.error}` : run.status;
  semanticText.textContent = run.label
    ? `${run.label} (${Number(run.confidence || 0).toFixed(2)})`
    : "-";
  logText.textContent = run.log_tail || "";
  logText.scrollTop = logText.scrollHeight;

  if (run.latest_frame_url) setFrame(run.latest_frame_url);
  if (run.live_video_url) {
    setVideo(run.live_video_url, "live rgb saved");
  } else if (run.video_url) {
    setVideo(run.video_url, "composite saved");
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
  const apiKey = apiKeyInput.value.trim();
  if (!text) return;
  if (!apiKey) {
    statusText.textContent = "enter API key";
    stagePill.textContent = "ready";
    return;
  }
  if (apiKey) localStorage.setItem("semanticCatApiKey", apiKey);

  setBusy(true);
  videoSection.classList.add("hidden");
  resultVideo.removeAttribute("src");
  liveFrame.removeAttribute("src");
  liveFrame.classList.remove("visible");
  frameEmpty.classList.remove("hidden");
  frameEmpty.textContent = "starting";
  logText.textContent = "";
  videoText.textContent = "-";

  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, api_key: apiKey }),
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

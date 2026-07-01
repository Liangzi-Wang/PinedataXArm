const state = {
  datasets: [],
  preferredDataset: null,
  selectedDataset: null,
  summary: null,
  selectedDatasetStamp: null,
  selectedEpisodeId: null,
  selectedStream: null,
  currentFrame: 0,
  isPlaying: false,
  playTimer: null,
  autoRefreshTimer: null,
  livePreviewTimer: null,
  autoRefreshTick: 0,
  refreshInFlight: false,
  frameRequestToken: 0,
  frameWatchdogTimer: null,
  lastFrameRequestKey: "",
  frameRetryCount: 0,
  recordingDefaults: null,
  recordingStatus: null,
  recordingCommandBusy: false,
  recordingBackend: "embedded",
  livePreviewSupported: true,
  deletingEpisode: false,
};

const FRAME_LOAD_TIMEOUT_MS = 2500;
const MAX_FRAME_RETRIES = 2;

const els = {};

window.addEventListener("DOMContentLoaded", async () => {
  cacheElements();
  bindEvents();
  await loadConfig();
  await refreshDatasets();
  await refreshRecordingStatus();
  startAutoRefresh();
});

function cacheElements() {
  els.configDataDir = document.getElementById("configDataDir");
  els.refreshDatasetsBtn = document.getElementById("refreshDatasetsBtn");
  els.deleteSelectedEpisodeBtn = document.getElementById("deleteSelectedEpisodeBtn");
  els.autoRefreshCheckbox = document.getElementById("autoRefreshCheckbox");
  els.datasetSelect = document.getElementById("datasetSelect");
  els.datasetEmpty = document.getElementById("datasetEmpty");
  els.episodesTableBody = document.getElementById("episodesTableBody");
  els.episodeTitle = document.getElementById("episodeTitle");
  els.episodeMetaBadge = document.getElementById("episodeMetaBadge");
  els.streamSelect = document.getElementById("streamSelect");
  els.prevFrameBtn = document.getElementById("prevFrameBtn");
  els.playPauseBtn = document.getElementById("playPauseBtn");
  els.nextFrameBtn = document.getElementById("nextFrameBtn");
  els.frameImage = document.getElementById("frameImage");
  els.frameSlider = document.getElementById("frameSlider");
  els.frameCounter = document.getElementById("frameCounter");
  els.frameTimestamp = document.getElementById("frameTimestamp");
  els.durationValue = document.getElementById("durationValue");
  els.fpsValue = document.getElementById("fpsValue");
  els.startTimeValue = document.getElementById("startTimeValue");
  els.episodeInfoField = document.getElementById("episodeInfoField");
  els.inventoryTableBody = document.getElementById("inventoryTableBody");
  els.recorderStatusBadge = document.getElementById("recorderStatusBadge");
  els.recorderMessage = document.getElementById("recorderMessage");
  els.initializeSpacemouseBtn = document.getElementById("initializeSpacemouseBtn");
  els.resetRecorderBtn = document.getElementById("resetRecorderBtn");
  els.reconnectRtdeBtn = document.getElementById("reconnectRtdeBtn");
  els.startRecordingBtn = document.getElementById("startRecordingBtn");
  els.labelSubtaskBtn = document.getElementById("labelSubtaskBtn");
  els.stopRecordingBtn = document.getElementById("stopRecordingBtn");
  els.deleteEpisodeBtn = document.getElementById("deleteEpisodeBtn");
  els.saveShutdownRecorderBtn = document.getElementById("saveShutdownRecorderBtn");
  els.shutdownRecorderBtn = document.getElementById("shutdownRecorderBtn");
  els.handFrameCount = document.getElementById("handFrameCount");
  els.wristFrameCount = document.getElementById("wristFrameCount");
  els.externalFrameCount = document.getElementById("externalFrameCount");
  els.robotFrameCount = document.getElementById("robotFrameCount");
  els.subtaskLabelCount = document.getElementById("subtaskLabelCount");
  els.instructionInput = document.getElementById("instructionInput");
  els.recordRootInput = document.getElementById("recordRootInput");
  els.fpsInput = document.getElementById("fpsInput");
  els.handWidthInput = document.getElementById("handWidthInput");
  els.handHeightInput = document.getElementById("handHeightInput");
  els.wristWidthInput = document.getElementById("wristWidthInput");
  els.wristHeightInput = document.getElementById("wristHeightInput");
  els.externalWidthInput = document.getElementById("externalWidthInput");
  els.externalHeightInput = document.getElementById("externalHeightInput");
  els.handSerialInput = document.getElementById("handSerialInput");
  els.wristSerialInput = document.getElementById("wristSerialInput");
  els.externalSerialInput = document.getElementById("externalSerialInput");
  els.handProductIdsInput = document.getElementById("handProductIdsInput");
  els.wristProductIdsInput = document.getElementById("wristProductIdsInput");
  els.externalProductIdsInput = document.getElementById("externalProductIdsInput");
  els.cameraStartRetriesInput = document.getElementById("cameraStartRetriesInput");
  els.cameraStartRetryDelayInput = document.getElementById("cameraStartRetryDelayInput");
  els.cameraPostResetWaitInput = document.getElementById("cameraPostResetWaitInput");
  els.robotIpInput = document.getElementById("robotIpInput");
  els.robotFpsInput = document.getElementById("robotFpsInput");
  els.gripperPortInput = document.getElementById("gripperPortInput");
  els.subtaskSegmentIndexInput = document.getElementById("subtaskSegmentIndexInput");
  els.subtaskResetNoiseInput = document.getElementById("subtaskResetNoiseInput");
  els.cameraBusyResetCheckbox = document.getElementById("cameraBusyResetCheckbox");
  els.handSourceBadge = document.getElementById("handSourceBadge");
  els.wristSourceBadge = document.getElementById("wristSourceBadge");
  els.externalSourceBadge = document.getElementById("externalSourceBadge");
  els.robotSourceBadge = document.getElementById("robotSourceBadge");
  els.gripperSourceBadge = document.getElementById("gripperSourceBadge");
  els.liveHandRgb = document.getElementById("liveHandRgb");
  els.liveWristRgb = document.getElementById("liveWristRgb");
  els.liveExternalRgb = document.getElementById("liveExternalRgb");
  els.robotLivePreview = document.getElementById("robotLivePreview");
  els.gripperLivePreview = document.getElementById("gripperLivePreview");
  els.spacemouseLivePreview = document.getElementById("spacemouseLivePreview");
}

function bindEvents() {
  els.refreshDatasetsBtn.addEventListener("click", () => refreshDatasets(false));
  els.deleteSelectedEpisodeBtn.addEventListener("click", runDeleteSelectedEpisode);
  els.autoRefreshCheckbox.addEventListener("change", startAutoRefresh);
  els.datasetSelect.addEventListener("change", async (event) => {
    state.selectedDataset = event.target.value || null;
    state.selectedEpisodeId = null;
    await loadSummary({ resetPlayback: true });
  });
  els.streamSelect.addEventListener("change", (event) => {
    state.selectedStream = event.target.value || null;
    stopPlayback();
    updateFrame();
  });
  els.frameSlider.addEventListener("input", (event) => {
    state.currentFrame = Number(event.target.value);
    stopPlayback();
    updateFrame();
  });
  els.playPauseBtn.addEventListener("click", togglePlayback);
  els.prevFrameBtn.addEventListener("click", () => stepFrame(-1));
  els.nextFrameBtn.addEventListener("click", () => stepFrame(1));
  els.initializeSpacemouseBtn.addEventListener("click", () => runRecordingInitialize("spacemouse"));
  els.resetRecorderBtn.addEventListener("click", () => runRecordingCommand("reset"));
  els.reconnectRtdeBtn.addEventListener("click", () => runRecordingCommand("stop-rtde-motion"));
  els.startRecordingBtn.addEventListener("click", () => runRecordingCommand("start"));
  els.labelSubtaskBtn.addEventListener("click", () => runRecordingCommand("label"));
  els.stopRecordingBtn.addEventListener("click", () => runRecordingCommand("stop"));
  els.deleteEpisodeBtn.addEventListener("click", () => runRecordingCommand("delete"));
  els.saveShutdownRecorderBtn.addEventListener("click", () => runRecordingCommand("shutdown-save"));
  els.shutdownRecorderBtn.addEventListener("click", () => runRecordingCommand("shutdown"));
  document.addEventListener("keydown", handleRecorderHotkeys);
}

function isEditableTarget(target) {
  if (!target) {
    return false;
  }
  const tagName = target.tagName ? target.tagName.toLowerCase() : "";
  return tagName === "input" || tagName === "textarea" || tagName === "select" || target.isContentEditable;
}

function clickIfEnabled(button) {
  if (!button || button.disabled) {
    return;
  }
  button.click();
}

function hasActiveTextSelection() {
  const selection = window.getSelection ? window.getSelection() : null;
  if (!selection || selection.isCollapsed) {
    return false;
  }
  return Boolean(selection.toString().trim());
}

function setTextContentWhenSafe(element, text) {
  if (!element) {
    return;
  }
  if (hasActiveTextSelection()) {
    element.dataset.pendingText = text;
    return;
  }
  if (element.textContent !== text) {
    element.textContent = text;
  }
  delete element.dataset.pendingText;
}

function flushDeferredTextUpdates() {
  if (hasActiveTextSelection()) {
    return;
  }
  for (const element of [els.recorderMessage, els.spacemouseLivePreview]) {
    if (!element) {
      continue;
    }
    const pendingText = element.dataset.pendingText;
    if (typeof pendingText === "string") {
      element.textContent = pendingText;
      delete element.dataset.pendingText;
    }
  }
}

function handleRecorderHotkeys(event) {
  if (event.defaultPrevented || event.repeat) {
    return;
  }
  if (isEditableTarget(event.target)) {
    return;
  }

  const key = String(event.key || "").toLowerCase();
  const hotkeyMap = {
    i: els.initializeSpacemouseBtn,
    r: els.resetRecorderBtn,
    u: els.reconnectRtdeBtn,
    c: els.startRecordingBtn,
    l: els.labelSubtaskBtn,
    s: els.stopRecordingBtn,
    d: els.deleteEpisodeBtn,
    x: els.saveShutdownRecorderBtn,
    q: els.shutdownRecorderBtn,
  };
  const button = hotkeyMap[key];
  if (!button) {
    return;
  }

  if (key === "r") {
    if (!event.ctrlKey || event.metaKey || event.altKey) {
      return;
    }
  } else if (event.ctrlKey || event.metaKey || event.altKey) {
    return;
  }

  event.preventDefault();
  clickIfEnabled(button);
}

async function apiJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const message = await readErrorMessage(response);
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return response.json();
}

async function apiPostJson(url, body = null) {
  const response = await fetch(url, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    const message = await readErrorMessage(response);
    throw new Error(message || `Request failed: ${response.status}`);
  }
  return response.json();
}

async function runRecordingInitialize(pipeline) {
  try {
    setRecordingMessage(`Running initialize (${pipeline})...`);
    const payload = await apiPostJson("/api/recording/initialize", collectRecordingForm(pipeline));
    renderRecordingStatus(payload);
  } catch (error) {
    console.error(error);
    setRecordingMessage(error.message || `Recorder command failed: initialize (${pipeline})`);
  }
}

async function readErrorMessage(response) {
  const text = await response.text();
  try {
    const payload = JSON.parse(text);
    return payload.detail || text;
  } catch (_error) {
    return text;
  }
}

async function loadConfig() {
  const payload = await apiJson("/api/config");
  els.configDataDir.textContent = payload.data_dir;
  state.preferredDataset = payload.default_dataset || null;
  state.recordingDefaults = payload.recording_defaults || null;
  state.recordingBackend = payload.recording_backend || "embedded";
  state.livePreviewSupported = payload.live_preview_supported !== false;
  if (state.recordingDefaults) {
    populateRecordingForm(state.recordingDefaults);
  }
}

async function refreshDatasets(preserveSelection = true) {
  if (state.refreshInFlight) {
    return;
  }

  state.refreshInFlight = true;
  try {
    const payload = await apiJson("/api/datasets");
    state.datasets = payload.datasets;
    const previousSelection = state.selectedDataset;
    renderDatasetOptions(preserveSelection);

    const currentMeta = state.datasets.find((item) => item.name === state.selectedDataset) || null;
    const nextStamp = currentMeta ? `${currentMeta.updated_at_ns}:${currentMeta.size_bytes}` : null;
    const selectionChanged = previousSelection !== state.selectedDataset;
    const summaryStale = state.selectedDatasetStamp !== nextStamp;

    if (state.selectedDataset && (selectionChanged || summaryStale || !state.summary)) {
      await loadSummary({ resetPlayback: selectionChanged });
    } else if (!state.selectedDataset) {
      clearSummary();
    } else {
      renderEpisodesTable();
    }
  } catch (error) {
    console.error(error);
    clearSummary("Could not read datasets from the watched directory.");
  } finally {
    state.refreshInFlight = false;
  }
}

function renderDatasetOptions(preserveSelection) {
  const previous = preserveSelection ? state.selectedDataset : null;
  const names = state.datasets.map((item) => item.name);
  const nextSelection =
    (previous && names.includes(previous) && previous) ||
    (state.preferredDataset && names.includes(state.preferredDataset) && state.preferredDataset) ||
    names[0] ||
    null;

  els.datasetSelect.innerHTML = "";
  for (const dataset of state.datasets) {
    const option = document.createElement("option");
    option.value = dataset.name;
    option.textContent = dataset.name;
    els.datasetSelect.appendChild(option);
  }

  state.selectedDataset = nextSelection;
  els.datasetSelect.value = nextSelection || "";
  els.datasetEmpty.classList.toggle("hidden", state.datasets.length > 0);
}

async function loadSummary({ resetPlayback = true } = {}) {
  if (!state.selectedDataset) {
    clearSummary();
    return;
  }

  if (resetPlayback) {
    stopPlayback();
  }

  const params = new URLSearchParams({ dataset: state.selectedDataset });
  const payload = await apiJson(`/api/summary?${params.toString()}`);
  state.summary = payload;
  state.selectedDatasetStamp = `${payload.updated_at_ns}:${payload.file_size_bytes}`;

  const episodeIds = payload.episodes.map((item) => item.id);
  if (!state.selectedEpisodeId || !episodeIds.includes(state.selectedEpisodeId)) {
    state.selectedEpisodeId = episodeIds[0] || null;
  }

  renderEpisodesTable();
  renderSelectedEpisode();
}

function clearSummary(message = "Choose a dataset to get started.") {
  state.summary = null;
  state.selectedDatasetStamp = null;
  state.selectedEpisodeId = null;
  state.selectedStream = null;
  state.currentFrame = 0;
  stopPlayback();

  els.episodesTableBody.innerHTML = `<tr><td colspan="7">${message}</td></tr>`;
  els.episodeTitle.textContent = "Choose an episode";
  els.episodeMetaBadge.textContent = "No episode selected";
  els.streamSelect.innerHTML = "";
  els.frameImage.removeAttribute("src");
  els.frameCounter.textContent = "step 0 / 0";
  els.frameTimestamp.textContent = "timestamp unavailable";
  els.durationValue.textContent = "-";
  els.fpsValue.textContent = "-";
  els.startTimeValue.textContent = "-";
  els.episodeInfoField.textContent = "Select an episode to inspect details.";
  els.inventoryTableBody.innerHTML = "<tr><td colspan=\"4\">No episode loaded.</td></tr>";
  updateDeleteSelectedEpisodeButtonState();
}

function renderEpisodesTable() {
  const episodes = state.summary?.episodes || [];
  if (!episodes.length) {
    els.episodesTableBody.innerHTML = "<tr><td colspan=\"7\">No recording episodes detected in this recordings folder.</td></tr>";
    updateDeleteSelectedEpisodeButtonState();
    return;
  }

  els.episodesTableBody.innerHTML = "";
  for (const episode of episodes) {
    const row = document.createElement("tr");
    if (episode.id === state.selectedEpisodeId) {
      row.classList.add("is-selected");
    }
    row.innerHTML = `
      <td><span class="episode-row-label">${episode.label}</span></td>
      <td>${episode.steps}</td>
      <td>${formatSeconds(episode.duration_s)}</td>
      <td>${episode.fps_estimate ? episode.fps_estimate.toFixed(2) : "-"}</td>
      <td>${formatBytes(episode.episode_size_bytes)}</td>
      <td>${episode.stream_count}</td>
      <td>${formatDateTime(episode.start_time_utc)}</td>
    `;
    row.addEventListener("click", () => {
      stopPlayback();
      state.selectedEpisodeId = episode.id;
      state.currentFrame = 0;
      renderEpisodesTable();
      renderSelectedEpisode();
    });
    els.episodesTableBody.appendChild(row);
  }
  updateDeleteSelectedEpisodeButtonState();
}

function renderSelectedEpisode() {
  const episode = getSelectedEpisode();
  if (!episode) {
    stopPlayback();
    els.episodeTitle.textContent = "Choose an episode";
    els.episodeMetaBadge.textContent = "No episode selected";
    els.streamSelect.innerHTML = "";
    els.frameImage.removeAttribute("src");
    els.frameCounter.textContent = "step 0 / 0";
    els.frameTimestamp.textContent = "timestamp unavailable";
    els.durationValue.textContent = "-";
    els.fpsValue.textContent = "-";
    els.startTimeValue.textContent = "-";
    els.episodeInfoField.textContent = "Select an episode to inspect details.";
    els.inventoryTableBody.innerHTML = "<tr><td colspan=\"4\">No episode loaded.</td></tr>";
    updateDeleteSelectedEpisodeButtonState();
    return;
  }

  els.episodeTitle.textContent = episode.label;
  els.episodeMetaBadge.textContent = `${episode.steps} steps - ${formatSeconds(episode.duration_s)}`;
  els.durationValue.textContent = formatSeconds(episode.duration_s);
  els.fpsValue.textContent = episode.fps_estimate ? `${episode.fps_estimate.toFixed(2)} Hz` : "-";
  els.startTimeValue.textContent = formatDateTime(episode.start_time_utc);
  renderEpisodeInfoField(episode);

  renderInventory(episode.datasets);
  renderStreamOptions(episode.streams);

  if (state.currentFrame >= episode.steps) {
    state.currentFrame = 0;
  }

  els.frameSlider.max = Math.max(episode.steps - 1, 0);
  els.frameSlider.value = state.currentFrame;
  updateFrame();
  updateDeleteSelectedEpisodeButtonState();
}

function renderStreamOptions(streams) {
  els.streamSelect.innerHTML = "";

  if (!streams.length) {
    state.selectedStream = null;
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No visual streams available";
    els.streamSelect.appendChild(option);
    return;
  }

  const available = streams.map((stream) => stream.path);
  if (!state.selectedStream || !available.includes(state.selectedStream)) {
    state.selectedStream = streams[0].path;
  }

  for (const stream of streams) {
    const option = document.createElement("option");
    option.value = stream.path;
    option.textContent = `${stream.display_path || stream.path} (${stream.kind}, ${stream.width}x${stream.height})`;
    els.streamSelect.appendChild(option);
  }
  els.streamSelect.value = state.selectedStream;
}

function renderInventory(datasets) {
  if (!datasets.length) {
    els.inventoryTableBody.innerHTML = "<tr><td colspan=\"4\">No datasets found.</td></tr>";
    return;
  }

  els.inventoryTableBody.innerHTML = "";
  for (const item of datasets) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${item.path}</td>
      <td>${JSON.stringify(item.shape)}</td>
      <td>${item.dtype}</td>
      <td>${formatBytes(item.size_bytes)}</td>
    `;
    els.inventoryTableBody.appendChild(row);
  }
}

function renderEpisodeInfoField(episode) {
  if (!els.episodeInfoField) {
    return;
  }
  const infoData = {
    episode_metadata: episode.group_attrs || {},
    folder_info: episode.file_attrs || {},
  };
  const rootList = document.createElement("ul");
  rootList.className = "info-list";
  appendInfoNodes(rootList, infoData);
  els.episodeInfoField.innerHTML = "";
  els.episodeInfoField.appendChild(rootList);
}

function appendInfoNodes(container, value) {
  if (value === null || value === undefined) {
    const li = document.createElement("li");
    li.className = "info-item";
    li.textContent = "null";
    container.appendChild(li);
    return;
  }

  if (Array.isArray(value)) {
    if (!value.length) {
      const li = document.createElement("li");
      li.className = "info-item";
      li.textContent = "[]";
      container.appendChild(li);
      return;
    }
    value.forEach((item, index) => {
      const li = document.createElement("li");
      li.className = "info-item";
      const key = document.createElement("span");
      key.className = "info-key";
      key.textContent = `[${index}]`;
      li.appendChild(key);
      if (isPrimitive(item)) {
        const text = document.createElement("span");
        text.className = "info-value";
        text.textContent = ` ${formatPrimitive(item)}`;
        li.appendChild(text);
      } else {
        const nested = document.createElement("ul");
        nested.className = "info-list nested";
        appendInfoNodes(nested, item);
        li.appendChild(nested);
      }
      container.appendChild(li);
    });
    return;
  }

  if (typeof value === "object") {
    const entries = Object.entries(value);
    if (!entries.length) {
      const li = document.createElement("li");
      li.className = "info-item";
      li.textContent = "{}";
      container.appendChild(li);
      return;
    }
    entries.forEach(([keyName, item]) => {
      const li = document.createElement("li");
      li.className = "info-item";
      const key = document.createElement("span");
      key.className = "info-key";
      key.textContent = keyName;
      li.appendChild(key);
      if (isPrimitive(item)) {
        const text = document.createElement("span");
        text.className = "info-value";
        text.textContent = `: ${formatPrimitive(item)}`;
        li.appendChild(text);
      } else {
        const nested = document.createElement("ul");
        nested.className = "info-list nested";
        appendInfoNodes(nested, item);
        li.appendChild(nested);
      }
      container.appendChild(li);
    });
    return;
  }

  const li = document.createElement("li");
  li.className = "info-item";
  li.textContent = formatPrimitive(value);
  container.appendChild(li);
}

function isPrimitive(value) {
  return value === null || value === undefined || ["string", "number", "boolean"].includes(typeof value);
}

function formatPrimitive(value) {
  if (typeof value === "string") {
    return value;
  }
  if (value === null) {
    return "null";
  }
  if (value === undefined) {
    return "undefined";
  }
  return String(value);
}

function updateFrame() {
  const episode = getSelectedEpisode();
  clearFrameWatchdog();
  if (!episode) {
    return;
  }

  els.frameSlider.value = state.currentFrame;
  els.frameCounter.textContent = `step ${state.currentFrame} / ${Math.max(episode.steps - 1, 0)}`;
  const approxSeconds = episode.steps > 1 ? (state.currentFrame / (episode.steps - 1)) * episode.duration_s : 0;
  els.frameTimestamp.textContent = `~ ${approxSeconds.toFixed(2)} s from episode start`;

  if (!state.selectedStream) {
    els.frameImage.removeAttribute("src");
    els.frameImage.alt = "No visual stream available";
    state.lastFrameRequestKey = "";
    state.frameRetryCount = 0;
    return;
  }

  const streamInfo = getSelectedStreamInfo();
  const streamWidth = Number(streamInfo?.width || 0);
  const maxWidth = streamWidth > 0 ? Math.min(streamWidth, 4096) : 1920;
  const frameRequestKey = [
    state.selectedDataset || "",
    episode.id || "",
    state.selectedStream || "",
    String(state.currentFrame),
    String(maxWidth),
  ].join("|");
  if (frameRequestKey !== state.lastFrameRequestKey) {
    state.lastFrameRequestKey = frameRequestKey;
    state.frameRetryCount = 0;
  }
  const params = new URLSearchParams({
    dataset: state.selectedDataset,
    episode: episode.id,
    stream: state.selectedStream,
    frame_index: String(state.currentFrame),
    max_width: String(maxWidth),
  });
  if (state.frameRetryCount > 0) {
    params.set("t", String(Date.now()));
  }

  const requestToken = ++state.frameRequestToken;
  const requestStartedAt = performance.now();
  els.frameImage.onload = () => {
    clearFrameWatchdog();
    if (requestToken !== state.frameRequestToken || !state.isPlaying) {
      return;
    }
    state.frameRetryCount = 0;
    scheduleNextPlayback(performance.now() - requestStartedAt);
  };
  els.frameImage.onerror = () => {
    handleFrameLoadFailure(requestToken, "error");
  };
  state.frameWatchdogTimer = window.setTimeout(() => {
    handleFrameLoadFailure(requestToken, "timeout");
  }, FRAME_LOAD_TIMEOUT_MS);
  els.frameImage.src = `/api/frame?${params.toString()}`;
  els.frameImage.alt = `${episode.label} - ${state.selectedStream} - frame ${state.currentFrame}`;
}

function clearFrameWatchdog() {
  if (state.frameWatchdogTimer) {
    window.clearTimeout(state.frameWatchdogTimer);
    state.frameWatchdogTimer = null;
  }
}

function handleFrameLoadFailure(requestToken, reason) {
  clearFrameWatchdog();
  if (requestToken !== state.frameRequestToken) {
    return;
  }

  const episode = getSelectedEpisode();
  if (!episode) {
    stopPlayback();
    return;
  }

  if (!state.isPlaying) {
    return;
  }

  if (state.frameRetryCount < MAX_FRAME_RETRIES) {
    state.frameRetryCount += 1;
    updateFrame();
    return;
  }

  console.warn(`Replay frame ${state.currentFrame} failed after retries (${reason}); skipping.`);
  state.frameRetryCount = 0;
  if (state.currentFrame >= episode.steps - 1) {
    stopPlayback();
    return;
  }
  state.currentFrame += 1;
  updateFrame();
}

async function runRecordingCommand(command) {
  if (state.recordingCommandBusy) {
    return;
  }
  state.recordingCommandBusy = true;
  applyRecorderButtonState();
  try {
    setRecordingMessage(`Running ${command}...`);
    let payload;
    if (command === "start") {
      payload = await apiPostJson("/api/recording/start", {
        instruction: els.instructionInput.value.trim() || "untitled",
      });
    } else if (command === "reset") {
      payload = await apiPostJson("/api/recording/reset", collectRecordingResetForm());
    } else {
      payload = await apiPostJson(`/api/recording/${command}`);
    }
    renderRecordingStatus(payload);
  } catch (error) {
    console.error(error);
    setRecordingMessage(error.message || `Recorder command failed: ${command}`);
  } finally {
    state.recordingCommandBusy = false;
    applyRecorderButtonState();
  }
}

function updateDeleteSelectedEpisodeButtonState() {
  const episode = getSelectedEpisode();
  const canDelete = Boolean(episode && !state.deletingEpisode);
  if (els.deleteSelectedEpisodeBtn) {
    els.deleteSelectedEpisodeBtn.disabled = !canDelete;
  }
}

async function runDeleteSelectedEpisode() {
  const episode = getSelectedEpisode();
  if (!episode || !state.selectedDataset || state.deletingEpisode) {
    return;
  }

  state.deletingEpisode = true;
  updateDeleteSelectedEpisodeButtonState();
  try {
    stopPlayback();
    await apiPostJson("/api/episode/delete", {
      dataset: state.selectedDataset,
      episode: episode.id,
    });
    state.selectedEpisodeId = null;
    state.currentFrame = 0;
    await refreshDatasets(false);
  } catch (error) {
    console.error(error);
  } finally {
    state.deletingEpisode = false;
    updateDeleteSelectedEpisodeButtonState();
  }
}

async function refreshRecordingStatus() {
  try {
    const payload = await apiJson("/api/recording/status");
    renderRecordingStatus(payload);
  } catch (error) {
    console.error(error);
    setRecordingMessage("Could not refresh recorder status.");
  }
}

function populateRecordingForm(config) {
  els.recordRootInput.value = config.root || "";
  els.instructionInput.value = config.instruction || "untitled";
  els.fpsInput.value = config.fps ?? 15;
  els.handWidthInput.value = config.hand_width ?? 640;
  els.handHeightInput.value = config.hand_height ?? 480;
  els.wristWidthInput.value = config.wrist_width ?? 640;
  els.wristHeightInput.value = config.wrist_height ?? 480;
  els.externalWidthInput.value = config.external_width ?? 848;
  els.externalHeightInput.value = config.external_height ?? 480;
  els.handSerialInput.value = config.hand_serial || "";
  els.wristSerialInput.value = config.wrist_serial || "";
  els.externalSerialInput.value = config.external_serial || "";
  els.handProductIdsInput.value = config.hand_product_ids || "0B5B";
  els.wristProductIdsInput.value = config.wrist_product_ids || "0B5B";
  els.externalProductIdsInput.value = config.external_product_ids || "0B5B";
  els.cameraStartRetriesInput.value = config.camera_start_retries ?? 20;
  els.cameraStartRetryDelayInput.value = config.camera_start_retry_delay ?? 0.5;
  els.cameraPostResetWaitInput.value = config.camera_post_reset_wait ?? 2.0;
  els.robotIpInput.value = config.robot_ip || "";
  els.robotFpsInput.value = config.robot_fps ?? 200;
  els.gripperPortInput.value = config.gripper_port ?? 63352;
  els.subtaskSegmentIndexInput.value = config.subtask_segment_index ?? 0;
  els.subtaskResetNoiseInput.value = config.subtask_reset_noise_xyz_m ?? 0.01;
  els.cameraBusyResetCheckbox.checked = Boolean(config.camera_busy_reset);
}

function collectRecordingForm(pipeline = "spacemouse") {
  return {
    pipeline,
    root: els.recordRootInput.value.trim(),
    instruction: els.instructionInput.value.trim() || "untitled",
    hand_serial: els.handSerialInput.value.trim(),
    wrist_serial: els.wristSerialInput.value.trim(),
    external_serial: els.externalSerialInput.value.trim(),
    allow_missing_hand: false,
    allow_missing_wrist: false,
    allow_missing_external: false,
    hand_product_ids: els.handProductIdsInput.value.trim(),
    wrist_product_ids: els.wristProductIdsInput.value.trim(),
    external_product_ids: els.externalProductIdsInput.value.trim(),
    fps: Number(els.fpsInput.value || 15),
    hand_width: Number(els.handWidthInput.value || 640),
    hand_height: Number(els.handHeightInput.value || 480),
    wrist_width: Number(els.wristWidthInput.value || 640),
    wrist_height: Number(els.wristHeightInput.value || 480),
    external_width: Number(els.externalWidthInput.value || 848),
    external_height: Number(els.externalHeightInput.value || 480),
    camera_start_retries: Number(els.cameraStartRetriesInput.value || 20),
    camera_start_retry_delay: Number(els.cameraStartRetryDelayInput.value || 0.5),
    camera_busy_reset: els.cameraBusyResetCheckbox.checked,
    camera_post_reset_wait: Number(els.cameraPostResetWaitInput.value || 2.0),
    robot_ip: els.robotIpInput.value.trim(),
    robot_fps: Number(els.robotFpsInput.value || 200),
    enable_gripper_state: true,
    gripper_port: Number(els.gripperPortInput.value || 63352),
    allow_missing_robot: false,
    allow_missing_gripper: true,
    subtask_segment_index: Number(els.subtaskSegmentIndexInput.value || 0),
    subtask_reset_noise_xyz_m: Number(els.subtaskResetNoiseInput.value || 0.01),
  };
}

function collectRecordingResetForm() {
  return {
    subtask_segment_index: Number(els.subtaskSegmentIndexInput.value || 0),
    subtask_reset_noise_xyz_m: Number(els.subtaskResetNoiseInput.value || 0.01),
  };
}

function renderRecordingStatus(payload) {
  state.recordingStatus = payload;
  if (payload.backend) {
    state.recordingBackend = payload.backend;
  }
  if (typeof payload.live_preview_supported === "boolean") {
    state.livePreviewSupported = payload.live_preview_supported;
  }
  const initialized = Boolean(payload.initialized);
  const sessionInitialized = Boolean(payload.session_initialized ?? payload.initialized);
  const recording = Boolean(payload.recording);

  els.recorderStatusBadge.textContent = recording ? "Recording" : initialized ? "Ready" : sessionInitialized ? "Tmux running" : "Not initialized";
  els.handFrameCount.textContent = payload.counts?.hand_frames ?? 0;
  els.wristFrameCount.textContent = payload.counts?.wrist_frames ?? 0;
  els.externalFrameCount.textContent = payload.counts?.external_frames ?? 0;
  els.robotFrameCount.textContent = payload.counts?.robot_frames ?? 0;
  els.subtaskLabelCount.textContent = payload.counts?.subtask_labels ?? 0;

  const saveLine = payload.data_folder ? `Save path: ${payload.data_folder}` : "";
  const episodeLine = payload.current_episode ? `Episode: ${payload.current_episode}` : "";
  const errorLine = payload.last_error ? `Last error: ${payload.last_error}` : "";
  const backendLine = state.recordingBackend ? `Backend: ${state.recordingBackend}` : "";
  const pipelineLine = payload.pipeline ? `Pipeline: ${payload.pipeline}` : "";
  const tmuxLine = formatTmuxStatus(payload.tmux);
  const subtaskResetLine = formatSubtaskReset(payload.subtask_reset);
  setTextContentWhenSafe(
    els.recorderMessage,
    [payload.message, pipelineLine, backendLine, subtaskResetLine, saveLine, episodeLine, errorLine, tmuxLine]
      .filter(Boolean)
      .join("\n"),
  );

  applyRecorderButtonState();

  renderSourceStatuses(payload);
  updateLiveImages(payload);
  renderLiveSignals(payload);
  renderSpacemousePreview(payload);
}

function applyRecorderButtonState() {
  const payload = state.recordingStatus || {};
  const initialized = Boolean(payload.initialized);
  const sessionInitialized = Boolean(payload.session_initialized ?? payload.initialized);
  const recording = Boolean(payload.recording);
  const saving = Boolean(payload.saving);
  const resetSupported = Boolean(payload.spacemouse?.requested);
  const busy = Boolean(state.recordingCommandBusy);

  els.startRecordingBtn.disabled = busy || !initialized || recording || saving;
  els.resetRecorderBtn.disabled = busy || !sessionInitialized || saving || !resetSupported;
  els.reconnectRtdeBtn.disabled = busy || !sessionInitialized || saving || !resetSupported;
  els.labelSubtaskBtn.disabled = busy || !initialized || !recording || saving;
  els.stopRecordingBtn.disabled = busy || !initialized || !recording || saving;
  els.deleteEpisodeBtn.disabled = busy || !initialized || recording || saving;
  els.saveShutdownRecorderBtn.disabled = busy || !sessionInitialized || saving;
  els.shutdownRecorderBtn.disabled = busy || !sessionInitialized || saving;
}

function renderSourceStatuses(payload) {
  setSourceCard(els.handSourceBadge, payload.hand_camera);
  setSourceCard(els.wristSourceBadge, payload.wrist_camera);
  setSourceCard(els.externalSourceBadge, payload.external_camera);
  setSourceCard(els.robotSourceBadge, payload.robot);
  setSourceCard(els.gripperSourceBadge, payload.gripper);
}

function setSourceCard(badgeEl, source) {
  const stateValue = inferSourceState(source);
  badgeEl.dataset.state = stateValue;
  badgeEl.textContent = sourceStateLabel(stateValue);
}

function inferSourceState(source) {
  if (!source) {
    return "disconnected";
  }
  if (source.available) {
    return "streaming";
  }
  if (source.connected || source.enabled) {
    return "connected";
  }
  return "disconnected";
}

function sourceStateLabel(stateValue) {
  if (stateValue === "streaming") {
    return "Streaming";
  }
  if (stateValue === "connected") {
    return "Connected";
  }
  return "Disconnected";
}

function formatSubtaskReset(subtaskReset) {
  if (!subtaskReset) {
    return "";
  }
  const index = Number(subtaskReset.active_segment_index ?? 0);
  const noise = Number(subtaskReset.noise_xyz_m ?? 0);
  const configured = Array.isArray(subtaskReset.configured_segments) ? subtaskReset.configured_segments.join(", ") : "";
  const configuredText = configured ? `; configured: ${configured}` : "";
  const labels = subtaskReset.labels && typeof subtaskReset.labels === "object" ? subtaskReset.labels : {};
  const modeLabel = {
    1: "Skill 1 - grasp RAM",
    2: "Skill 2 - insert RAM",
    3: "Skill 3 - insert CPU",
    4: "Skill 4 - insert GPU",
    5: "Skill 5 - insert DISK",
  }[index] || labels[String(index)] || `segment ${index}`;
  if (index > 0) {
    return `Collection mode: ${modeLabel}, XYZ noise +/-${noise.toFixed(3)} m${configuredText}`;
  }
  return `Collection mode: Full task - normal reset${configuredText}`;
}

function formatTmuxStatus(tmux) {
  if (!tmux || !Array.isArray(tmux.panes) || !tmux.panes.length) {
    return "";
  }

  const sections = [`Tmux session: ${tmux.session || "unknown"}`];
  for (const pane of tmux.panes) {
    const label = pane.label || `pane ${pane.index}`;
    const command = pane.command ? `: ${pane.command}` : "";
    const lines = Array.isArray(pane.lines) ? pane.lines.filter(Boolean) : [];
    sections.push(`[${label}${command}]`);
    sections.push(lines.length ? lines.join("\n") : "(no recent output)");
  }
  return sections.join("\n");
}

function updateLiveImages(payload) {
  if (!state.livePreviewSupported) {
    clearLiveImages();
    return;
  }

  const stamp = Date.now();
  setLiveImage(els.liveExternalRgb, payload.external_camera, "external", "rgb", stamp);
  setLiveImage(els.liveWristRgb, payload.wrist_camera, "wrist", "rgb", stamp);
  setLiveImage(els.liveHandRgb, payload.hand_camera, "hand", "rgb", stamp);
}

function setLiveImage(element, source, camera, kind, stamp) {
  if (!source?.available) {
    element.removeAttribute("src");
    return;
  }
  element.onerror = () => {
    element.removeAttribute("src");
  };
  const sourceWidth = Number(source.width || 0);
  const maxWidth = sourceWidth > 0 ? Math.min(sourceWidth, 4096) : 640;
  const params = new URLSearchParams({
    camera,
    kind,
    max_width: String(maxWidth),
    t: String(stamp),
  });
  element.src = `/api/recording/frame?${params.toString()}`;
}

function clearLiveImages() {
  for (const element of [els.liveHandRgb, els.liveWristRgb, els.liveExternalRgb]) {
    element.removeAttribute("src");
  }
}

function renderLiveSignals(payload) {
  setTextContentWhenSafe(els.robotLivePreview, formatRobotPreview(payload.robot));
  setTextContentWhenSafe(els.gripperLivePreview, formatGripperPreview(payload.gripper));
}

function renderSpacemousePreview(payload) {
  const spacemouse = payload.spacemouse || {};
  const motion = formatNumericList(spacemouse.motion_state, 6);
  const buttons = spacemouse.buttons || {};
  const lines = [
    `Status: ${sourceStateLabel(inferSourceState(spacemouse))}`,
    `Motion: ${motion || "-"}`,
    `Buttons: left ${buttons.left ? "pressed" : "released"}, right ${buttons.right ? "pressed" : "released"}`,
  ];
  if (!spacemouse.connected && spacemouse.text) {
    lines.push(spacemouse.text);
  }
  setTextContentWhenSafe(els.spacemouseLivePreview, lines.join("\n"));
}

function formatRobotPreview(robot) {
  if (!robot || !(robot.connected || robot.enabled)) {
    return "Robot disconnected.";
  }
  const latest = robot.latest || {};
  const jointStateLimit = Array.isArray(latest.joint_state) ? latest.joint_state.length : 6;
  const jointState = formatNumericList(latest.joint_state, jointStateLimit);
  const tcpPose = formatNumericList(latest.eef_pose, 6);
  const tcpWrench = formatNumericList(latest.tcp_wrench, 6);
  const lines = [
    `Status: ${sourceStateLabel(inferSourceState(robot))}`,
    `Joint q: ${jointState || "-"}`,
    `TCP pose: ${tcpPose || "-"}`,
    `TCP wrench: ${tcpWrench || "-"}`,
  ];
  return lines.join("\n");
}

function formatGripperPreview(gripper) {
  if (!gripper || !(gripper.connected || gripper.enabled)) {
    return "Gripper disconnected.";
  }
  const position = formatNumericValue(gripper.latest_position);
  return [
    `Status: ${sourceStateLabel(inferSourceState(gripper))}`,
    `Position: ${position ?? "-"}`,
  ].join("\n");
}

function formatNumericList(values, limit = 6) {
  if (!Array.isArray(values) || !values.length) {
    return "";
  }
  return values
    .slice(0, limit)
    .map((value) => formatNumericValue(value) ?? "-")
    .join(", ");
}

function formatNumericValue(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  return value.toFixed(2);
}

function stopLivePreviewRefresh() {
  if (state.livePreviewTimer) {
    window.clearInterval(state.livePreviewTimer);
    state.livePreviewTimer = null;
  }
}

function startLivePreviewRefresh() {
  stopLivePreviewRefresh();
  if (!els.autoRefreshCheckbox.checked) {
    return;
  }
  state.livePreviewTimer = window.setInterval(() => {
    if (!state.recordingStatus) {
      return;
    }
    updateLiveImages(state.recordingStatus);
  }, 120);
}

function setRecordingMessage(message) {
  setTextContentWhenSafe(els.recorderMessage, message);
}

function togglePlayback() {
  if (state.isPlaying) {
    stopPlayback();
    return;
  }

  const episode = getSelectedEpisode();
  if (!episode || !state.selectedStream) {
    return;
  }

  if (state.currentFrame >= episode.steps - 1) {
    state.currentFrame = 0;
  }

  state.isPlaying = true;
  els.playPauseBtn.textContent = "Pause";
  scheduleNextPlayback(0);
}

function stopPlayback() {
  state.isPlaying = false;
  if (state.playTimer) {
    window.clearTimeout(state.playTimer);
    state.playTimer = null;
  }
  clearFrameWatchdog();
  state.frameRetryCount = 0;
  state.lastFrameRequestKey = "";
  els.playPauseBtn.textContent = "Play";
}

function stepFrame(delta) {
  const episode = getSelectedEpisode();
  if (!episode) {
    return;
  }
  stopPlayback();
  state.currentFrame = clamp(state.currentFrame + delta, 0, Math.max(episode.steps - 1, 0));
  updateFrame();
}

function getSelectedEpisode() {
  return state.summary?.episodes?.find((item) => item.id === state.selectedEpisodeId) || null;
}

function getSelectedStreamInfo() {
  const episode = getSelectedEpisode();
  return episode?.streams?.find((item) => item.path === state.selectedStream) || null;
}

function startAutoRefresh() {
  if (state.autoRefreshTimer) {
    window.clearInterval(state.autoRefreshTimer);
    state.autoRefreshTimer = null;
  }
  stopLivePreviewRefresh();
  if (!els.autoRefreshCheckbox.checked) {
    return;
  }
  startLivePreviewRefresh();
  state.autoRefreshTick = 0;
  state.autoRefreshTimer = window.setInterval(() => {
    if (hasActiveTextSelection()) {
      return;
    }
    flushDeferredTextUpdates();
    state.autoRefreshTick += 1;
    refreshRecordingStatus();
    if (!state.isPlaying && state.autoRefreshTick % 5 === 0) {
      refreshDatasets(true);
    }
  }, 1000);
}

function scheduleNextPlayback(lastFrameLoadMs) {
  const episode = getSelectedEpisode();
  if (!state.isPlaying || !episode) {
    return;
  }
  if (state.currentFrame >= episode.steps - 1) {
    stopPlayback();
    return;
  }

  const fps = Math.max(1, Math.min(30, Math.round(episode.fps_estimate || 8)));
  const frameIntervalMs = Math.round(1000 / fps);
  const delayMs = Math.max(0, frameIntervalMs - Math.round(lastFrameLoadMs));

  if (state.playTimer) {
    window.clearTimeout(state.playTimer);
  }
  state.playTimer = window.setTimeout(() => {
    if (!state.isPlaying) {
      return;
    }
    state.currentFrame += 1;
    updateFrame();
  }, delayMs);
}

function formatJson(value) {
  return JSON.stringify(value || {}, null, 2);
}

function formatSeconds(value) {
  return `${Number(value || 0).toFixed(2)} s`;
}

function formatDateTime(value) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "-";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const precision = size >= 100 || unitIndex === 0 ? 0 : size >= 10 ? 1 : 2;
  return `${size.toFixed(precision)} ${units[unitIndex]}`;
}


function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

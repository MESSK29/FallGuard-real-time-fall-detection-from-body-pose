/**
 * FallGuard - Live Monitor Engine (Enhanced with WebSocket Streamer)
 * Powered by @mediapipe/tasks-vision (Pose Landmarker, Full/Lite model, runningMode: VIDEO)
 *
 * Privacy Invariant:
 * Zero raw camera frames or pixel data are ever stored or uploaded.
 * Only mathematical 33-landmark coordinate arrays are communicated to the local backend.
 */

import {
  PoseLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ============================================================================
// Model & Pipeline Configuration Constants (User Configurable)
// ============================================================================

export const MODEL_VARIANT = "full"; // Options: "lite" | "full"

export const MODEL_ASSET_PATHS = {
  lite: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
  full: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
};

const WASM_CDN_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

export const MIN_POSE_DETECTION_CONFIDENCE = 0.5;
export const MIN_POSE_PRESENCE_CONFIDENCE = 0.5;
export const MIN_TRACKING_CONFIDENCE = 0.5;
export const MIN_VISIBILITY_THRESHOLD = 0.5;
export const SMOOTHING_ALPHA = 0.5;
export const MAX_LOST_FRAMES = 5;

// Throttling for WebSocket Landmark Telemetry (approx. 15 messages per second)
const WS_SEND_INTERVAL_MS = 66;
const WS_RECONNECT_INTERVAL_MS = 2000;

export const POSE_CONNECTIONS = [
  // Torso & Shoulders
  [11, 12], [11, 23], [12, 24], [23, 24],
  // Left Arm
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
  // Right Arm
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22],
  // Left Leg
  [23, 25], [25, 27], [27, 29], [27, 31],
  // Right Leg
  [24, 26], [26, 28], [28, 30], [28, 32],
  // Head & Face
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10]
];

// ============================================================================
// Internal Module State
// ============================================================================

let poseLandmarker = null;
let activeDelegate = "CPU"; // "GPU" or "CPU"
let videoStream = null;
let animationFrameId = null;
let lastVideoTime = -1;
let isMonitoring = false;
let audioContext = null;

// Landmarks tracking
let rawLandmarks = null;       // Unsmoothed landmarks directly from MediaPipe
let smoothedLandmarks = null;  // EMA smoothed landmarks for jitter-free visual overlay
let lostFramesCount = 0;       // Consecutive frames with no detection
let currentConfidence = 0;     // Average visibility score [0..1]

// Telemetry & Performance
let currentFps = 0;
let frameCount = 0;
let lastFpsUpdateTime = performance.now();
let prevHipY = null;
let prevFrameTime = null;

// WebSocket State
let socket = null;
let reconnectTimer = null;
let serverWakeupTimer = null;
let lastWsSendTime = 0;

// Settings State
const SETTINGS_KEY = "fallguard_settings";
let currentSettings = {
  chatId: "",
  sensitivity: 3,
  lyingDelay: 2.0,
  sirenEnabled: true
};

// DOM References
let videoEl = null;
let canvasEl = null;
let canvasCtx = null;
let btnStart = null;
let btnStop = null;
let btnToggleSiren = null;
let sirenToggleLabel = null;
let btnTestAlert = null;
let btnDismissAlert = null;
let soundIndicator = null;
let fallAlertBanner = null;
let bannerTimestamp = null;
let bannerIdTag = null;
let dismissToast = null;
let dismissToastMessage = null;
let backendErrorBanner = null;
let backendErrorText = null;

let fpsBadge = null;
let confidenceBadge = null;
let connectionBadge = null;
let statusBadge = null;
let noPersonWarning = null;
let loaderOverlay = null;
let loaderText = null;
let standbyState = null;
let originalStandbyHtml = "";
let delegateTag = null;
let modelVariantTag = null;

// Settings DOM Elements
let settingChatId = null;
let settingSensitivity = null;
let settingSensitivityLabel = null;
let settingLyingDelay = null;
let settingLyingDelayLabel = null;
let settingSirenToggle = null;
let btnResetSettings = null;
let btnToggleSettings = null;
let settingsHeader = null;
let settingsChevron = null;
let settingsBody = null;
let serverWakeupBanner = null;

// Kinematic Preview Readouts
let readoutTorsoAngle = null;
let readoutHipHeight = null;
let readoutHipSpeed = null;
let readoutAspectRatio = null;

// Emergency Fall Alert & Siren State
let currentActiveAlertId = null;
let sirenOsc = null;
let sirenGain = null;
let sirenSweepInterval = null;
let sirenIsPlaying = false;
let sirenEnabled = true;

// ============================================================================
// Initialization & Event Binding
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
  initDOMElements();
  loadStoredSettings();
  applySettingsToInputs();
  bindEvents();
  updateModelInfoLabels();
  updateSoundIndicator();
  initWebSocket();
});

function getClientId() {
  if (typeof window.getFallGuardClientId === "function") {
    return window.getFallGuardClientId();
  }
  let id = localStorage.getItem("fallguard_client_id");
  if (!id) {
    if (typeof crypto !== "undefined" && crypto.randomUUID) {
      id = crypto.randomUUID();
    } else {
      id = "client-" + Math.random().toString(36).substring(2, 10);
    }
    localStorage.setItem("fallguard_client_id", id);
  }
  return id;
}

function loadStoredSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object") {
        if (typeof parsed.chatId === "string") currentSettings.chatId = parsed.chatId.trim();
        if (parsed.sensitivity !== undefined) currentSettings.sensitivity = parseInt(parsed.sensitivity, 10) || 3;
        if (parsed.lyingDelay !== undefined) currentSettings.lyingDelay = parseFloat(parsed.lyingDelay) || 2.0;
        if (parsed.sirenEnabled !== undefined) currentSettings.sirenEnabled = Boolean(parsed.sirenEnabled);
      }
    }
  } catch (e) {
    console.warn("Could not load stored settings:", e);
  }
  sirenEnabled = currentSettings.sirenEnabled;
}

function saveCurrentSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(currentSettings));
  } catch (e) {
    console.warn("Could not save settings to localStorage:", e);
  }
}

function applySettingsToInputs() {
  if (settingChatId) settingChatId.value = currentSettings.chatId || "";
  if (settingSensitivity) settingSensitivity.value = currentSettings.sensitivity;
  if (settingLyingDelay) settingLyingDelay.value = currentSettings.lyingDelay;
  if (settingSirenToggle) settingSirenToggle.checked = currentSettings.sirenEnabled;
  updateSensitivityDisplay();
  updateLyingDelayDisplay();
}

function updateSensitivityDisplay(serverThreshold) {
  if (!settingSensitivityLabel) return;
  const sens = currentSettings.sensitivity;
  let level = "Medium";
  if (sens <= 1) level = "Low";
  else if (sens === 2) level = "Med-Low";
  else if (sens === 3) level = "Medium (Default)";
  else if (sens === 4) level = "Med-High";
  else if (sens >= 5) level = "High";

  let thresh = serverThreshold;
  if (thresh === undefined) {
    thresh = Math.max(0.30, Math.min(0.95, 0.65 + (3 - sens) * 0.10));
  }
  settingSensitivityLabel.textContent = `${level} (Thr: ${Number(thresh).toFixed(2)})`;
}

function updateLyingDelayDisplay() {
  if (!settingLyingDelayLabel) return;
  settingLyingDelayLabel.textContent = `${parseFloat(currentSettings.lyingDelay).toFixed(1)}s`;
}

function sendSettingsOverWs() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  const payload = {
    type: "settings",
    telegram_chat_id: currentSettings.chatId ? currentSettings.chatId.trim() : "",
    sensitivity: currentSettings.sensitivity,
    lying_seconds: currentSettings.lyingDelay
  };
  try {
    socket.send(JSON.stringify(payload));
    console.log("[FallGuard] Sent settings over WebSocket:", payload);
  } catch (err) {
    console.warn("[FallGuard] Failed to send settings over WebSocket:", err);
  }
}

function initDOMElements() {
  videoEl = document.getElementById("webcam-video");
  canvasEl = document.getElementById("output-canvas");
  if (canvasEl) {
    canvasCtx = canvasEl.getContext("2d");
  }

  btnStart = document.getElementById("btn-start-monitor");
  btnStop = document.getElementById("btn-stop-monitor");
  btnToggleSiren = document.getElementById("btn-toggle-siren");
  sirenToggleLabel = document.getElementById("siren-toggle-label");
  btnTestAlert = document.getElementById("btn-test-alert");
  btnDismissAlert = document.getElementById("btn-dismiss-alert");
  soundIndicator = document.getElementById("sound-indicator");

  fallAlertBanner = document.getElementById("fall-alert-banner");
  bannerTimestamp = document.getElementById("banner-timestamp");
  bannerIdTag = document.getElementById("banner-id-tag");
  dismissToast = document.getElementById("dismiss-toast");
  dismissToastMessage = document.getElementById("dismiss-toast-message");
  backendErrorBanner = document.getElementById("backend-error-banner");
  backendErrorText = document.getElementById("backend-error-text");

  fpsBadge = document.getElementById("fps-badge");
  confidenceBadge = document.getElementById("confidence-badge");
  connectionBadge = document.getElementById("connection-badge");
  statusBadge = document.getElementById("status-badge");
  noPersonWarning = document.getElementById("no-person-warning");
  loaderOverlay = document.getElementById("model-loader-overlay");
  loaderText = document.getElementById("loader-status-text");
  standbyState = document.getElementById("video-standby-state");
  if (standbyState) {
    originalStandbyHtml = standbyState.innerHTML;
  }
  delegateTag = document.getElementById("delegate-tag");
  modelVariantTag = document.getElementById("model-variant-tag");

  settingChatId = document.getElementById("setting-chat-id");
  settingSensitivity = document.getElementById("setting-sensitivity");
  settingSensitivityLabel = document.getElementById("setting-sensitivity-label");
  settingLyingDelay = document.getElementById("setting-lying-delay");
  settingLyingDelayLabel = document.getElementById("setting-lying-delay-label");
  settingSirenToggle = document.getElementById("setting-siren-toggle");
  btnResetSettings = document.getElementById("btn-reset-settings");
  btnToggleSettings = document.getElementById("btn-toggle-settings");
  settingsHeader = document.getElementById("settings-header");
  settingsChevron = document.getElementById("settings-chevron");
  settingsBody = document.getElementById("settings-body");
  serverWakeupBanner = document.getElementById("server-wakeup-banner");

  readoutTorsoAngle = document.getElementById("telemetry-torso-angle");
  readoutHipHeight = document.getElementById("telemetry-hip-height");
  readoutHipSpeed = document.getElementById("telemetry-hip-speed");
  readoutAspectRatio = document.getElementById("telemetry-aspect-ratio");
}

function toggleSettingsPanel() {
  if (!settingsBody) return;
  const isCurrentlyHidden = settingsBody.style.display === "none";
  settingsBody.style.display = isCurrentlyHidden ? "block" : "none";
  if (settingsChevron) {
    settingsChevron.style.transform = isCurrentlyHidden ? "rotate(0deg)" : "rotate(-90deg)";
  }
  if (btnToggleSettings) {
    btnToggleSettings.setAttribute("aria-expanded", isCurrentlyHidden ? "true" : "false");
  }
}

function bindEvents() {
  if (btnStart) {
    btnStart.addEventListener("click", () => {
      unlockAudioContext();
      startMonitoring();
    });
  }
  if (btnStop) {
    btnStop.addEventListener("click", () => stopMonitoring());
  }
  if (btnDismissAlert) {
    btnDismissAlert.addEventListener("click", () => dismissCurrentAlert());
  }
  if (btnToggleSiren) {
    btnToggleSiren.addEventListener("click", () => toggleSiren());
  }
  if (btnTestAlert) {
    btnTestAlert.addEventListener("click", () => triggerTestAlert());
  }

  // Settings Card Event Listeners
  if (btnToggleSettings) {
    btnToggleSettings.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleSettingsPanel();
    });
  }
  if (settingsHeader) {
    settingsHeader.addEventListener("click", () => toggleSettingsPanel());
  }

  if (settingChatId) {
    settingChatId.addEventListener("change", () => {
      currentSettings.chatId = settingChatId.value.trim();
      saveCurrentSettings();
      sendSettingsOverWs();
    });
  }

  if (settingSensitivity) {
    settingSensitivity.addEventListener("input", () => {
      currentSettings.sensitivity = parseInt(settingSensitivity.value, 10);
      updateSensitivityDisplay();
    });
    settingSensitivity.addEventListener("change", () => {
      currentSettings.sensitivity = parseInt(settingSensitivity.value, 10);
      saveCurrentSettings();
      sendSettingsOverWs();
    });
  }

  if (settingLyingDelay) {
    settingLyingDelay.addEventListener("input", () => {
      currentSettings.lyingDelay = parseFloat(settingLyingDelay.value);
      updateLyingDelayDisplay();
    });
    settingLyingDelay.addEventListener("change", () => {
      currentSettings.lyingDelay = parseFloat(settingLyingDelay.value);
      saveCurrentSettings();
      sendSettingsOverWs();
    });
  }

  if (settingSirenToggle) {
    settingSirenToggle.addEventListener("change", () => {
      sirenEnabled = settingSirenToggle.checked;
      currentSettings.sirenEnabled = sirenEnabled;
      saveCurrentSettings();
      updateSoundIndicator();
    });
  }

  if (btnResetSettings) {
    btnResetSettings.addEventListener("click", () => {
      currentSettings = {
        chatId: "",
        sensitivity: 3,
        lyingDelay: 2.0,
        sirenEnabled: true
      };
      sirenEnabled = true;
      saveCurrentSettings();
      applySettingsToInputs();
      updateSoundIndicator();
      sendSettingsOverWs();
      showToast("Settings reset to defaults.");
    });
  }
}

function updateModelInfoLabels() {
  if (modelVariantTag) {
    modelVariantTag.textContent = `Model: MediaPipe Pose (${MODEL_VARIANT.toUpperCase()})`;
  }
}

// ============================================================================
// WebSocket Client: Connect, Auto-Reconnect, and Message Streaming
// ============================================================================

function startServerWakeupTimer() {
  if (serverWakeupTimer) clearTimeout(serverWakeupTimer);
  serverWakeupTimer = setTimeout(() => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      if (serverWakeupBanner) {
        serverWakeupBanner.style.display = "block";
      }
    }
  }, 5000);
}

function clearServerWakeupTimer() {
  if (serverWakeupTimer) {
    clearTimeout(serverWakeupTimer);
    serverWakeupTimer = null;
  }
  if (serverWakeupBanner) {
    serverWakeupBanner.style.display = "none";
  }
}

/**
 * Initializes WebSocket connection to backend /ws endpoint on same host with client_id query param.
 */
function initWebSocket() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }

  const clientId = getClientId();
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host || "localhost:8000";
  const wsUrl = `${protocol}//${host}/ws?client_id=${encodeURIComponent(clientId)}`;

  console.log(`[FallGuard] Connecting to WebSocket: ${wsUrl}`);
  startServerWakeupTimer();

  try {
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log("[FallGuard] WebSocket connected to backend.");
      clearServerWakeupTimer();
      updateConnectionBadge(true);
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      sendSettingsOverWs();
    };

    socket.onmessage = (event) => {
      handleServerResponse(event.data);
    };

    socket.onclose = () => {
      console.warn("[FallGuard] WebSocket disconnected.");
      updateConnectionBadge(false);
      scheduleReconnect();
    };

    socket.onerror = (err) => {
      console.warn("[FallGuard] WebSocket connection error:", err);
      updateConnectionBadge(false);
      try {
        socket.close();
      } catch (e) {
        // socket may already be closing
      }
    };
  } catch (err) {
    console.error("[FallGuard] WebSocket creation failed:", err);
    updateConnectionBadge(false);
    scheduleReconnect();
  }
}

/**
 * Schedules an automatic reconnection attempt every 2 seconds.
 */
function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    initWebSocket();
  }, WS_RECONNECT_INTERVAL_MS);
}

/**
 * Updates on-screen WebSocket connection indicator pill (green/red)
 */
function updateConnectionBadge(connected) {
  if (!connectionBadge) return;
  if (connected) {
    connectionBadge.className = "connection-pill connected";
    connectionBadge.innerHTML = `<span class="connection-dot"></span>Connected`;
  } else {
    connectionBadge.className = "connection-pill disconnected";
    connectionBadge.innerHTML = `<span class="connection-dot"></span>Reconnecting...`;
  }
}

/**
 * Handles incoming JSON messages from FastAPI /ws.
 * Updates status badge: Normal / Possible fall / ALERT
 */
function handleServerResponse(rawText) {
  try {
    const data = JSON.parse(rawText);

    if (data.type === "settings_ack") {
      if (data.threshold !== undefined) {
        updateSensitivityDisplay(data.threshold);
      }
      if (data.lying_seconds !== undefined) {
        currentSettings.lyingDelay = data.lying_seconds;
        if (settingLyingDelay) settingLyingDelay.value = data.lying_seconds;
        updateLyingDelayDisplay();
      }
      return;
    }

    const status = (data.status || "normal").toLowerCase();
    const prob = data.probability !== undefined ? data.probability : 0.0;
    const probPercent = Math.round(prob * 100);
    const lyingProgress = data.lying_progress !== undefined ? Number(data.lying_progress) : 0.0;

    if (data.alert_id && !currentActiveAlertId) {
      currentActiveAlertId = data.alert_id;
    }

    if (status === "alert") {
      updateStatusBadge("alert", `ALERT (${probPercent}%)`);
      updateLyingProgressBar(1.0);
      showFallAlertBanner(data.alert_id, data.local_time);
    } else if (status === "possible_fall" || status === "warning" || status.includes("fall")) {
      const progPct = Math.round(lyingProgress * 100);
      updateStatusBadge("warning", `Possible Fall (${probPercent}%)`);
      updateLyingProgressBar(lyingProgress);
    } else {
      updateStatusBadge("normal", "Normal");
      updateLyingProgressBar(0.0);
    }
  } catch (err) {
    console.warn("[FallGuard] Error processing server WebSocket response:", err);
  }
}

function updateLyingProgressBar(progress) {
  let progressContainer = document.getElementById("lying-progress-container");
  let progressBar = document.getElementById("lying-progress-bar");

  if (!progressContainer && statusBadge && statusBadge.parentElement) {
    progressContainer = document.createElement("div");
    progressContainer.id = "lying-progress-container";
    progressContainer.style.width = "120px";
    progressContainer.style.height = "6px";
    progressContainer.style.background = "rgba(148, 163, 184, 0.3)";
    progressContainer.style.borderRadius = "4px";
    progressContainer.style.overflow = "hidden";
    progressContainer.style.marginTop = "6px";
    progressContainer.style.display = "none";

    progressBar = document.createElement("div");
    progressBar.id = "lying-progress-bar";
    progressBar.style.width = "0%";
    progressBar.style.height = "100%";
    progressBar.style.backgroundColor = "var(--warning, #f59e0b)";
    progressBar.style.transition = "width 0.15s ease-out";

    progressContainer.appendChild(progressBar);
    statusBadge.parentElement.appendChild(progressContainer);
  }

  if (progress > 0) {
    if (progressContainer) progressContainer.style.display = "block";
    if (progressBar) progressBar.style.width = `${Math.min(100, Math.round(progress * 100))}%`;
  } else {
    if (progressContainer) progressContainer.style.display = "none";
    if (progressBar) progressBar.style.width = "0%";
  }

  // Also update telemetry card timer if present
  const timerVals = document.querySelectorAll(".telemetry-val");
  if (timerVals && timerVals.length >= 2) {
    const sec = (progress * 2.0).toFixed(1);
    timerVals[1].textContent = `${sec}s / 2.0s`;
  }
}

/**
 * Sends landmark telemetry to the backend at ~10 messages/sec.
 * Strictly sends nothing if no person is detected in frame.
 */
function maybeSendLandmarksToWebSocket() {
  if (!rawLandmarks || !socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }

  const now = performance.now();
  if (now - lastWsSendTime < WS_SEND_INTERVAL_MS) {
    return; // Throttle to ~10 msg/sec
  }
  lastWsSendTime = now;

  // Format: {"t": timestamp_ms, "landmarks": [[x,y,z,visibility] x 33]}
  const payload = {
    t: Date.now(),
    landmarks: rawLandmarks.map((pt) => [
      Number(pt.x.toFixed(4)),
      Number(pt.y.toFixed(4)),
      Number(pt.z.toFixed(4)),
      Number((pt.visibility !== undefined ? pt.visibility : 1.0).toFixed(3))
    ])
  };

  try {
    socket.send(JSON.stringify(payload));
  } catch (err) {
    console.warn("[FallGuard] Failed to send WebSocket message:", err);
  }
}

// ============================================================================
// Web Audio API Siren & Alert Notification System
// ============================================================================

function unlockAudioContext() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;

    if (!audioContext) {
      audioContext = new AudioCtx();
    }
    if (audioContext.state === "suspended") {
      audioContext.resume();
    }
    const silentBuffer = audioContext.createBuffer(1, 1, 22050);
    const src = audioContext.createBufferSource();
    src.buffer = silentBuffer;
    src.connect(audioContext.destination);
    src.start(0);
    console.log("[FallGuard] AudioContext unlocked.");
  } catch (err) {
    console.warn("[FallGuard] AudioContext unlock notice:", err);
  }
}

/**
 * Synthesizes a looping siren sweeping between 600 Hz and 1000 Hz every second.
 * Pure Web Audio API synthesis - zero audio files required.
 */
function startSiren() {
  if (!sirenEnabled || sirenIsPlaying) return;
  unlockAudioContext();
  if (!audioContext) return;
  if (audioContext.state === "suspended") {
    audioContext.resume();
  }

  try {
    sirenIsPlaying = true;
    sirenOsc = audioContext.createOscillator();
    sirenGain = audioContext.createGain();

    sirenOsc.type = "sawtooth";
    sirenGain.gain.setValueAtTime(0.2, audioContext.currentTime);

    sirenOsc.connect(sirenGain);
    sirenGain.connect(audioContext.destination);

    const now = audioContext.currentTime;
    sirenOsc.frequency.setValueAtTime(600, now);
    sirenOsc.start(now);

    function scheduleSweep() {
      if (!sirenIsPlaying || !sirenOsc || !audioContext) return;
      const t = audioContext.currentTime;
      sirenOsc.frequency.cancelScheduledValues(t);
      sirenOsc.frequency.setValueAtTime(600, t);
      sirenOsc.frequency.linearRampToValueAtTime(1000, t + 0.5);
      sirenOsc.frequency.linearRampToValueAtTime(600, t + 1.0);
    }

    scheduleSweep();
    sirenSweepInterval = setInterval(scheduleSweep, 960);
  } catch (err) {
    console.warn("[FallGuard] Siren audio synthesis warning:", err);
  }
}

function stopSiren() {
  sirenIsPlaying = false;
  if (sirenSweepInterval) {
    clearInterval(sirenSweepInterval);
    sirenSweepInterval = null;
  }
  if (sirenOsc) {
    try {
      sirenOsc.stop();
      sirenOsc.disconnect();
    } catch (e) {}
    sirenOsc = null;
  }
  if (sirenGain) {
    try {
      sirenGain.disconnect();
    } catch (e) {}
    sirenGain = null;
  }
}

function toggleSiren() {
  sirenEnabled = !sirenEnabled;
  if (!sirenEnabled && sirenIsPlaying) {
    stopSiren();
  }
  updateSoundIndicator();
}

function updateSoundIndicator() {
  const ind = document.getElementById("sound-indicator");
  const label = document.getElementById("siren-toggle-label");
  if (sirenEnabled) {
    if (ind) {
      ind.className = "sound-pill enabled";
      ind.innerHTML = `<span class="sound-dot"></span><span>Sound Enabled</span>`;
    }
    if (label) label.textContent = "Siren: ON";
  } else {
    if (ind) {
      ind.className = "sound-pill muted";
      ind.innerHTML = `<span class="sound-dot muted"></span><span>Sound Muted</span>`;
    }
    if (label) label.textContent = "Siren: OFF";
  }
}

/**
 * Displays full-width red emergency banner and starts looping siren.
 * Keeps banner visible until explicitly dismissed, preventing duplicate banners.
 */
function showFallAlertBanner(alertId, localTimeStr) {
  const banner = document.getElementById("fall-alert-banner");
  if (!banner) return;

  if (alertId) {
    currentActiveAlertId = alertId;
  }

  const timestampEl = document.getElementById("banner-timestamp");
  if (timestampEl) {
    timestampEl.textContent = localTimeStr
      ? `(${localTimeStr})`
      : `(${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })})`;
  }

  const idTagEl = document.getElementById("banner-id-tag");
  if (idTagEl) {
    idTagEl.textContent = currentActiveAlertId ? `Alert #${currentActiveAlertId}` : "";
  }

  banner.style.display = "block";
  startSiren();
}

/**
 * Dismiss button handler: stops siren, hides banner, calls POST /alerts/{id}/dismiss,
 * and displays confirmation toast. Passes client_id via X-Client-Id header.
 */
async function dismissCurrentAlert() {
  stopSiren();
  const banner = document.getElementById("fall-alert-banner");
  if (banner) {
    banner.style.display = "none";
  }

  const idToDismiss = currentActiveAlertId;
  currentActiveAlertId = null;

  if (idToDismiss) {
    try {
      const clientId = getClientId();
      const resp = await fetch(`/alerts/${idToDismiss}/dismiss`, {
        method: "POST",
        headers: {
          "X-Client-Id": clientId
        }
      });
      if (resp.ok) {
        showToast(`Alert #${idToDismiss} dismissed. Caregiver notified of false alarm.`);
      } else {
        showToast(`Alert #${idToDismiss} dismissed locally.`);
      }
    } catch (err) {
      console.warn("Could not reach backend to dismiss alert:", err);
      showBackendError("Backend unreachable while attempting to dismiss alert.");
    }
  } else {
    showToast("Alert dismissed.");
  }
}

/**
 * Triggers a test alert, plays siren for 3 seconds, displays Telegram result with masked chat ID,
 * and passes client_id with rate limiting support.
 */
async function triggerTestAlert() {
  startSiren();
  setTimeout(() => {
    if (!currentActiveAlertId) {
      stopSiren();
    }
  }, 3000);

  try {
    const clientId = getClientId();
    const payload = {};
    if (currentSettings.chatId && currentSettings.chatId.trim()) {
      payload.telegram_chat_id = currentSettings.chatId.trim();
    }

    const resp = await fetch("/test-alert", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Client-Id": clientId
      },
      body: JSON.stringify(payload)
    });

    if (resp.status === 429) {
      const errData = await resp.json().catch(() => ({ detail: "Rate limit exceeded" }));
      showToast(`Rate limit: ${errData.detail || "Max 3 test messages per hour"}`);
      return;
    }

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${resp.status}`);
    }

    const result = await resp.json();
    const masked = result.masked_chat_id ? ` (chat ...${result.masked_chat_id})` : "";
    const tgSuccess = result.telegram_sent;
    const statusText = tgSuccess
      ? `Telegram test alert sent${masked}!`
      : `Telegram notification skipped or failed${masked}.`;
    showToast(`Test alert generated (#${result.alert.id}). ${statusText}`);
  } catch (err) {
    console.error("Test alert failed:", err);
    showBackendError(`Test alert failed: ${err.message}`);
  }
}

function showToast(message) {
  const toast = document.getElementById("dismiss-toast");
  const textEl = document.getElementById("dismiss-toast-message");
  if (toast && textEl) {
    textEl.textContent = message;
    toast.style.display = "flex";
    setTimeout(() => {
      toast.style.display = "none";
    }, 4500);
  }
}

function showBackendError(message) {
  const banner = document.getElementById("backend-error-banner");
  const textEl = document.getElementById("backend-error-text");
  if (banner && textEl) {
    textEl.textContent = message;
    banner.style.display = "flex";
    setTimeout(() => {
      banner.style.display = "none";
    }, 6000);
  }
}

// ============================================================================
// MediaPipe Pose Model Loader (GPU with automatic CPU fallback)
// ============================================================================

async function loadPoseModel() {
  if (poseLandmarker) return poseLandmarker;

  const modelPath = MODEL_ASSET_PATHS[MODEL_VARIANT] || MODEL_ASSET_PATHS.full;
  showLoader(true, `Downloading MediaPipe Tasks Vision runtime...`);
  const vision = await FilesetResolver.forVisionTasks(WASM_CDN_PATH);

  try {
    showLoader(true, `Initializing Pose Landmarker (${MODEL_VARIANT.toUpperCase()}) with GPU acceleration...`);
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelPath,
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: MIN_POSE_DETECTION_CONFIDENCE,
      minPosePresenceConfidence: MIN_POSE_PRESENCE_CONFIDENCE,
      minTrackingConfidence: MIN_TRACKING_CONFIDENCE
    });
    activeDelegate = "GPU";
    console.log(`[FallGuard] PoseLandmarker initialized successfully using GPU delegate (${MODEL_VARIANT}).`);
  } catch (gpuError) {
    console.warn("[FallGuard] GPU delegate initialization failed, falling back to CPU:", gpuError);
    showLoader(true, `GPU delegate unavailable. Falling back to CPU (${MODEL_VARIANT.toUpperCase()})...`);

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: modelPath,
        delegate: "CPU"
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: MIN_POSE_DETECTION_CONFIDENCE,
      minPosePresenceConfidence: MIN_POSE_PRESENCE_CONFIDENCE,
      minTrackingConfidence: MIN_TRACKING_CONFIDENCE
    });
    activeDelegate = "CPU";
    console.log(`[FallGuard] PoseLandmarker initialized successfully using CPU delegate (${MODEL_VARIANT}).`);
  }

  if (delegateTag) {
    delegateTag.textContent = `On-Device Wasm (${activeDelegate})`;
  }

  showLoader(false);
  return poseLandmarker;
}

// ============================================================================
// Camera & Monitoring Control
// ============================================================================

async function startMonitoring() {
  unlockAudioContext();
  initWebSocket();

  try {
    btnStart.disabled = true;
    showLoader(true, "Requesting webcam stream (640x480 @ 30fps)...");

    videoStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 30 },
        facingMode: "user"
      },
      audio: false
    });

    videoEl.srcObject = videoStream;
    await new Promise((resolve) => {
      videoEl.onloadedmetadata = () => {
        videoEl.play();
        resolve();
      };
    });

    syncCanvasDimensions();
    await loadPoseModel();

    if (standbyState) standbyState.style.display = "none";
    videoEl.style.display = "block";
    canvasEl.style.display = "block";
    showLoader(false);

    isMonitoring = true;
    btnStop.disabled = false;
    updateStatusBadge("normal");

    lastVideoTime = -1;
    frameCount = 0;
    lostFramesCount = 0;
    rawLandmarks = null;
    smoothedLandmarks = null;
    lastFpsUpdateTime = performance.now();
    prevHipY = null;
    prevFrameTime = null;

    requestAnimationFrame(detectionLoop);
  } catch (err) {
    console.error("[FallGuard] Camera start failed:", err);
    showLoader(false);
    btnStart.disabled = false;
    btnStop.disabled = true;
    handleCameraError(err);
  }
}

function resetStandbyState() {
  if (standbyState && originalStandbyHtml) {
    standbyState.innerHTML = originalStandbyHtml;
    standbyState.style.display = "flex";
  }
}

function handleCameraError(err) {
  const errName = err.name || "";
  let title = "Unable to Access Camera";
  let explanation = `An error occurred: ${err.message || "Unknown error"}. Please check your camera settings and try again.`;

  if (errName === "NotAllowedError" || errName === "PermissionDeniedError") {
    title = "Camera Permission Denied";
    explanation = "Camera access was blocked by your browser. Please allow camera permissions by clicking the lock or camera icon in your address bar, selecting 'Allow', and clicking Retry.";
  } else if (errName === "NotFoundError" || errName === "DevicesNotFoundError") {
    title = "No Camera Found";
    explanation = "No video input device was found. Please ensure your webcam is connected properly, powered on, and recognized by your system.";
  } else if (errName === "NotReadableError" || errName === "TrackStartError") {
    title = "Camera Already in Use";
    explanation = "Your camera is currently in use by another application or browser tab. Please close other software using the camera (Zoom, Teams, etc.) and click Retry.";
  }

  if (standbyState) {
    standbyState.style.display = "flex";
    standbyState.innerHTML = `
      <div style="text-align: center; max-width: 480px; padding: 1.5rem; background: rgba(15, 23, 42, 0.7); border-radius: var(--radius-md); border: 1px solid var(--border-color); box-shadow: 0 8px 32px rgba(0,0,0,0.4);">
        <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📷⚠️</div>
        <h3 style="font-size: 1.15rem; font-weight: 700; color: #f87171; margin-bottom: 0.5rem;">${title}</h3>
        <p style="font-size: 0.85rem; line-height: 1.5; color: #94a3b8; margin-bottom: 1.25rem;">
          ${explanation}
        </p>
        <button id="btn-camera-retry" class="btn btn-primary btn-sm">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="23 4 23 10 17 10"></polyline>
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
          </svg>
          <span>Retry Camera</span>
        </button>
      </div>
    `;
    const btnRetry = document.getElementById("btn-camera-retry");
    if (btnRetry) {
      btnRetry.addEventListener("click", () => {
        resetStandbyState();
        startMonitoring();
      });
    }
  }
}

function stopMonitoring() {
  isMonitoring = false;

  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  if (videoStream) {
    videoStream.getTracks().forEach((track) => track.stop());
    videoStream = null;
  }

  if (videoEl) {
    videoEl.srcObject = null;
    videoEl.style.display = "none";
  }

  if (canvasCtx && canvasEl) {
    canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    canvasEl.style.display = "none";
  }

  resetStandbyState();
  if (noPersonWarning) noPersonWarning.style.display = "none";

  rawLandmarks = null;
  smoothedLandmarks = null;
  lostFramesCount = 0;
  currentFps = 0;
  currentConfidence = 0;
  prevHipY = null;
  prevFrameTime = null;

  if (fpsBadge) fpsBadge.textContent = "FPS: --";
  if (confidenceBadge) confidenceBadge.textContent = "Conf: --%";
  resetTelemetryReadouts();

  btnStart.disabled = false;
  btnStop.disabled = true;
  updateStatusBadge("normal");
}

function syncCanvasDimensions() {
  if (!videoEl || !canvasEl) return;
  const vw = videoEl.videoWidth || 640;
  const vh = videoEl.videoHeight || 480;
  if (canvasEl.width !== vw || canvasEl.height !== vh) {
    canvasEl.width = vw;
    canvasEl.height = vh;
  }
}

// ============================================================================
// Landmark Smoothing (Exponential Moving Average)
// ============================================================================

function updateSmoothedLandmarks(raw) {
  if (!smoothedLandmarks || smoothedLandmarks.length !== raw.length) {
    smoothedLandmarks = raw.map((p) => ({
      x: p.x,
      y: p.y,
      z: p.z,
      visibility: p.visibility !== undefined ? p.visibility : 1.0
    }));
    return smoothedLandmarks;
  }

  for (let i = 0; i < raw.length; i++) {
    const r = raw[i];
    const s = smoothedLandmarks[i];
    const rVis = r.visibility !== undefined ? r.visibility : 1.0;
    const sVis = s.visibility !== undefined ? s.visibility : 1.0;

    s.x = SMOOTHING_ALPHA * r.x + (1 - SMOOTHING_ALPHA) * s.x;
    s.y = SMOOTHING_ALPHA * r.y + (1 - SMOOTHING_ALPHA) * s.y;
    s.z = SMOOTHING_ALPHA * r.z + (1 - SMOOTHING_ALPHA) * s.z;
    s.visibility = SMOOTHING_ALPHA * rVis + (1 - SMOOTHING_ALPHA) * sVis;
  }

  return smoothedLandmarks;
}

// ============================================================================
// Detection & Animation Frame Loop
// ============================================================================

function detectionLoop(timestamp) {
  if (!isMonitoring) return;

  calculateFps(timestamp);
  syncCanvasDimensions();

  if (videoEl.readyState >= 2 && videoEl.currentTime !== lastVideoTime) {
    lastVideoTime = videoEl.currentTime;

    canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    const startTimeMs = performance.now();
    const result = poseLandmarker.detectForVideo(videoEl, startTimeMs);

    if (result && result.landmarks && result.landmarks.length > 0) {
      lostFramesCount = 0;

      // Keep raw unsmoothed landmarks available
      rawLandmarks = result.landmarks[0];

      // Apply light smoothing for jitter-free visual rendering
      smoothedLandmarks = updateSmoothedLandmarks(rawLandmarks);

      // Compute on-screen pose confidence
      currentConfidence = computeAverageConfidence(rawLandmarks);
      updateConfidenceBadge(currentConfidence);

      if (noPersonWarning) {
        noPersonWarning.style.display = "none";
      }

      // Draw skeleton with visibility threshold (> 0.5)
      drawSkeleton(smoothedLandmarks, canvasCtx, canvasEl.width, canvasEl.height);

      // Compute kinematic preview metrics
      updateLiveTelemetry(smoothedLandmarks, startTimeMs);

      // Stream landmarks over WebSocket throttled at ~10 msg/sec
      maybeSendLandmarksToWebSocket();
    } else {
      // Frame has no detection
      rawLandmarks = null;
      lostFramesCount++;

      if (lostFramesCount <= MAX_LOST_FRAMES && smoothedLandmarks) {
        drawSkeleton(smoothedLandmarks, canvasCtx, canvasEl.width, canvasEl.height, 0.85);
        if (noPersonWarning) noPersonWarning.style.display = "none";
      } else {
        smoothedLandmarks = null;
        currentConfidence = 0;
        updateConfidenceBadge(0);

        if (noPersonWarning) {
          noPersonWarning.style.display = "flex";
        }
        resetTelemetryReadouts();
      }
    }

    if (typeof window.onLandmarksUpdated === "function") {
      try {
        window.onLandmarksUpdated(window.getLandmarks());
      } catch (err) {
        console.error("[FallGuard] onLandmarksUpdated hook error:", err);
      }
    }
  }

  animationFrameId = requestAnimationFrame(detectionLoop);
}

// ============================================================================
// Canvas Skeleton Rendering
// ============================================================================

function drawSkeleton(landmarks, ctx, width, height, alpha = 1.0) {
  if (!landmarks || landmarks.length === 0) return;

  ctx.save();
  ctx.globalAlpha = alpha;

  ctx.lineWidth = 3.5;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#0d9488";
  ctx.shadowColor = "rgba(13, 148, 136, 0.45)";
  ctx.shadowBlur = 6;

  for (const [i, j] of POSE_CONNECTIONS) {
    const ptA = landmarks[i];
    const ptB = landmarks[j];
    if (!ptA || !ptB) continue;

    const visA = ptA.visibility !== undefined ? ptA.visibility : 1.0;
    const visB = ptB.visibility !== undefined ? ptB.visibility : 1.0;

    if (visA > MIN_VISIBILITY_THRESHOLD && visB > MIN_VISIBILITY_THRESHOLD) {
      ctx.beginPath();
      ctx.moveTo(ptA.x * width, ptA.y * height);
      ctx.lineTo(ptB.x * width, ptB.y * height);
      ctx.stroke();
    }
  }

  ctx.shadowBlur = 0;
  for (let idx = 0; idx < landmarks.length; idx++) {
    const pt = landmarks[idx];
    const vis = pt.visibility !== undefined ? pt.visibility : 1.0;

    if (vis > MIN_VISIBILITY_THRESHOLD) {
      const x = pt.x * width;
      const y = pt.y * height;

      ctx.beginPath();
      ctx.arc(x, y, 4.5, 0, 2 * Math.PI);
      ctx.fillStyle = "#ffffff";
      ctx.fill();

      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fillStyle = idx >= 11 && idx <= 24 ? "#14b8a6" : "#f59e0b";
      ctx.fill();
    }
  }

  ctx.restore();
}

// ============================================================================
// Confidence & Telemetry
// ============================================================================

function computeAverageConfidence(landmarks) {
  if (!landmarks || landmarks.length === 0) return 0;
  let total = 0;
  for (const pt of landmarks) {
    total += pt.visibility !== undefined ? pt.visibility : 1.0;
  }
  return total / landmarks.length;
}

function updateConfidenceBadge(score) {
  if (!confidenceBadge) return;
  if (score <= 0) {
    confidenceBadge.textContent = "Conf: --%";
    confidenceBadge.style.color = "var(--text-faint)";
  } else {
    const pct = Math.round(score * 100);
    confidenceBadge.textContent = `Conf: ${pct}%`;
    confidenceBadge.style.color = pct >= 70 ? "var(--safe)" : pct >= 50 ? "var(--warning)" : "var(--alert)";
  }
}

function calculateFps(now) {
  frameCount++;
  const elapsed = now - lastFpsUpdateTime;
  if (elapsed >= 500) {
    currentFps = Math.round((frameCount * 1000) / elapsed);
    if (fpsBadge) {
      fpsBadge.textContent = `FPS: ${currentFps}`;
    }
    frameCount = 0;
    lastFpsUpdateTime = now;
  }
}

function updateLiveTelemetry(lm, currentTimeMs) {
  try {
    const leftShoulder = lm[11];
    const rightShoulder = lm[12];
    const leftHip = lm[23];
    const rightHip = lm[24];

    if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) return;

    const shoulderX = (leftShoulder.x + rightShoulder.x) / 2;
    const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
    const hipX = (leftHip.x + rightHip.x) / 2;
    const hipY = (leftHip.y + rightHip.y) / 2;

    const dx = hipX - shoulderX;
    const dy = hipY - shoulderY;
    const torsoLength = Math.sqrt(dx * dx + dy * dy) || 1e-4;

    const angleRad = Math.atan2(Math.abs(dx), Math.abs(dy));
    const angleDeg = (angleRad * 180) / Math.PI;

    let normSpeed = 0;
    if (prevHipY !== null && prevFrameTime !== null) {
      const dt = (currentTimeMs - prevFrameTime) / 1000;
      if (dt > 0.005) {
        const deltaHipY = hipY - prevHipY;
        normSpeed = (deltaHipY / torsoLength) / dt;
      }
    }
    prevHipY = hipY;
    prevFrameTime = currentTimeMs;

    let minX = 1, maxX = 0, minY = 1, maxY = 0;
    for (const p of lm) {
      if ((p.visibility || 1) > MIN_VISIBILITY_THRESHOLD) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      }
    }
    const bboxW = Math.max(0.01, maxX - minX);
    const bboxH = Math.max(0.01, maxY - minY);
    const aspectRatio = bboxW / bboxH;
    const hipHeightNorm = (1.0 - hipY) / torsoLength;

    if (readoutTorsoAngle) readoutTorsoAngle.textContent = `${angleDeg.toFixed(1)}°`;
    if (readoutHipHeight) readoutHipHeight.textContent = hipHeightNorm.toFixed(2);
    if (readoutHipSpeed) readoutHipSpeed.textContent = `${Math.abs(normSpeed).toFixed(2)} /s`;
    if (readoutAspectRatio) readoutAspectRatio.textContent = aspectRatio.toFixed(2);
  } catch (err) {
    console.debug("[FallGuard] Telemetry calc notice:", err);
  }
}

function resetTelemetryReadouts() {
  if (readoutTorsoAngle) readoutTorsoAngle.textContent = "--";
  if (readoutHipHeight) readoutHipHeight.textContent = "--";
  if (readoutHipSpeed) readoutHipSpeed.textContent = "--";
  if (readoutAspectRatio) readoutAspectRatio.textContent = "--";
}

export function updateStatusBadge(status, customLabel) {
  if (!statusBadge) return;
  statusBadge.className = `monitor-badge ${status}`;

  if (status === "alert") {
    statusBadge.innerHTML = `<span class="pulse-dot"></span>${customLabel || "ALERT"}`;
  } else if (status === "warning") {
    statusBadge.innerHTML = `<span class="pulse-dot"></span>${customLabel || "Possible Fall"}`;
  } else {
    statusBadge.innerHTML = `<span class="pulse-dot"></span>${customLabel || "Normal"}`;
  }
}

function showLoader(show, text = "Loading...") {
  if (!loaderOverlay) return;
  loaderOverlay.style.display = show ? "flex" : "none";
  if (loaderText) loaderText.textContent = text;
}

// ============================================================================
// Public Interface (Raw Unsmoothed Landmarks for Backend)
// ============================================================================

window.getLandmarks = function () {
  return {
    timestamp: Date.now(),
    fps: currentFps,
    detected: rawLandmarks !== null,
    confidence: Number(currentConfidence.toFixed(3)),
    landmarks: rawLandmarks
      ? rawLandmarks.map((pt) => ({
          x: Number(pt.x.toFixed(4)),
          y: Number(pt.y.toFixed(4)),
          z: Number(pt.z.toFixed(4)),
          visibility: Number((pt.visibility !== undefined ? pt.visibility : 1.0).toFixed(3))
        }))
      : null
  };
};

window.updateStatusBadge = updateStatusBadge;
window.startMonitoring = startMonitoring;
window.stopMonitoring = stopMonitoring;
window.onLandmarksUpdated = null;

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
const CAMERA_PREF_KEY = "fallguard_camera_facing";
let currentSettings = {
  chatId: "",
  isChatVerified: false,
  sensitivity: 3,
  lyingDelay: 2.0,
  sirenEnabled: true,
  sendClip: false,
  clipConsent: false
};

// Camera & Device State
let currentFacingMode = "front"; // "front" or "back"
let preferredBackDeviceId = null;
let isSwitchingCamera = false;
let wakeLock = null;

// DOM References
let videoEl = null;
let canvasEl = null;
let canvasCtx = null;
let btnStart = null;
let btnStop = null;
let btnSwitchCamera = null;
let activeCameraLabel = null;
let cameraSwitchWrapper = null;
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
let httpsWarningBanner = null;
let visibilityWarningBanner = null;

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
let btnVerifyChatId = null;
let btnVerifyChatText = null;
let chatIdStatusBadge = null;
let chatIdFeedback = null;
let settingSensitivity = null;
let settingSensitivityLabel = null;
let settingLyingDelay = null;
let settingLyingDelayLabel = null;
let settingSirenToggle = null;
let settingClipToggle = null;
let settingClipConsent = null;
let clipConsentContainer = null;
let testClipActionRow = null;
let btnTestClip = null;
let testClipStatus = null;
let clipRecordingIndicator = null;
let clipIndicatorText = null;
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

// Incident Clip Recording & State Machine
let mediaRecorder = null;
let segmentIntervalTimer = null;
let segmentStartTime = 0;
let segmentRingBuffer = []; // stores at most 6 segments: { blob, start_ms, end_ms }
let isRecordingSegments = false;
let pageHideTimeout = null;
let isTestingClip = false;

let fallIncident = {
  active: false,
  T: 0,
  preSegments: [],
  postSegments: [],
  alertId: null,
  uploadStarted: false,
  uploadComplete: false,
  countdownInterval: null,
  discardTimeout: null
};

// ============================================================================
// Initialization & Event Binding
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
  initDOMElements();
  loadStoredSettings();
  loadCameraPreference();
  checkHttpsRequirement();
  setupVisibilityAndWakeLockHandlers();
  applySettingsToInputs();
  bindEvents();
  updateModelInfoLabels();
  updateSoundIndicator();
  updateCameraUI();
  checkDeviceCount();
  initWebSocket();
  // On page load, re-send the SAVED consent from localStorage once
  syncClipConsentWithServer(Boolean(currentSettings.sendClip && currentSettings.clipConsent));
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
        if (parsed.isChatVerified !== undefined) currentSettings.isChatVerified = Boolean(parsed.isChatVerified);
        if (parsed.sensitivity !== undefined) currentSettings.sensitivity = parseInt(parsed.sensitivity, 10) || 3;
        if (parsed.lyingDelay !== undefined) currentSettings.lyingDelay = parseFloat(parsed.lyingDelay) || 2.0;
        if (parsed.sirenEnabled !== undefined) currentSettings.sirenEnabled = Boolean(parsed.sirenEnabled);
        if (parsed.clipConsent !== undefined) currentSettings.clipConsent = Boolean(parsed.clipConsent);
        if (parsed.sendClip !== undefined) currentSettings.sendClip = Boolean(parsed.sendClip);
        if (!currentSettings.clipConsent) {
          currentSettings.sendClip = false;
        }
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
  if (settingClipConsent) settingClipConsent.checked = Boolean(currentSettings.clipConsent);
  if (settingClipToggle) settingClipToggle.checked = Boolean(currentSettings.sendClip && currentSettings.clipConsent);
  if (currentSettings.chatId && currentSettings.isChatVerified && chatIdStatusBadge) {
    chatIdStatusBadge.style.display = "inline-block";
    chatIdStatusBadge.style.background = "rgba(16, 185, 129, 0.15)";
    chatIdStatusBadge.style.color = "#10b981";
    chatIdStatusBadge.style.border = "1px solid #10b981";
    chatIdStatusBadge.textContent = "✓ Verified Key";
  }
  updateSensitivityDisplay();
  updateLyingDelayDisplay();
  updateClipUIState();
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
    lying_seconds: currentSettings.lyingDelay,
    clip_consent: Boolean(currentSettings.sendClip && currentSettings.clipConsent)
  };
  try {
    socket.send(JSON.stringify(payload));
    console.log("[FallGuard] Sent settings over WebSocket:", payload);
  } catch (err) {
    console.warn("[FallGuard] Failed to send settings over WebSocket:", err);
  }
}

async function syncClipConsentWithServer(targetConsent) {
  const clientId = getClientId();
  try {
    const resp = await fetch("/api/consent/clip", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Client-Id": clientId
      },
      body: JSON.stringify({
        consent: Boolean(targetConsent)
      })
    });

    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }

    const data = await resp.json();
    const serverConsent = Boolean(data.consent);

    // If server's stored value differs from the toggle, set the toggle to the server's value and show a warning
    if (serverConsent !== Boolean(targetConsent)) {
      console.warn("[FallGuard] Server consent differs from toggle:", serverConsent, "vs", targetConsent);
      showToast("Warning: Server consent setting differs from device toggle.");
    }

    currentSettings.sendClip = serverConsent;
    if (settingClipToggle) {
      settingClipToggle.checked = Boolean(serverConsent && currentSettings.clipConsent);
    }
    saveCurrentSettings();
    updateClipRecordingState();
    updateClipUIState();
    return serverConsent;
  } catch (err) {
    console.error("[FallGuard] Failed to sync clip consent with server:", err);
    showToast("Warning: Failed to update video consent with server.");
    if (settingClipToggle) {
      settingClipToggle.checked = Boolean(currentSettings.sendClip && currentSettings.clipConsent);
    }
    return null;
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
  btnSwitchCamera = document.getElementById("btn-switch-camera");
  activeCameraLabel = document.getElementById("active-camera-label");
  cameraSwitchWrapper = document.getElementById("camera-switch-wrapper");
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
  httpsWarningBanner = document.getElementById("https-warning-banner");
  visibilityWarningBanner = document.getElementById("visibility-warning-banner");

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
  btnVerifyChatId = document.getElementById("btn-verify-chat-id");
  btnVerifyChatText = document.getElementById("btn-verify-chat-text");
  chatIdStatusBadge = document.getElementById("chat-id-status-badge");
  chatIdFeedback = document.getElementById("chat-id-feedback");
  settingSensitivity = document.getElementById("setting-sensitivity");
  settingSensitivityLabel = document.getElementById("setting-sensitivity-label");
  settingLyingDelay = document.getElementById("setting-lying-delay");
  settingLyingDelayLabel = document.getElementById("setting-lying-delay-label");
  settingSirenToggle = document.getElementById("setting-siren-toggle");
  settingClipToggle = document.getElementById("setting-clip-toggle");
  settingClipConsent = document.getElementById("setting-clip-consent");
  clipConsentContainer = document.getElementById("clip-consent-container");
  testClipActionRow = document.getElementById("test-clip-action-row");
  btnTestClip = document.getElementById("btn-test-clip");
  testClipStatus = document.getElementById("test-clip-status");
  clipRecordingIndicator = document.getElementById("clip-recording-indicator");
  clipIndicatorText = document.getElementById("clip-indicator-text");
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
  if (btnSwitchCamera) {
    btnSwitchCamera.addEventListener("click", () => switchCamera());
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

  if (btnVerifyChatId) {
    btnVerifyChatId.addEventListener("click", () => verifyChatId());
  }

  if (settingChatId) {
    settingChatId.addEventListener("input", () => {
      currentSettings.isChatVerified = false;
      if (chatIdStatusBadge) chatIdStatusBadge.style.display = "none";
      if (chatIdFeedback) chatIdFeedback.style.display = "none";
    });
    settingChatId.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        verifyChatId();
      }
    });
    settingChatId.addEventListener("change", () => {
      currentSettings.chatId = settingChatId.value.trim();
      currentSettings.isChatVerified = false;
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

  if (settingClipToggle) {
    settingClipToggle.addEventListener("change", async () => {
      const isChecked = settingClipToggle.checked;
      if (isChecked && !currentSettings.clipConsent) {
        settingClipToggle.checked = false;
        if (clipConsentContainer) {
          clipConsentContainer.style.borderColor = "var(--alert, #ef4444)";
          clipConsentContainer.style.boxShadow = "0 0 8px rgba(239, 68, 68, 0.4)";
          setTimeout(() => {
            if (clipConsentContainer) {
              clipConsentContainer.style.borderColor = "var(--border-color)";
              clipConsentContainer.style.boxShadow = "none";
            }
          }, 2000);
        }
        showToast("Please agree to the consent statement first to enable video clips.");
        return;
      }

      // Send consent from one place only and wait for server response
      const targetConsent = Boolean(isChecked && currentSettings.clipConsent);
      await syncClipConsentWithServer(targetConsent);
    });
  }

  if (settingClipConsent) {
    settingClipConsent.addEventListener("change", async () => {
      currentSettings.clipConsent = settingClipConsent.checked;
      if (!currentSettings.clipConsent) {
        currentSettings.sendClip = false;
        if (settingClipToggle) settingClipToggle.checked = false;
        await syncClipConsentWithServer(false);
      } else {
        saveCurrentSettings();
      }
    });
  }

  if (btnTestClip) {
    btnTestClip.addEventListener("click", () => triggerTestClip());
  }

  if (btnResetSettings) {
    btnResetSettings.addEventListener("click", async () => {
      currentSettings = {
        chatId: "",
        isChatVerified: false,
        sensitivity: 3,
        lyingDelay: 2.0,
        sirenEnabled: true,
        sendClip: false,
        clipConsent: false
      };
      sirenEnabled = true;
      saveCurrentSettings();
      applySettingsToInputs();
      updateSoundIndicator();
      sendSettingsOverWs();
      await syncClipConsentWithServer(false);
      showToast("Settings reset to defaults.");
    });
  }
}

// ============================================================================
// Incident Clip Recording Engine & MediaRecorder Ring Buffer
// ============================================================================

function updateClipUIState() {
  const isConsented = Boolean(currentSettings.sendClip && currentSettings.clipConsent);
  if (testClipActionRow) {
    testClipActionRow.style.display = isConsented ? "flex" : "none";
  }
  if (clipRecordingIndicator) {
    if (isMonitoring && isConsented) {
      clipRecordingIndicator.style.display = "inline-flex";
    } else {
      clipRecordingIndicator.style.display = "none";
    }
  }
}

function updateClipRecordingState() {
  const isConsented = Boolean(currentSettings.sendClip && currentSettings.clipConsent);
  if (isMonitoring && isConsented && videoStream && videoStream.active) {
    startSegmentRecorder();
  } else {
    stopSegmentRecorder();
    clearRingBuffer();
    resetIncidentCapture();
  }
  updateClipUIState();
}

function getSupportedMimeType() {
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4"
  ];
  if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) {
    return "";
  }
  for (const t of candidates) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return "";
}

function startSegmentRecorder() {
  if (!isMonitoring || !currentSettings.sendClip || !currentSettings.clipConsent) {
    stopSegmentRecorder();
    return;
  }
  if (!videoStream || !videoStream.active) return;
  if (isRecordingSegments) return;

  const mime = getSupportedMimeType();
  if (!mime) {
    console.warn("[FallGuard] No supported video MIME type found for MediaRecorder.");
    return;
  }

  isRecordingSegments = true;
  updateClipUIState();
  recordNextTwoSecondSegment();
}

function recordNextTwoSecondSegment() {
  if (!isRecordingSegments || !videoStream || !videoStream.active) {
    isRecordingSegments = false;
    updateClipUIState();
    return;
  }

  const mime = getSupportedMimeType();
  try {
    mediaRecorder = new MediaRecorder(videoStream, {
      mimeType: mime,
      videoBitsPerSecond: 800000
    });
  } catch (err) {
    console.error("[FallGuard] Failed to instantiate MediaRecorder:", err);
    isRecordingSegments = false;
    updateClipUIState();
    return;
  }

  segmentStartTime = Date.now();
  let chunks = [];

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) {
      chunks.push(e.data);
    }
  };

  mediaRecorder.onstop = () => {
    if (chunks.length > 0) {
      const segBlob = new Blob(chunks, { type: mime });
      const segObj = {
        blob: segBlob,
        start_ms: segmentStartTime,
        end_ms: Date.now()
      };
      handleNewSegment(segObj);
    }
    if (isRecordingSegments) {
      recordNextTwoSecondSegment();
    }
  };

  try {
    mediaRecorder.start();
    if (segmentIntervalTimer) clearTimeout(segmentIntervalTimer);
    segmentIntervalTimer = setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        try {
          mediaRecorder.stop();
        } catch (e) {}
      }
    }, 2000);
  } catch (err) {
    console.warn("[FallGuard] Error starting MediaRecorder segment:", err);
    isRecordingSegments = false;
  }
}

function stopSegmentRecorder() {
  isRecordingSegments = false;
  if (segmentIntervalTimer) {
    clearTimeout(segmentIntervalTimer);
    segmentIntervalTimer = null;
  }
  if (mediaRecorder) {
    try {
      if (mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      }
    } catch (e) {}
    mediaRecorder = null;
  }
}

function clearRingBuffer() {
  segmentRingBuffer = [];
}

function handleNewSegment(segObj) {
  // Maintain ring buffer of the 6 most recent 2-second segments (12s total) in volatile memory
  segmentRingBuffer.push(segObj);
  if (segmentRingBuffer.length > 6) {
    segmentRingBuffer.shift();
  }

  // If capturing post-fall segments for an active incident:
  if (fallIncident.active && !fallIncident.uploadStarted) {
    if (segObj.end_ms <= fallIncident.T + 12000 || fallIncident.postSegments.length < 5) {
      fallIncident.postSegments.push(segObj);
    }
    checkIncidentUploadReadiness();
  }
}

function triggerPossibleFallIncident(existingAlertId) {
  if (!currentSettings.sendClip || !currentSettings.clipConsent) return;
  if (fallIncident.active) return;

  const now = Date.now();
  fallIncident.active = true;
  fallIncident.T = now;
  // Freeze shallow copy of buffered pre-fall segments
  fallIncident.preSegments = [...segmentRingBuffer];
  fallIncident.postSegments = [];
  fallIncident.alertId = existingAlertId || null;
  fallIncident.uploadStarted = false;
  fallIncident.uploadComplete = false;

  console.log(`[FallGuard] Possible fall onset T=${now}. Pre-fall segments frozen: ${fallIncident.preSegments.length}`);

  if (fallIncident.countdownInterval) clearInterval(fallIncident.countdownInterval);
  fallIncident.countdownInterval = setInterval(updateIncidentCountdownUI, 250);

  // Requirement: Give up and discard after T + 40 s
  if (fallIncident.discardTimeout) clearTimeout(fallIncident.discardTimeout);
  fallIncident.discardTimeout = setTimeout(() => {
    if (fallIncident.active && !fallIncident.uploadComplete) {
      console.log("[FallGuard] Incident clip upload expired after T + 40s. Discarding.");
      resetIncidentCapture("Expired after 40s timeout");
    }
  }, 40000);
}

function updateIncidentCountdownUI() {
  if (!fallIncident.active) return;
  if (fallIncident.uploadStarted) return;

  const now = Date.now();
  const targetTime = fallIncident.T + 10000;
  const remainingSec = Math.max(0, Math.ceil((targetTime - now) / 1000));

  if (clipIndicatorText) {
    if (remainingSec > 0) {
      clipIndicatorText.textContent = `Clip will be sent in ${remainingSec} s`;
    } else {
      clipIndicatorText.textContent = fallIncident.alertId
        ? "Uploading clip..."
        : "Waiting for alert confirmation...";
    }
  }
}

function checkIncidentUploadReadiness() {
  if (!fallIncident.active || fallIncident.uploadStarted) return;

  const now = Date.now();
  const timeElapsed = (now >= fallIncident.T + 10000);
  const hasAlert = Boolean(fallIncident.alertId);

  // Upload when BOTH are true: at least T + 10 s has passed AND server reported alert
  if (timeElapsed && hasAlert) {
    fallIncident.uploadStarted = true;
    if (fallIncident.countdownInterval) {
      clearInterval(fallIncident.countdownInterval);
      fallIncident.countdownInterval = null;
    }

    if (clipIndicatorText) {
      clipIndicatorText.textContent = "Uploading clip...";
    }

    const segmentsToUpload = [...fallIncident.preSegments, ...fallIncident.postSegments];
    const alertId = fallIncident.alertId;
    const T_val = fallIncident.T;

    uploadClipPayload(alertId, T_val, segmentsToUpload, false);
  }
}

async function uploadClipPayload(alertId, T_val, segmentsList, isTestClip = false) {
  if (!segmentsList || segmentsList.length === 0) {
    const reason = "No video footage available in buffer";
    handleClipUploadFailure(reason, isTestClip);
    return;
  }

  try {
    const formData = new FormData();
    const metaSegments = segmentsList.map((s) => ({ start_ms: s.start_ms }));
    const metadataObj = {
      T: T_val,
      segments: metaSegments
    };
    formData.append("metadata", JSON.stringify(metadataObj));

    segmentsList.forEach((seg, idx) => {
      const type = (seg.blob && seg.blob.type) || "video/webm";
      const ext = type.includes("mp4") ? "mp4" : "webm";
      formData.append("segments", seg.blob, `segment_${String(idx).padStart(3, "0")}.${ext}`);
    });

    const clientId = getClientId();
    const resp = await fetch(`/alerts/${alertId}/clip`, {
      method: "POST",
      headers: {
        "X-Client-Id": clientId
      },
      body: formData
    });

    const data = await resp.json().catch(() => ({}));

    if (resp.ok) {
      handleClipUploadSuccess(data, isTestClip);
    } else {
      const errMsg = data.detail || `Upload failed (HTTP ${resp.status})`;
      handleClipUploadFailure(errMsg, isTestClip);
    }
  } catch (err) {
    console.error("[FallGuard] Clip upload error:", err);
    handleClipUploadFailure(err.message || "Network error", isTestClip);
  }
}

function handleClipUploadSuccess(data, isTestClip = false) {
  if (isTestClip) {
    if (testClipStatus) {
      testClipStatus.textContent = "Clip sent!";
      testClipStatus.style.color = "var(--safe, #10b981)";
    }
    showToast("Test clip sent successfully to caregiver!");
  } else {
    fallIncident.uploadComplete = true;
    if (clipIndicatorText) {
      clipIndicatorText.textContent = "Clip sent";
    }
    showToast("Incident video clip sent to caregiver's Telegram chat.");
    setTimeout(() => {
      resetIncidentCapture();
    }, 6000);
  }
}

function handleClipUploadFailure(reason, isTestClip = false) {
  if (isTestClip) {
    if (testClipStatus) {
      testClipStatus.textContent = `Failed: ${reason}`;
      testClipStatus.style.color = "var(--alert, #ef4444)";
    }
    showToast(`Test clip failed: ${reason}`);
  } else {
    if (clipIndicatorText) {
      clipIndicatorText.textContent = `Clip failed: ${reason}`;
    }
    showToast(`Video clip upload failed: ${reason}`);
    setTimeout(() => {
      resetIncidentCapture();
    }, 8000);
  }
}

function resetIncidentCapture(reason) {
  if (fallIncident.countdownInterval) {
    clearInterval(fallIncident.countdownInterval);
    fallIncident.countdownInterval = null;
  }
  if (fallIncident.discardTimeout) {
    clearTimeout(fallIncident.discardTimeout);
    fallIncident.discardTimeout = null;
  }
  fallIncident.active = false;
  fallIncident.T = 0;
  fallIncident.preSegments = [];
  fallIncident.postSegments = [];
  fallIncident.alertId = null;
  fallIncident.uploadStarted = false;
  fallIncident.uploadComplete = false;

  if (clipIndicatorText) {
    clipIndicatorText.textContent = "Recording the last 10 s on this device only until a fall is confirmed.";
  }
}

async function triggerTestClip() {
  if (isTestingClip) return;
  if (!currentSettings.sendClip || !currentSettings.clipConsent) {
    showToast("Please enable and consent to video clips first.");
    return;
  }

  if (!isMonitoring || !videoStream || !videoStream.active) {
    showToast("Please start monitoring first so camera video is available.");
    return;
  }

  isTestingClip = true;
  if (btnTestClip) btnTestClip.disabled = true;
  if (testClipStatus) {
    testClipStatus.style.color = "var(--text-faint)";
    testClipStatus.textContent = "Authorizing test...";
  }

  try {
    const clientId = getClientId();
    const payload = {
      is_test_clip: true
    };
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
      const err = await resp.json().catch(() => ({}));
      const msg = err.detail || "Rate limit: at most 2 test clips per hour.";
      if (testClipStatus) {
        testClipStatus.textContent = msg;
        testClipStatus.style.color = "var(--alert, #ef4444)";
      }
      showToast(msg);
      return;
    }

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    const testAlertData = await resp.json();
    const testAlertId = testAlertData.alert ? testAlertData.alert.id : null;
    if (!testAlertId) throw new Error("Server did not return a test alert ID.");

    // Start 10 second countdown while recording test segments (no siren or escalation)
    let countdownSec = 10;
    const testSegments = [...segmentRingBuffer];
    const testStartT = Date.now();

    const testCountdownInterval = setInterval(() => {
      countdownSec--;
      if (testClipStatus) {
        testClipStatus.textContent = `Recording test clip (${countdownSec}s)...`;
      }
      if (countdownSec <= 0) {
        clearInterval(testCountdownInterval);
        if (testClipStatus) testClipStatus.textContent = "Uploading test clip...";
        const allSegments = segmentRingBuffer.length > 0 ? [...segmentRingBuffer] : testSegments;
        uploadClipPayload(testAlertId, testStartT, allSegments, true);
      }
    }, 1000);

    if (testClipStatus) {
      testClipStatus.textContent = `Recording test clip (${countdownSec}s)...`;
    }
  } catch (err) {
    console.error("[FallGuard] Test clip error:", err);
    if (testClipStatus) {
      testClipStatus.textContent = `Failed: ${err.message}`;
      testClipStatus.style.color = "var(--alert, #ef4444)";
    }
    showToast(`Test clip failed: ${err.message}`);
  } finally {
    setTimeout(() => {
      isTestingClip = false;
      if (btnTestClip) btnTestClip.disabled = false;
    }, 12000);
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
      // On every WebSocket reconnect, re-send the SAVED consent from localStorage once
      syncClipConsentWithServer(Boolean(currentSettings.sendClip && currentSettings.clipConsent));
    };

    socket.onmessage = (event) => {
      handleServerResponse(event.data);
    };

    socket.onclose = () => {
      console.warn("[FallGuard] WebSocket disconnected.");
      updateConnectionBadge(false);
      scheduleReconnect();
      if (isMonitoring) {
        reportErrorToTelegram("WebSocket disconnected during active monitoring. Attempting reconnect...", "WebSocket");
      }
    };

    socket.onerror = (err) => {
      console.warn("[FallGuard] WebSocket connection error:", err);
      updateConnectionBadge(false);
      if (isMonitoring) {
        reportErrorToTelegram("WebSocket communication error encountered.", "WebSocket");
      }
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

    if (data.alert_id) {
      if (!currentActiveAlertId) {
        currentActiveAlertId = data.alert_id;
      }
      if (fallIncident.active && !fallIncident.alertId) {
        fallIncident.alertId = data.alert_id;
        checkIncidentUploadReadiness();
      }
    }

    if (status === "alert") {
      updateStatusBadge("alert", `ALERT (${probPercent}%)`);
      updateLyingProgressBar(1.0);
      showFallAlertBanner(data.alert_id, data.local_time);
      if (data.alert_id && fallIncident.active) {
        fallIncident.alertId = data.alert_id;
        checkIncidentUploadReadiness();
      }
    } else if (status === "possible_fall" || status === "warning" || status.includes("fall")) {
      const progPct = Math.round(lyingProgress * 100);
      updateStatusBadge("warning", `Possible Fall (${probPercent}%)`);
      updateLyingProgressBar(lyingProgress);
      // Incident clip: first moment server status becomes possible_fall
      if (currentSettings.sendClip && currentSettings.clipConsent && !fallIncident.active) {
        triggerPossibleFallIncident(data.alert_id || null);
      }
    } else {
      updateStatusBadge("normal", "Normal");
      updateLyingProgressBar(0.0);
      // If a possible fall was active without an alert and status returned to normal, discard
      if (fallIncident.active && !fallIncident.alertId && !fallIncident.uploadStarted) {
        if (Date.now() > fallIncident.T + 12000) {
          resetIncidentCapture("Possible fall timed out with no alert");
        }
      }
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
  resetIncidentCapture("Alert dismissed by user before upload");
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
  reportErrorToTelegram(message, "Backend");
}

function escapeHtml(text) {
  if (!text) return "";
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function maskChatId(chatId) {
  if (!chatId) return "";
  const str = String(chatId).trim();
  if (str.length <= 4) return `...${str}`;
  return `...${str.slice(-4)}`;
}

/**
 * Performs real internal checks with Telegram Bot API and sends a verification test message.
 * Never blindly displays valid key without internal confirmation from Telegram API.
 */
async function verifyChatId() {
  if (!settingChatId) return;
  const rawId = settingChatId.value.trim();

  if (!rawId) {
    if (chatIdStatusBadge) {
      chatIdStatusBadge.style.display = "inline-block";
      chatIdStatusBadge.style.background = "rgba(239, 68, 68, 0.15)";
      chatIdStatusBadge.style.color = "#ef4444";
      chatIdStatusBadge.style.border = "1px solid #ef4444";
      chatIdStatusBadge.textContent = "✗ Missing";
    }
    if (chatIdFeedback) {
      chatIdFeedback.style.display = "block";
      chatIdFeedback.style.background = "rgba(239, 68, 68, 0.08)";
      chatIdFeedback.style.border = "1px solid rgba(239, 68, 68, 0.3)";
      chatIdFeedback.style.color = "var(--text-main)";
      chatIdFeedback.innerHTML = "<strong>Please enter a Telegram Chat ID first.</strong>";
    }
    settingChatId.focus();
    return;
  }

  // Visual loading state
  if (btnVerifyChatId) btnVerifyChatId.disabled = true;
  if (btnVerifyChatText) btnVerifyChatText.textContent = "Verifying...";
  if (chatIdStatusBadge) {
    chatIdStatusBadge.style.display = "inline-block";
    chatIdStatusBadge.style.background = "rgba(245, 158, 11, 0.15)";
    chatIdStatusBadge.style.color = "#d97706";
    chatIdStatusBadge.style.border = "1px solid rgba(245, 158, 11, 0.4)";
    chatIdStatusBadge.textContent = "Checking...";
  }
  if (chatIdFeedback) {
    chatIdFeedback.style.display = "block";
    chatIdFeedback.style.background = "rgba(245, 158, 11, 0.08)";
    chatIdFeedback.style.border = "1px solid rgba(245, 158, 11, 0.25)";
    chatIdFeedback.style.color = "var(--text-main)";
    chatIdFeedback.innerHTML = "<span>⏳ Performing internal checks with Telegram Bot API...</span>";
  }

  try {
    const clientId = getClientId();
    const resp = await fetch("/api/telegram/verify", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Client-Id": clientId,
      },
      body: JSON.stringify({ chat_id: rawId }),
    });

    const data = await resp.json().catch(() => ({}));

    if (resp.ok && data.valid) {
      // Internal checks passed!
      const botHandle = data.bot_username ? `@${data.bot_username}` : (data.bot_name || "FallGuard Bot");
      const masked = data.masked_chat_id || maskChatId(rawId);

      if (chatIdStatusBadge) {
        chatIdStatusBadge.style.display = "inline-block";
        chatIdStatusBadge.style.background = "rgba(16, 185, 129, 0.15)";
        chatIdStatusBadge.style.color = "#10b981";
        chatIdStatusBadge.style.border = "1px solid #10b981";
        chatIdStatusBadge.textContent = "✓ Valid Key";
      }

      if (chatIdFeedback) {
        chatIdFeedback.style.display = "block";
        chatIdFeedback.style.background = "rgba(16, 185, 129, 0.08)";
        chatIdFeedback.style.border = "1px solid rgba(16, 185, 129, 0.3)";
        chatIdFeedback.style.color = "var(--text-main)";
        chatIdFeedback.innerHTML = `
          <div style="font-weight: 600; color: #10b981; margin-bottom: 0.2rem;">✓ Token & Chat ID Verified</div>
          <div>Bot: <strong>${escapeHtml(botHandle)}</strong> (${escapeHtml(data.bot_name || "FallGuard")})</div>
          <div>Verification message sent to chat <code>${escapeHtml(masked)}</code>.</div>
          <div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 0.25rem;">
            Emergency fall alerts and system errors will route to this chat ID only.
          </div>
        `;
      }

      currentSettings.chatId = rawId;
      currentSettings.isChatVerified = true;
      saveCurrentSettings();
      sendSettingsOverWs();
      showToast(`Telegram verified! Test message sent to ${masked}.`);
    } else {
      // Internal checks failed!
      const errMsg = data.error || (data.detail || `Verification failed with HTTP ${resp.status}`);
      currentSettings.isChatVerified = false;
      saveCurrentSettings();

      if (chatIdStatusBadge) {
        chatIdStatusBadge.style.display = "inline-block";
        chatIdStatusBadge.style.background = "rgba(239, 68, 68, 0.15)";
        chatIdStatusBadge.style.color = "#ef4444";
        chatIdStatusBadge.style.border = "1px solid #ef4444";
        chatIdStatusBadge.textContent = "✗ Invalid";
      }

      if (chatIdFeedback) {
        chatIdFeedback.style.display = "block";
        chatIdFeedback.style.background = "rgba(239, 68, 68, 0.08)";
        chatIdFeedback.style.border = "1px solid rgba(239, 68, 68, 0.3)";
        chatIdFeedback.style.color = "var(--text-main)";
        chatIdFeedback.innerHTML = `
          <div style="font-weight: 600; color: #ef4444; margin-bottom: 0.2rem;">✗ Internal Check Failed</div>
          <div>${escapeHtml(errMsg)}</div>
        `;
      }

      showToast(`Telegram check failed: ${errMsg}`);
    }
  } catch (err) {
    console.error("[FallGuard] verifyChatId error:", err);
    if (chatIdStatusBadge) {
      chatIdStatusBadge.style.display = "inline-block";
      chatIdStatusBadge.style.background = "rgba(239, 68, 68, 0.15)";
      chatIdStatusBadge.style.color = "#ef4444";
      chatIdStatusBadge.style.border = "1px solid #ef4444";
      chatIdStatusBadge.textContent = "✗ Error";
    }
    if (chatIdFeedback) {
      chatIdFeedback.style.display = "block";
      chatIdFeedback.style.background = "rgba(239, 68, 68, 0.08)";
      chatIdFeedback.style.border = "1px solid rgba(239, 68, 68, 0.3)";
      chatIdFeedback.style.color = "var(--text-main)";
      chatIdFeedback.innerHTML = `<strong>Network/Server Error:</strong> ${escapeHtml(err.message)}`;
    }
  } finally {
    if (btnVerifyChatId) btnVerifyChatId.disabled = false;
    if (btnVerifyChatText) btnVerifyChatText.textContent = "Send";
  }
}

/**
 * Forwards system and monitoring errors directly to the verified Telegram chat ID.
 */
async function reportErrorToTelegram(errorMessage, errorType = "System") {
  const chatId = currentSettings.chatId ? currentSettings.chatId.trim() : "";
  if (!chatId) return;

  try {
    const clientId = getClientId();
    await fetch("/api/telegram/notify-error", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Client-Id": clientId,
      },
      body: JSON.stringify({
        error_message: errorMessage,
        error_type: errorType,
        chat_id: chatId,
      }),
    });
  } catch (err) {
    console.debug("[FallGuard] Failed to forward error to Telegram:", err);
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
// Camera Preference, Mirroring & HTTPS / Wake Lock Helpers
// ============================================================================

function loadCameraPreference() {
  try {
    const saved = localStorage.getItem(CAMERA_PREF_KEY);
    if (saved === "back" || saved === "front") {
      currentFacingMode = saved;
    } else {
      currentFacingMode = "front";
    }
  } catch (e) {
    currentFacingMode = "front";
  }
}

function saveCameraPreference(facing) {
  currentFacingMode = facing;
  try {
    localStorage.setItem(CAMERA_PREF_KEY, facing);
  } catch (e) {
    console.warn("[FallGuard] Could not save camera preference:", e);
  }
  updateCameraUI();
}

function updateCameraUI() {
  if (activeCameraLabel) {
    activeCameraLabel.textContent = currentFacingMode === "back" ? "Back camera" : "Front camera";
  }
  syncMirroringStyles();
}

function syncMirroringStyles() {
  // Mirror preview & overlay skeleton ONLY for front camera (scaleX(-1)), unmirrored for back camera
  const isFront = currentFacingMode === "front";
  if (videoEl) {
    if (isFront) {
      videoEl.classList.add("mirrored");
    } else {
      videoEl.classList.remove("mirrored");
    }
  }
  if (canvasEl) {
    if (isFront) {
      canvasEl.classList.add("mirrored");
    } else {
      canvasEl.classList.remove("mirrored");
    }
  }
}

function checkHttpsRequirement() {
  const isLocalhost =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1" ||
    window.location.hostname === "[::1]";

  if (window.location.protocol !== "https:" && !isLocalhost) {
    if (httpsWarningBanner) {
      httpsWarningBanner.style.display = "flex";
    }
  } else {
    if (httpsWarningBanner) {
      httpsWarningBanner.style.display = "none";
    }
  }
}

async function checkDeviceCount() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    return;
  }
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter((d) => d.kind === "videoinput");

    // If there is only 1 camera (e.g. laptop webcam), hide or disable the button
    if (videoInputs.length <= 1) {
      if (btnSwitchCamera) {
        btnSwitchCamera.disabled = true;
        btnSwitchCamera.style.display = "none";
      }
      if (cameraSwitchWrapper) {
        // Keep activeCameraLabel or hide wrapper if preferred; hiding the button satisfies requirement
      }
    } else {
      if (btnSwitchCamera) {
        btnSwitchCamera.disabled = false;
        btnSwitchCamera.style.display = "inline-flex";
      }
    }

    // Identify preferred back device if available
    inspectBackCameras(videoInputs);
  } catch (err) {
    console.debug("[FallGuard] Device enumeration notice:", err);
  }
}

function inspectBackCameras(videoInputs) {
  if (!videoInputs || videoInputs.length === 0) return;

  const backDevices = videoInputs.filter((d) => {
    const lbl = (d.label || "").toLowerCase();
    return lbl.includes("back") || lbl.includes("rear") || lbl.includes("environment");
  });

  if (backDevices.length > 0) {
    // Prefer main back camera: first one without "wide" or "tele" in label, otherwise the first back device
    const mainBack =
      backDevices.find((d) => {
        const lbl = (d.label || "").toLowerCase();
        return !lbl.includes("wide") && !lbl.includes("tele") && !lbl.includes("ultra");
      }) || backDevices[0];

    if (mainBack && mainBack.deviceId) {
      preferredBackDeviceId = mainBack.deviceId;
    }
  }
}

// Phone Wake Lock & Page Visibility
async function requestScreenWakeLock() {
  if (!("wakeLock" in navigator)) return;
  try {
    if (!wakeLock) {
      wakeLock = await navigator.wakeLock.request("screen");
      wakeLock.addEventListener("release", () => {
        wakeLock = null;
      });
      console.log("[FallGuard] Screen wake lock active.");
    }
  } catch (err) {
    // Fail silently if unsupported or denied
    console.debug("[FallGuard] Wake lock request error (silently handled):", err);
  }
}

async function releaseScreenWakeLock() {
  if (wakeLock) {
    try {
      await wakeLock.release();
    } catch (e) {}
    wakeLock = null;
  }
}

function setupVisibilityAndWakeLockHandlers() {
  document.addEventListener("visibilitychange", async () => {
    if (document.visibilityState === "hidden") {
      if (isMonitoring && visibilityWarningBanner) {
        visibilityWarningBanner.style.display = "flex";
      }
      // Requirement: clear buffer and stop recording on page hide for more than 30 s
      if (pageHideTimeout) clearTimeout(pageHideTimeout);
      pageHideTimeout = setTimeout(() => {
        console.log("[FallGuard] Page hidden > 30s: stopping recorder and clearing buffer.");
        stopSegmentRecorder();
        clearRingBuffer();
        resetIncidentCapture("Page hidden > 30s");
      }, 30000);
    } else if (document.visibilityState === "visible") {
      if (pageHideTimeout) {
        clearTimeout(pageHideTimeout);
        pageHideTimeout = null;
      }
      if (visibilityWarningBanner) {
        visibilityWarningBanner.style.display = "none";
      }
      if (isMonitoring) {
        await requestScreenWakeLock();
        if (currentSettings.sendClip && currentSettings.clipConsent && !isRecordingSegments) {
          startSegmentRecorder();
        }
      }
    }
  });
}

// ============================================================================
// Camera & Monitoring Control
// ============================================================================

async function acquireMediaStream(facing) {
  const targetMode = facing === "back" ? "environment" : "user";

  // Build constraints: Front = facingMode "user", back = facingMode "environment", 640x480 ideal @ 30fps
  // Try { facingMode: { exact: ... } } first, retry with { ideal: ... } on OverconstrainedError
  // If we have a remembered preferredBackDeviceId for back camera, include it as ideal
  let videoConstraintExact = {
    width: { ideal: 640 },
    height: { ideal: 480 },
    frameRate: { ideal: 30 },
    facingMode: { exact: targetMode }
  };

  let videoConstraintIdeal = {
    width: { ideal: 640 },
    height: { ideal: 480 },
    frameRate: { ideal: 30 },
    facingMode: { ideal: targetMode }
  };

  if (facing === "back" && preferredBackDeviceId) {
    videoConstraintIdeal.deviceId = { ideal: preferredBackDeviceId };
  }

  let stream = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: videoConstraintExact,
      audio: false
    });
  } catch (err) {
    if (err.name === "OverconstrainedError" || err.name === "ConstraintNotSatisfiedError") {
      console.warn(`[FallGuard] Exact facingMode (${targetMode}) overconstrained. Retrying with ideal...`);
      stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraintIdeal,
        audio: false
      });
    } else {
      throw err;
    }
  }

  // After permission is granted, enumerate devices to count video inputs and inspect back cameras
  checkDeviceCount();

  return stream;
}

async function startMonitoring() {
  unlockAudioContext();
  initWebSocket();

  try {
    btnStart.disabled = true;
    showLoader(true, `Requesting ${currentFacingMode === "back" ? "back" : "front"} camera (640x480 @ 30fps)...`);

    syncMirroringStyles();

    videoStream = await acquireMediaStream(currentFacingMode);

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

    await requestScreenWakeLock();
    updateClipRecordingState();

    requestAnimationFrame(detectionLoop);
  } catch (err) {
    console.error("[FallGuard] Camera start failed:", err);
    showLoader(false);
    btnStart.disabled = false;
    btnStop.disabled = true;
    handleCameraError(err, currentFacingMode);
  }
}

async function switchCamera() {
  if (isSwitchingCamera) return;
  isSwitchingCamera = true;

  if (btnSwitchCamera) {
    btnSwitchCamera.disabled = true;
  }

  const previousFacing = currentFacingMode;
  const newFacing = previousFacing === "front" ? "back" : "front";

  // If monitoring is NOT running, just change preference and UI
  if (!isMonitoring) {
    saveCameraPreference(newFacing);
    isSwitchingCamera = false;
    if (btnSwitchCamera) btnSwitchCamera.disabled = false;
    return;
  }

  // Monitoring IS running: switch safely
  try {
    // Clear buffer, stop recorder, and reset incident before camera switch
    stopSegmentRecorder();
    clearRingBuffer();
    resetIncidentCapture();

    // 1. Pause detection loop
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }

    showLoader(true, `Switching to ${newFacing === "back" ? "back" : "front"} camera...`);

    // 2. Try acquiring new stream first before destroying old stream
    const newStream = await acquireMediaStream(newFacing);

    // 3. Stop ALL tracks of old stream
    if (videoStream) {
      videoStream.getTracks().forEach((track) => track.stop());
      videoStream = null;
    }

    // 4. Update preference & mirroring
    saveCameraPreference(newFacing);
    videoStream = newStream;
    videoEl.srcObject = newStream;

    // 5. Wait for loadedmetadata
    await new Promise((resolve) => {
      videoEl.onloadedmetadata = () => {
        videoEl.play();
        resolve();
      };
    });

    // 6. Resize overlay canvas to video dimensions
    syncCanvasDimensions();

    // 7. Reset detection state:
    // Close & reopen WebSocket to guarantee backend detector buffer is fresh
    resetWebSocketConnection();

    // Clear last smoothed skeleton & pending possible-fall telemetry in UI
    rawLandmarks = null;
    smoothedLandmarks = null;
    lostFramesCount = 0;
    prevHipY = null;
    prevFrameTime = null;
    resetTelemetryReadouts();

    // Reset pending possible fall status badge (NEVER raise or clear an active alert)
    if (statusBadge && !statusBadge.classList.contains("alert")) {
      updateStatusBadge("normal");
      updateLyingProgressBar(0.0);
    }

    showLoader(false);
    showToast(`Switched to ${newFacing === "back" ? "back" : "front"} camera.`);

    // Restart segment recording with the new stream
    updateClipRecordingState();

    // 8. Resume animation frame loop safely (single loop)
    lastVideoTime = -1;
    requestAnimationFrame(detectionLoop);
  } catch (err) {
    console.error("[FallGuard] Camera switch failed:", err);
    showLoader(false);

    // If the switch fails, go back to previous camera automatically and tell user
    saveCameraPreference(previousFacing);
    showToast(`Camera switch failed: ${err.message || err.name}. Reverting to ${previousFacing} camera.`);

    try {
      // Re-acquire previous camera stream
      const fallbackStream = await acquireMediaStream(previousFacing);
      if (videoStream) {
        videoStream.getTracks().forEach((track) => track.stop());
      }
      videoStream = fallbackStream;
      videoEl.srcObject = fallbackStream;
      await new Promise((resolve) => {
        videoEl.onloadedmetadata = () => {
          videoEl.play();
          resolve();
        };
      });
      syncCanvasDimensions();
      resetWebSocketConnection();
      lastVideoTime = -1;
      updateClipRecordingState();
      requestAnimationFrame(detectionLoop);
    } catch (revertErr) {
      console.error("[FallGuard] Fallback to previous camera also failed:", revertErr);
      stopMonitoring();
      handleCameraError(revertErr, previousFacing);
    }
  } finally {
    isSwitchingCamera = false;
    if (btnSwitchCamera) {
      btnSwitchCamera.disabled = false;
    }
  }
}

/**
 * Resets the WebSocket connection cleanly so the backend starts a fresh FallDetector rolling buffer.
 * Preserves active alerts if currently displaying one.
 */
function resetWebSocketConnection() {
  if (socket) {
    try {
      // Remove onclose auto-reconnect temporarily to control restart
      socket.onclose = null;
      socket.close();
    } catch (e) {}
    socket = null;
  }
  initWebSocket();
}

function resetStandbyState() {
  if (standbyState && originalStandbyHtml) {
    standbyState.innerHTML = originalStandbyHtml;
    standbyState.style.display = "flex";
  }
}

function handleCameraError(err, attemptedFacing = "front") {
  const errName = err.name || "";
  let title = "Unable to Access Camera";
  let explanation = `An error occurred: ${err.message || "Unknown error"}. Please check your camera settings and try again.`;

  if (errName === "NotAllowedError" || errName === "PermissionDeniedError") {
    title = "Camera Permission Denied";
    explanation = "Camera access was blocked by your browser. To allow camera access, open browser site settings (click the lock or camera icon in the address bar), select 'Allow', and click Retry.";
  } else if (errName === "NotFoundError" || errName === "DevicesNotFoundError") {
    title = "Camera Not Found";
    explanation = `The requested ${attemptedFacing === "back" ? "back" : "front"} camera was not found on this device. Ensure your device has this camera and it is recognized by the operating system.`;
  } else if (errName === "NotReadableError" || errName === "TrackStartError") {
    title = "Camera Already in Use";
    explanation = "Your camera is currently in use by another application or browser tab. Please close other software using the camera (such as Zoom, Teams, or another tab) and click Retry.";
  } else if (errName === "OverconstrainedError" || errName === "ConstraintNotSatisfiedError") {
    title = "Camera Constraints Unsupported";
    explanation = `The requested ${attemptedFacing === "back" ? "back" : "front"} camera could not satisfy the required 640x480 resolution or facingMode settings.`;
  }

  // Forward camera failure directly to Telegram
  reportErrorToTelegram(`${title}: ${explanation}`, "Camera");

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
  stopSegmentRecorder();
  clearRingBuffer();
  resetIncidentCapture();
  updateClipUIState();

  releaseScreenWakeLock();

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
  if (visibilityWarningBanner) visibilityWarningBanner.style.display = "none";

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

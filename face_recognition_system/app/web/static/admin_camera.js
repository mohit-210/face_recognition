let sharedStream = null;
const monitorTimers = { register: null, verify: null };
const readyStreak = { register: 0, verify: 0 };
const lockStreak = { register: 0, verify: 0 };
const lastCaptureAt = { register: 0, verify: 0 };
const guidanceInFlight = { register: false, verify: false };
const focusPoint = { register: { x: 0.5, y: 0.5 }, verify: { x: 0.5, y: 0.5 } };
let identifyInFlight = false;

const MONITOR_INTERVAL_MS = 420;
const READY_STREAK_REQUIRED = 1;
const LOCK_STREAK_REQUIRED = 2;
const LOCK_SCORE_THRESHOLD = 0.70;
const CAPTURE_COOLDOWN_MS = 1500;
const VERIFY_IDENTIFY_INTERVAL_MS = 180;
const GUIDANCE_FRAME_MAX_WIDTH = 640;
const GUIDANCE_FRAME_QUALITY = 0.82;
const CAPTURE_FRAME_QUALITY = 0.9;

async function ensureCameraStream() {
  if (sharedStream) return sharedStream;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Camera API not supported in this browser.");
  }
  sharedStream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 960 },
      height: { ideal: 720 },
    },
    audio: false,
  });
  return sharedStream;
}

function getVideoByMode(mode) {
  return document.getElementById(mode === "register" ? "register-video" : "verify-video");
}

function getCanvasByMode(mode) {
  return document.getElementById(mode === "register" ? "register-overlay" : "verify-overlay");
}

function attachStreamToVideo(videoId, stream) {
  const video = document.getElementById(videoId);
  if (!video) return;
  video.srcObject = stream;
  video.play().catch(() => {});
}

function syncOverlaySize(mode) {
  const video = getVideoByMode(mode);
  const canvas = getCanvasByMode(mode);
  if (!video || !canvas) return;

  const width = video.videoWidth || video.clientWidth;
  const height = video.videoHeight || video.clientHeight;
  if (!width || !height) return;
  if (canvas.width === width && canvas.height === height) return;

  canvas.width = width;
  canvas.height = height;
}

function clearOverlay(mode) {
  const canvas = getCanvasByMode(mode);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawGuideFrame(mode) {
  const canvas = getCanvasByMode(mode);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  if (!w || !h) return;

  const guideW = Math.floor(w * 0.44);
  const guideH = Math.floor(h * 0.58);
  const centerX = Math.floor((focusPoint[mode]?.x || 0.5) * w);
  const centerY = Math.floor((focusPoint[mode]?.y || 0.5) * h);
  const x = Math.max(0, Math.min(w - guideW, Math.floor(centerX - guideW / 2)));
  const y = Math.max(0, Math.min(h - guideH, Math.floor(centerY - guideH / 2)));

  ctx.save();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.45)";
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 8]);
  ctx.strokeRect(x, y, guideW, guideH);
  ctx.beginPath();
  ctx.arc(centerX, centerY, 5, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.7)";
  ctx.fill();
  ctx.restore();
}

function mapBboxToCanvas(bbox, frameWidth, frameHeight, canvasWidth, canvasHeight) {
  if (!bbox || bbox.length !== 4) return null;
  const fw = Number(frameWidth || canvasWidth);
  const fh = Number(frameHeight || canvasHeight);
  if (!fw || !fh || !canvasWidth || !canvasHeight) return null;

  const sx = canvasWidth / fw;
  const sy = canvasHeight / fh;
  const x1 = Math.max(0, Math.round(bbox[0] * sx));
  const y1 = Math.max(0, Math.round(bbox[1] * sy));
  const x2 = Math.min(canvasWidth, Math.round(bbox[2] * sx));
  const y2 = Math.min(canvasHeight, Math.round(bbox[3] * sy));
  if (x2 <= x1 || y2 <= y1) return null;
  return [x1, y1, x2, y2];
}

function drawFaceLock(mode, bbox, lockScore, frameWidth = 0, frameHeight = 0) {
  const canvas = getCanvasByMode(mode);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGuideFrame(mode);

  const mapped = mapBboxToCanvas(bbox, frameWidth, frameHeight, canvas.width, canvas.height);
  if (!mapped) return;
  const [x1, y1, x2, y2] = mapped;
  const boxW = Math.max(0, x2 - x1);
  const boxH = Math.max(0, y2 - y1);
  if (!boxW || !boxH) return;

  const locked = lockScore >= LOCK_SCORE_THRESHOLD;
  ctx.save();
  ctx.strokeStyle = locked ? "#16a34a" : "#f59e0b";
  ctx.fillStyle = locked ? "rgba(22, 163, 74, 0.14)" : "rgba(245, 158, 11, 0.14)";
  ctx.lineWidth = 3;
  ctx.setLineDash([]);
  ctx.fillRect(x1, y1, boxW, boxH);
  ctx.strokeRect(x1, y1, boxW, boxH);

  const label = locked ? `LOCK ${Math.round(lockScore * 100)}%` : `TRACK ${Math.round(lockScore * 100)}%`;
  const labelPadding = 8;
  ctx.font = "600 14px 'Segoe UI', Tahoma, sans-serif";
  const labelW = ctx.measureText(label).width + labelPadding * 2;
  const labelH = 24;
  const labelX = x1;
  const labelY = Math.max(0, y1 - labelH - 4);
  ctx.fillStyle = locked ? "rgba(22, 163, 74, 0.92)" : "rgba(180, 83, 9, 0.92)";
  ctx.fillRect(labelX, labelY, labelW, labelH);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, labelX + labelPadding, labelY + 16);
  ctx.restore();
}

function getScaledDimensions(sourceW, sourceH, maxWidth = 0) {
  if (!maxWidth || sourceW <= maxWidth) {
    return { width: sourceW, height: sourceH };
  }
  const scale = maxWidth / sourceW;
  return {
    width: Math.max(1, Math.round(sourceW * scale)),
    height: Math.max(1, Math.round(sourceH * scale)),
  };
}

function captureBlobFromVideo(videoId, options = {}) {
  const video = document.getElementById(videoId);
  if (!video || !video.videoWidth || !video.videoHeight) return null;

  const {
    maxWidth = 0,
    quality = CAPTURE_FRAME_QUALITY,
  } = options;
  const scaled = getScaledDimensions(video.videoWidth, video.videoHeight, maxWidth);
  const canvas = document.createElement("canvas");
  canvas.width = scaled.width;
  canvas.height = scaled.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", quality);
  });
}

function setGuidance(mode, message, isReady = false, lockScore = 0, lockFrames = 0) {
  const el = document.getElementById(`${mode}-guidance`);
  if (!el) return;

  const lockPercent = Math.round((lockScore || 0) * 100);
  const suffix = ` | lock=${lockPercent}% | stable=${lockFrames}/${LOCK_STREAK_REQUIRED}`;
  el.textContent = `Guidance: ${message}${suffix}`;
  el.classList.toggle("ready", Boolean(isReady));
}

function setLockState(mode, label, active = false) {
  const el = document.getElementById(`${mode}-lock-state`);
  if (!el) return;
  el.textContent = `Face lock: ${label}`;
  el.classList.toggle("active", Boolean(active));
}

function setVerifyResult(message, state = "") {
  const el = document.getElementById("verify-result");
  if (!el) return;
  el.textContent = `Result: ${message}`;
  el.classList.remove("ok", "err");
  if (state === "ok") el.classList.add("ok");
  if (state === "err") el.classList.add("err");
}

function setFilesForInput(inputId, newFile, multiple) {
  const input = document.getElementById(inputId);
  if (!input) return;

  const dt = new DataTransfer();
  if (multiple) {
    for (const existing of input.files) {
      dt.items.add(existing);
    }
  }
  dt.items.add(newFile);
  input.files = dt.files;

  if (inputId === "register-images") {
    const count = document.getElementById("register-count");
    if (count) count.textContent = `${input.files.length} frame(s) captured`;
  }
}

async function captureFrameToInput(videoId, inputId, multiple) {
  const blob = await captureBlobFromVideo(videoId, { quality: CAPTURE_FRAME_QUALITY });
  if (!blob) {
    alert("Camera is not ready. Start camera and try again.");
    return false;
  }
  const file = new File([blob], `capture_${Date.now()}.jpg`, { type: "image/jpeg" });
  setFilesForInput(inputId, file, multiple);
  return true;
}

async function identifyFromVideo() {
  if (identifyInFlight) return false;
  identifyInFlight = true;

  try {
    const blob = await captureBlobFromVideo("verify-video", { maxWidth: 640, quality: 0.82 });
    if (!blob) {
      setVerifyResult("camera is not ready", "err");
      return false;
    }

    const form = new FormData();
    form.append("image", new File([blob], `identify_${Date.now()}.jpg`, { type: "image/jpeg" }));
    form.append("device_id", "admin-web-live");
    form.append("fast_mode", "true");

    const res = await fetch("/admin/face/identify", {
      method: "POST",
      body: form,
      credentials: "same-origin",
    });
    if (!res.ok) {
      setVerifyResult("identify request failed", "err");
      return false;
    }

    const data = await res.json();
    if (data.verified && data.name) {
      setVerifyResult(`Verified: ${data.name} (${Math.round((data.confidence || 0) * 100)}%)`, "ok");
      return true;
    }

    const reason = data.reason || "Face not recognized";
    setVerifyResult(reason, "err");
    return false;
  } catch {
    setVerifyResult("unable to contact identify API", "err");
    return false;
  } finally {
    identifyInFlight = false;
  }
}

function evaluateLock(mode, data) {
  const isReady = Boolean(data.ready);
  const lockScore = Number(data.lock_score || 0);
  const isLocked = isReady && lockScore >= LOCK_SCORE_THRESHOLD;

  if (isLocked) {
    lockStreak[mode] += 1;
    readyStreak[mode] += 1;
  } else {
    lockStreak[mode] = 0;
    readyStreak[mode] = isReady ? 1 : 0;
  }
  return { isReady, lockScore, isLocked };
}

async function requestFrameGuidance(mode) {
  if (guidanceInFlight[mode]) return;
  guidanceInFlight[mode] = true;

  const videoId = mode === "register" ? "register-video" : "verify-video";
  syncOverlaySize(mode);

  const blob = await captureBlobFromVideo(videoId, {
    maxWidth: GUIDANCE_FRAME_MAX_WIDTH,
    quality: GUIDANCE_FRAME_QUALITY,
  });
  if (!blob) {
    setGuidance(mode, "Camera is not ready.", false, 0, 0);
    setLockState(mode, "not found", false);
    clearOverlay(mode);
    guidanceInFlight[mode] = false;
    return;
  }

  const form = new FormData();
  form.append("image", new File([blob], `frame_${Date.now()}.jpg`, { type: "image/jpeg" }));
  form.append("target_x", String(focusPoint[mode]?.x ?? 0.5));
  form.append("target_y", String(focusPoint[mode]?.y ?? 0.5));

  try {
    const res = await fetch("/admin/face/frame-guide", {
      method: "POST",
      body: form,
      credentials: "same-origin",
    });

    if (!res.ok) {
      setGuidance(mode, "Frame analysis failed.", false, 0, 0);
      setLockState(mode, "error", false);
      readyStreak[mode] = 0;
      lockStreak[mode] = 0;
      clearOverlay(mode);
      return;
    }

    const data = await res.json();
    const { isReady, lockScore, isLocked } = evaluateLock(mode, data);
    setGuidance(mode, data.guidance || "Monitoring", isLocked, lockScore, lockStreak[mode]);
    setLockState(mode, isLocked ? "LOCKED" : "tracking", isLocked);
    drawFaceLock(mode, data.bbox, lockScore, data.frame_width, data.frame_height);

    if (mode === "verify") return;

    if (!isLocked) return;

    const now = Date.now();
    const cooledDown = now - lastCaptureAt[mode] > CAPTURE_COOLDOWN_MS;
    const stable = lockStreak[mode] >= LOCK_STREAK_REQUIRED && readyStreak[mode] >= READY_STREAK_REQUIRED;
    if (!stable || !cooledDown) return;

    const captured = mode === "register" ? await captureFrameToInput("register-video", "register-images", true) : await identifyFromVideo();

    if (captured) {
      lastCaptureAt[mode] = now;
      readyStreak[mode] = 0;
      lockStreak[mode] = 0;
    }
  } catch {
    setGuidance(mode, "Unable to reach guidance API.", false, 0, 0);
    setLockState(mode, "offline", false);
    readyStreak[mode] = 0;
    lockStreak[mode] = 0;
    clearOverlay(mode);
  } finally {
    guidanceInFlight[mode] = false;
  }
}

function setFocusPointFromEvent(mode, event) {
  const canvas = getCanvasByMode(mode);
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  const point = event.touches && event.touches.length ? event.touches[0] : event;
  const x = (point.clientX - rect.left) / rect.width;
  const y = (point.clientY - rect.top) / rect.height;
  focusPoint[mode] = {
    x: Math.max(0.05, Math.min(0.95, x)),
    y: Math.max(0.05, Math.min(0.95, y)),
  };
  drawGuideFrame(mode);
  setGuidance(mode, "Focus moved. Hold face near tap point.", false, 0, 0);
}

function bindTapFocus() {
  ["register", "verify"].forEach((mode) => {
    const canvas = getCanvasByMode(mode);
    if (!canvas) return;
    canvas.addEventListener("click", (event) => setFocusPointFromEvent(mode, event));
    canvas.addEventListener(
      "touchstart",
      (event) => {
        setFocusPointFromEvent(mode, event);
      },
      { passive: true }
    );
  });
}

async function startMonitoring(mode) {
  const videoId = mode === "register" ? "register-video" : "verify-video";
  try {
    const stream = await ensureCameraStream();
    attachStreamToVideo(videoId, stream);
  } catch (err) {
    setGuidance(mode, `Unable to open camera: ${err.message || err}`, false, 0, 0);
    setLockState(mode, "camera-error", false);
    return;
  }

  stopMonitoring(mode);
  setGuidance(mode, "Monitoring started. Center your face in the guide frame.", false, 0, 0);
  setLockState(mode, "searching", false);
  if (mode === "verify") {
    setVerifyResult("scanning face...");
    setGuidance(mode, "Instant identify mode active.", true, 1, 0);
    setLockState(mode, "instant", true);
  }

  const video = getVideoByMode(mode);
  if (video) {
    video.onloadedmetadata = () => {
      syncOverlaySize(mode);
      drawGuideFrame(mode);
    };
  }
  syncOverlaySize(mode);
  drawGuideFrame(mode);

  if (mode === "verify") {
    monitorTimers[mode] = setInterval(async () => {
      const now = Date.now();
      if (now - lastCaptureAt[mode] < VERIFY_IDENTIFY_INTERVAL_MS) return;
      lastCaptureAt[mode] = now;
      await identifyFromVideo();
    }, VERIFY_IDENTIFY_INTERVAL_MS);
    return;
  }

  monitorTimers[mode] = setInterval(() => {
    requestFrameGuidance(mode);
  }, MONITOR_INTERVAL_MS);
}

function stopMonitoring(mode) {
  if (monitorTimers[mode]) {
    clearInterval(monitorTimers[mode]);
    monitorTimers[mode] = null;
  }
  guidanceInFlight[mode] = false;
  readyStreak[mode] = 0;
  lockStreak[mode] = 0;
  setGuidance(mode, "Monitoring stopped.", false, 0, 0);
  setLockState(mode, "inactive", false);
  clearOverlay(mode);
}

function bindCameraActions() {
  bindTapFocus();
  document.querySelectorAll("[data-camera-start]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const targetVideo = btn.getAttribute("data-camera-start");
      try {
        const stream = await ensureCameraStream();
        attachStreamToVideo(targetVideo, stream);
      } catch (err) {
        alert(`Unable to open camera: ${err.message || err}`);
      }
    });
  });

  document.querySelectorAll("[data-camera-capture]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const videoId = btn.getAttribute("data-camera-capture");
      const inputId = btn.getAttribute("data-target-input");
      const multiple = btn.getAttribute("data-multiple") === "true";
      await captureFrameToInput(videoId, inputId, multiple);
    });
  });

  document.querySelectorAll("[data-monitor-start]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const mode = btn.getAttribute("data-monitor-start");
      await startMonitoring(mode);
    });
  });

  document.querySelectorAll("[data-monitor-stop]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.getAttribute("data-monitor-stop");
      stopMonitoring(mode);
    });
  });

  const identifyNow = document.getElementById("verify-identify-now");
  if (identifyNow) {
    identifyNow.addEventListener("click", async () => {
      await identifyFromVideo();
    });
  }

  window.addEventListener("resize", () => {
    ["register", "verify"].forEach((mode) => {
      syncOverlaySize(mode);
      if (monitorTimers[mode]) drawGuideFrame(mode);
    });
  });
}

document.addEventListener("DOMContentLoaded", bindCameraActions);

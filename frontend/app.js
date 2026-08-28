/* ==========================================================================
   Drishti — frontend/app.js
   Phase 5: phone-facing PWA. Plain JS, no build step, no framework.

   Pipeline this file drives:
     camera --(every ~500ms)--> downscaled JPEG --(WebSocket)--> backend
     backend --(JSON: speech/detections/latency_ms)--> speak() + overlay
                                                      + green bbox draw (NEW)
   ========================================================================== */

// ---- Tunables -------------------------------------------------------------
const FRAME_INTERVAL_MS = 500;   // how often we CAPTURE a frame client-side
const CAPTURE_WIDTH     = 480;   // resize target width in px before encoding
const JPEG_QUALITY      = 0.6;   // 0..1, JPEG compression quality
const WS_PATH           = '/ws/stream';
const BBOX_COLOR        = '#22ff66';  // green, per request -- high-contrast on any background

// ---- DOM references --------------------------------------------------------
const videoEl        = document.getElementById('camera');
const canvasEl       = document.getElementById('captureCanvas');
const ctx            = canvasEl.getContext('2d');
const bboxCanvasEl   = document.getElementById('bboxCanvas');
const bboxCtx        = bboxCanvasEl.getContext('2d');
const startScreenEl  = document.getElementById('startScreen');
const startBtnEl     = document.getElementById('startBtn');
const startErrorEl   = document.getElementById('startError');
const overlayEl      = document.getElementById('overlay');
const statusDotEl    = document.getElementById('statusDot');
const statusTextEl   = document.getElementById('statusText');
const latencyTextEl  = document.getElementById('latencyText');
const lastSpeechEl   = document.getElementById('lastSpeech');
const mockBtnEl      = document.getElementById('mockBtn');

// ---- State ------------------------------------------------------------------
let socket = null;
let captureTimer = null;
let mockTimer = null;
let mockMode = false;
let frameInFlight = false;

// NEW: dimensions (in pixels) of the LAST frame actually encoded and sent
// to the backend. detections[].bbox comes back in this same pixel space
// (the backend runs inference on exactly the frame we sent, unmodified),
// so we need to remember it to correctly scale boxes onto the screen.
let sentFrameW = 0;
let sentFrameH = 0;

// ============================================================================
// 1. Start flow — camera permission + first user gesture
// ============================================================================

startBtnEl.addEventListener('click', async () => {
  startBtnEl.disabled = true;
  try {
    await initCamera();
    speak('Drishti ready.');
    startScreenEl.hidden = true;
    overlayEl.hidden = false;
    resizeBboxCanvas();
    connectSocket();
  } catch (err) {
    startBtnEl.disabled = false;
    startErrorEl.hidden = false;
    startErrorEl.textContent = describeMediaError(err);
  }
});

async function initCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment' },
    audio: false,
  });
  videoEl.srcObject = stream;
  await videoEl.play();
}

function describeMediaError(err) {
  if (err && err.name === 'NotAllowedError') {
    return 'Camera permission was denied. Allow camera access in your browser settings and reload.';
  }
  if (err && err.name === 'NotFoundError') {
    return 'No camera was found on this device.';
  }
  if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
    return 'Camera requires HTTPS or localhost. This page is neither — see Phase 8 for the fix.';
  }
  return 'Could not start the camera: ' + (err && err.message ? err.message : String(err));
}

// ============================================================================
// 2. WebSocket connection to the backend (Phase 4's /ws/stream)
// ============================================================================

function connectSocket() {
  if (mockMode) return;

  const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${wsProtocol}//${location.host}${WS_PATH}`;

  setStatus('connecting', 'Connecting…');
  socket = new WebSocket(url);

  socket.addEventListener('open', () => {
    setStatus('connected', 'Live');
    startCaptureLoop();
  });

  socket.addEventListener('message', (event) => {
    frameInFlight = false;
    handleServerMessage(event.data);
  });

  socket.addEventListener('close', () => {
    setStatus('disconnected', 'Disconnected — retrying…');
    stopCaptureLoop();
    clearBoundingBoxes(); // NEW: don't leave stale boxes on screen once we've lost the feed
    if (!mockMode) {
      setTimeout(connectSocket, 1500);
    }
  });

  socket.addEventListener('error', () => {
    // 'error' is always followed by 'close' per the WebSocket spec.
  });
}

function handleServerMessage(raw) {
  let data;
  try {
    data = JSON.parse(raw);
  } catch (e) {
    console.error('Malformed message from backend:', raw);
    return;
  }

  if (typeof data.latency_ms === 'number') {
    latencyTextEl.textContent = `${Math.round(data.latency_ms)} ms`;
  }

  if (data.speech) {
    lastSpeechEl.textContent = data.speech;
    speak(data.speech);
  }

  // NEW: draw (or clear) green boxes every response, independent of
  // whether this particular frame produced new speech -- detections is
  // sent on every frame, speech only on debounce-approved frames.
  drawBoundingBoxes(Array.isArray(data.detections) ? data.detections : []);
}

// ============================================================================
// 3. Frame capture loop
// ============================================================================

function startCaptureLoop() {
  stopCaptureLoop();
  captureTimer = setInterval(sendFrame, FRAME_INTERVAL_MS);
}

function stopCaptureLoop() {
  if (captureTimer) clearInterval(captureTimer);
  captureTimer = null;
}

function sendFrame() {
  if (mockMode) return;
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  if (frameInFlight) return;
  if (!videoEl.videoWidth) return;

  const base64Jpeg = captureFrameAsBase64();
  if (!base64Jpeg) return;

  frameInFlight = true;
  socket.send(base64Jpeg);
}

function captureFrameAsBase64() {
  const videoW = videoEl.videoWidth;
  const videoH = videoEl.videoHeight;
  if (!videoW || !videoH) return null;

  const scale = CAPTURE_WIDTH / videoW;
  const targetW = CAPTURE_WIDTH;
  const targetH = Math.round(videoH * scale);

  if (canvasEl.width !== targetW || canvasEl.height !== targetH) {
    canvasEl.width = targetW;
    canvasEl.height = targetH;
  }

  ctx.drawImage(videoEl, 0, 0, targetW, targetH);

  // NEW: remember exactly what we encoded -- this is the pixel space
  // detections[].bbox will come back in, since the backend performs no
  // further resizing of what we send it.
  sentFrameW = targetW;
  sentFrameH = targetH;

  const dataUrl = canvasEl.toDataURL('image/jpeg', JPEG_QUALITY);
  const commaIndex = dataUrl.indexOf(',');
  return commaIndex === -1 ? null : dataUrl.slice(commaIndex + 1);
}

// ============================================================================
// 4. Speech output
// ============================================================================

function speak(text) {
  if (!('speechSynthesis' in window)) {
    console.warn('SpeechSynthesis not supported in this browser.');
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.0;
  window.speechSynthesis.speak(utterance);
}

// ============================================================================
// 5. NEW — Bounding-box overlay
// ============================================================================
// Three-stage coordinate transform, because the video element uses
// object-fit:cover (fills the screen, cropping overflow) while detections
// come back in the pixel space of the small downscaled frame we actually
// sent (sentFrameW x sentFrameH), not the screen's pixel space:
//
//   1. bbox pixel (in sent-frame space)  -> normalized fraction (0..1)
//   2. normalized fraction               -> rendered-video pixel
//   3. rendered-video pixel              -> screen pixel (cover offset)
//
// Steps 2+3 collapse into one scale+offset because the sent frame is
// downscaled WITHOUT changing aspect ratio, so it shares the same aspect
// ratio as the raw camera feed and therefore the same aspect ratio as
// what's actually rendered on screen under object-fit:cover.

function resizeBboxCanvas() {
  bboxCanvasEl.width = window.innerWidth;
  bboxCanvasEl.height = window.innerHeight;
}
window.addEventListener('resize', resizeBboxCanvas);

// Computes where the VIDEO CONTENT is actually drawn on screen -- i.e. the
// visible rectangle after object-fit:cover has scaled-to-fill and cropped
// the overflow, centered both axes (the CSS default for object-position).
function computeVideoRenderRect() {
  const videoW = videoEl.videoWidth;
  const videoH = videoEl.videoHeight;
  const boxW = bboxCanvasEl.width;
  const boxH = bboxCanvasEl.height;
  if (!videoW || !videoH || !boxW || !boxH) return null;

  const scale = Math.max(boxW / videoW, boxH / videoH); // "cover" = scale to fill, crop overflow
  const renderedW = videoW * scale;
  const renderedH = videoH * scale;
  const offsetX = (boxW - renderedW) / 2;  // video is centered by default
  const offsetY = (boxH - renderedH) / 2;
  return { renderedW, renderedH, offsetX, offsetY };
}

function clearBoundingBoxes() {
  bboxCtx.clearRect(0, 0, bboxCanvasEl.width, bboxCanvasEl.height);
}

function drawBoundingBoxes(detections) {
  clearBoundingBoxes();
  if (!detections.length || !sentFrameW || !sentFrameH) return;

  const rect = computeVideoRenderRect();
  if (!rect) return;

  // Sent-frame pixel -> rendered-video pixel is a single scale factor,
  // since sentFrame and the rendered video share the same aspect ratio
  // (see comment above).
  const scaleX = rect.renderedW / sentFrameW;
  const scaleY = rect.renderedH / sentFrameH;

  bboxCtx.lineWidth = 3;
  bboxCtx.strokeStyle = BBOX_COLOR;
  bboxCtx.fillStyle = BBOX_COLOR;
  bboxCtx.font = '600 15px system-ui, sans-serif';
  bboxCtx.textBaseline = 'bottom';

  for (const det of detections) {
    if (!Array.isArray(det.bbox) || det.bbox.length !== 4) continue;
    const [x1, y1, x2, y2] = det.bbox;

    const sx = rect.offsetX + x1 * scaleX;
    const sy = rect.offsetY + y1 * scaleY;
    const sw = (x2 - x1) * scaleX;
    const sh = (y2 - y1) * scaleY;

    bboxCtx.strokeRect(sx, sy, sw, sh);

    // Label: "class_name  distance_m" when available (matches the JSON
    // shape returned by backend/main.py's detections_out).
    const label = det.distance_m != null
      ? `${det.class_name} ${det.distance_m.toFixed(1)}m`
      : det.class_name;
    const textW = bboxCtx.measureText(label).width;
    const labelY = sy > 20 ? sy : sy + sh + 18;

    bboxCtx.fillRect(sx - 1, labelY - 17, textW + 8, 19);
    bboxCtx.fillStyle = '#001a08';
    bboxCtx.fillText(label, sx + 3, labelY);
    bboxCtx.fillStyle = BBOX_COLOR;
  }
}

// ============================================================================
// 6. Mock mode — test speech + overlay without the backend running at all
// ============================================================================

const MOCK_SENTENCES = [
  { speech: 'Person very close, center.', bboxFrac: [0.38, 0.30, 0.62, 0.95], class_name: 'person', distance_m: 0.8 },
  { speech: 'Car 4.2 meters, right.', bboxFrac: [0.68, 0.40, 0.95, 0.68], class_name: 'car', distance_m: 4.2 },
  { speech: 'Bicycle 2.1 meters, left.', bboxFrac: [0.05, 0.42, 0.32, 0.80], class_name: 'bicycle', distance_m: 2.1 },
  { speech: 'Chair 1.3 meters, center.', bboxFrac: [0.40, 0.48, 0.60, 0.90], class_name: 'chair', distance_m: 1.3 },
  { speech: 'Dog very close, left.', bboxFrac: [0.04, 0.58, 0.30, 0.90], class_name: 'dog', distance_m: 0.6 },
];
let mockIndex = 0;

mockBtnEl.addEventListener('click', () => {
  mockMode = !mockMode;
  mockBtnEl.textContent = mockMode ? 'Mock: ON' : 'Mock: OFF';
  mockBtnEl.classList.toggle('active', mockMode);

  if (mockMode) {
    stopCaptureLoop();
    if (socket) {
      socket.onclose = null;
      socket.close();
      socket = null;
    }
    setStatus('mock', 'Mock mode');
    // NEW: mock detections are expressed in a pretend sent-frame that
    // matches the REAL camera's aspect ratio (falling back to 4:3 only if
    // the camera hasn't reported its dimensions yet) -- otherwise the
    // demo boxes would be scaled using the wrong aspect ratio and drift
    // from where they visually should sit, even though the transform
    // math itself is correct for the real pipeline.
    sentFrameW = 480;
    sentFrameH = videoEl.videoWidth
      ? Math.round(480 * (videoEl.videoHeight / videoEl.videoWidth))
      : 360;
    startMockResponses();
  } else {
    stopMockResponses();
    clearBoundingBoxes();
    connectSocket();
  }
});

function startMockResponses() {
  stopMockResponses();
  mockTimer = setInterval(() => {
    const item = MOCK_SENTENCES[mockIndex % MOCK_SENTENCES.length];
    mockIndex++;
    const fakeLatency = 300 + Math.round(Math.random() * 400);
    // Convert this mock item's fractional bbox into pixel coords using
    // whatever sentFrameW/H currently are -- computed fresh each tick so
    // it stays correct even if the camera dimensions weren't ready the
    // instant Mock mode was toggled on.
    const [fx1, fy1, fx2, fy2] = item.bboxFrac;
    const bbox = [
      Math.round(fx1 * sentFrameW), Math.round(fy1 * sentFrameH),
      Math.round(fx2 * sentFrameW), Math.round(fy2 * sentFrameH),
    ];
    handleServerMessage(JSON.stringify({
      speech: item.speech,
      detections: [{ class_name: item.class_name, distance_m: item.distance_m, bbox }],
      latency_ms: fakeLatency,
    }));
  }, 2500);
}

function stopMockResponses() {
  if (mockTimer) clearInterval(mockTimer);
  mockTimer = null;
}

// ============================================================================
// 7. Status overlay helper
// ============================================================================

function setStatus(state, text) {
  statusTextEl.textContent = text;
  statusDotEl.classList.remove('connected', 'mock');
  if (state === 'connected') statusDotEl.classList.add('connected');
  if (state === 'mock') statusDotEl.classList.add('mock');
}

// ============================================================================
// 8. Service worker registration (installable PWA)
// ============================================================================

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('sw.js').catch((err) => {
      console.warn('Service worker registration failed:', err);
    });
  });
}

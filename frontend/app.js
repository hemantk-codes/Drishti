/* ==========================================================================
   Drishti — frontend/app.js
   Phase 5: phone-facing PWA. Plain JS, no build step, no framework.

   Pipeline this file drives:
     camera --(every ~500ms)--> downscaled JPEG --(WebSocket)--> backend
     backend --(JSON: speech/detections/latency_ms)--> speak() + overlay

   Everything here is commented for viva defense, not just for maintenance:
   each constant/decision has a "why", because "it just works" won't survive
   a follow-up question from an evaluator.
   ========================================================================== */

// ---- Tunables -------------------------------------------------------------
// These match Section 6 / Phase 5's spec. Pulled to the top so they're easy
// to retune for a demo without hunting through the file (config.yaml on the
// backend does the equivalent job server-side, from Phase 6 onward).
const FRAME_INTERVAL_MS = 500;   // how often we CAPTURE a frame client-side
const CAPTURE_WIDTH     = 480;   // resize target width in px before encoding
const JPEG_QUALITY      = 0.6;   // 0..1, JPEG compression quality
const WS_PATH           = '/ws/stream';

// ---- DOM references --------------------------------------------------------
const videoEl        = document.getElementById('camera');
const canvasEl       = document.getElementById('captureCanvas');
const ctx            = canvasEl.getContext('2d');
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
let frameInFlight = false;   // true while we're waiting on a response for a
                              // frame we already sent. Prevents the frontend
                              // from piling up sends on a slow/laggy link —
                              // the backend already drops stale frames
                              // (Phase 4), this is the client-side half of
                              // that same "always prefer the latest reality"
                              // principle.

// ============================================================================
// 1. Start flow — camera permission + first user gesture
// ============================================================================

startBtnEl.addEventListener('click', async () => {
  startBtnEl.disabled = true;
  try {
    await initCamera();

    // Speak once immediately, using the SAME gesture that unlocked speech.
    // This does two jobs at once: (a) proves to the user — who may not be
    // able to see the screen at all — that the system is actually live,
    // and (b) "warms up" the SpeechSynthesis engine on browsers that lazily
    // initialize voices on first use.
    speak('Drishti ready.');

    startScreenEl.hidden = true;
    overlayEl.hidden = false;

    connectSocket();
  } catch (err) {
    startBtnEl.disabled = false;
    startErrorEl.hidden = false;
    startErrorEl.textContent = describeMediaError(err);
  }
});

async function initCamera() {
  // facingMode 'environment' = rear camera, i.e. pointed at the world,
  // not a front-facing selfie camera. Per BUILD_PHASES.md this is a hard
  // requirement, not a default we'd otherwise leave unset.
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment' },
    audio: false,
  });
  videoEl.srcObject = stream;
  await videoEl.play();
}

function describeMediaError(err) {
  // getUserMedia rejects with a small, well-known set of DOMException
  // names — surfacing the real one saves a debugging round-trip versus a
  // generic "camera failed" message.
  if (err && err.name === 'NotAllowedError') {
    return 'Camera permission was denied. Allow camera access in your browser settings and reload.';
  }
  if (err && err.name === 'NotFoundError') {
    return 'No camera was found on this device.';
  }
  if (location.protocol !== 'https:' && location.hostname !== 'localhost') {
    // The single most common Phase 5 dead-end: camera silently blocked
    // because we're not in a secure context. Surface it explicitly instead
    // of leaving the person to guess (see Section 6, PROJECT_CONTEXT.md).
    return 'Camera requires HTTPS or localhost. This page is neither — see Phase 8 for the fix.';
  }
  return 'Could not start the camera: ' + (err && err.message ? err.message : String(err));
}

// ============================================================================
// 2. WebSocket connection to the backend (Phase 4's /ws/stream)
// ============================================================================

function connectSocket() {
  if (mockMode) return; // mock mode never touches the real socket

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
    if (!mockMode) {
      // Simple fixed-delay reconnect. Good enough for a demo; Phase 8's
      // fallback-UI requirement can build on top of this same status text.
      setTimeout(connectSocket, 1500);
    }
  });

  socket.addEventListener('error', () => {
    // 'error' is always followed by 'close' per the WebSocket spec, so the
    // reconnect logic above already covers this — this handler just avoids
    // an unhandled-error console warning.
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

  // speech is null on frames where the fusion engine's debounce (Phase 3)
  // decided nothing new is worth announcing — that's expected and NOT an
  // error, so we simply leave the last spoken sentence on screen.
  if (data.speech) {
    lastSpeechEl.textContent = data.speech;
    speak(data.speech);
  }
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
  if (frameInFlight) return; // still waiting on the previous frame — skip
  if (!videoEl.videoWidth) return; // video metadata not ready yet

  const base64Jpeg = captureFrameAsBase64();
  if (!base64Jpeg) return;

  frameInFlight = true;

  // --------------------------------------------------------------------
  // ASSUMPTION TO VERIFY AGAINST YOUR PHASE 4 CODE:
  // BUILD_PHASES.md's Phase 4 wording is "receives a base64-encoded JPEG
  // frame" — read literally, so this sends the raw base64 STRING as the
  // WebSocket message body (no JSON wrapper, no data-URI prefix).
  //
  // If your backend/main.py instead expects a JSON envelope, e.g.
  //   { "frame": "<base64>" }
  // then change ONLY the line below to:
  //   socket.send(JSON.stringify({ frame: base64Jpeg }));
  // Everything else in this file is unaffected either way.
  // --------------------------------------------------------------------
  socket.send(base64Jpeg);
}

function captureFrameAsBase64() {
  const videoW = videoEl.videoWidth;
  const videoH = videoEl.videoHeight;
  if (!videoW || !videoH) return null;

  // Downscale to CAPTURE_WIDTH, preserving aspect ratio. Smaller frame =
  // less to encode, less to send over WiFi, less for the backend to run
  // inference on — this matters a lot more on CPU-only than it would with
  // a GPU behind it.
  const scale = CAPTURE_WIDTH / videoW;
  const targetW = CAPTURE_WIDTH;
  const targetH = Math.round(videoH * scale);

  if (canvasEl.width !== targetW || canvasEl.height !== targetH) {
    canvasEl.width = targetW;
    canvasEl.height = targetH;
  }

  ctx.drawImage(videoEl, 0, 0, targetW, targetH);

  // toDataURL gives us "data:image/jpeg;base64,<data>" — strip the prefix
  // since the backend only wants the raw base64 payload.
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
  // Cancel whatever is currently being spoken FIRST. Without this, if
  // announcements arrive faster than they can be spoken, they queue up and
  // the user ends up hearing stale, backed-up sentences describing a scene
  // that's no longer in front of them — actively dangerous for a mobility
  // aid. We always want the newest announcement, spoken immediately.
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.0;
  window.speechSynthesis.speak(utterance);
}

// ============================================================================
// 5. Mock mode — test speech + overlay without the backend running at all
// ============================================================================
// Toggled entirely client-side. Useful for: iterating on overlay styling,
// confirming SpeechSynthesis behaves correctly on a given phone/browser, or
// demoing the frontend before Phase 4's backend is reachable — none of that
// should require a working model pipeline.

const MOCK_SENTENCES = [
  'Person very close, center.',
  'Car 4.2 meters, right.',
  'Bicycle 2.1 meters, left.',
  'Chair 1.3 meters, center.',
  'Dog very close, left.',
];
let mockIndex = 0;

mockBtnEl.addEventListener('click', () => {
  mockMode = !mockMode;
  mockBtnEl.textContent = mockMode ? 'Mock: ON' : 'Mock: OFF';
  mockBtnEl.classList.toggle('active', mockMode);

  if (mockMode) {
    // Tear down any real connection so mock and live data can't interleave.
    stopCaptureLoop();
    if (socket) {
      socket.onclose = null; // don't trigger the real reconnect logic
      socket.close();
      socket = null;
    }
    setStatus('mock', 'Mock mode');
    startMockResponses();
  } else {
    stopMockResponses();
    connectSocket();
  }
});

function startMockResponses() {
  stopMockResponses();
  mockTimer = setInterval(() => {
    const sentence = MOCK_SENTENCES[mockIndex % MOCK_SENTENCES.length];
    mockIndex++;
    const fakeLatency = 300 + Math.round(Math.random() * 400);
    handleServerMessage(JSON.stringify({
      speech: sentence,
      detections: [],
      latency_ms: fakeLatency,
    }));
  }, 2500); // slower than the real 500ms capture cadence, to mimic the
            // debounce meaning "not every frame produces new speech"
}

function stopMockResponses() {
  if (mockTimer) clearInterval(mockTimer);
  mockTimer = null;
}

// ============================================================================
// 6. Status overlay helper
// ============================================================================

function setStatus(state, text) {
  statusTextEl.textContent = text;
  statusDotEl.classList.remove('connected', 'mock');
  if (state === 'connected') statusDotEl.classList.add('connected');
  if (state === 'mock') statusDotEl.classList.add('mock');
}

// ============================================================================
// 7. Service worker registration (installable PWA)
// ============================================================================

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('sw.js').catch((err) => {
      console.warn('Service worker registration failed:', err);
    });
  });
}

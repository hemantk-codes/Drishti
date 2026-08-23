/* ==========================================================================
   Drishti — frontend/sw.js
   Minimal service worker. Its ONLY job in Phase 5 is to cache the static
   "app shell" (HTML/JS/icons) so the browser considers this installable as
   a PWA and it opens instantly on repeat visits.

   It deliberately does NOT try to cache or intercept:
     - the /ws/stream WebSocket (service workers can't intercept WS anyway)
     - the /health or any future API routes
   Real offline operation of the AI pipeline itself is out of scope (see
   PROJECT_CONTEXT.md Section 8) — this is shell caching, not offline AI.
   ========================================================================== */

const CACHE_NAME = 'drishti-shell-v1';
const SHELL_FILES = [
  '/',
  '/index.html',
  '/app.js',
  '/manifest.json',
  '/icon-192.png',
  '/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_FILES))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  // Only ever handle simple GETs for shell files. Everything else
  // (WebSocket upgrade requests, POSTs, etc.) passes straight through
  // untouched.
  if (event.request.method !== 'GET') return;

  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});

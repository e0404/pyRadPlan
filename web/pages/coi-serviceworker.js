/*
 * Cross-origin isolation for hosts that cannot set response headers (GitHub Pages).
 * One file, two roles: loaded as a classic <script> in index.html it registers itself as a
 * service worker and reloads the page once; running as the service worker it re-serves every
 * response of its scope with COOP/COEP so that SharedArrayBuffer (Pyodide's interrupt buffer,
 * the Stop button) becomes available. COEP is require-corp: same-origin files and the jsDelivr
 * package downloads (CORS + Cross-Origin-Resource-Policy: cross-origin) qualify.
 * Idea and structure follow coi-serviceworker (MIT, Guido Zuidhof and contributors).
 */
if (typeof window === "undefined") {
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));
  self.addEventListener("fetch", (event) => {
    const request = event.request;
    if (request.cache === "only-if-cached" && request.mode !== "same-origin") return;
    event.respondWith(
      fetch(request).then((response) => {
        if (response.status === 0) return response;
        const headers = new Headers(response.headers);
        headers.set("Cross-Origin-Opener-Policy", "same-origin");
        headers.set("Cross-Origin-Embedder-Policy", "require-corp");
        headers.set("Cross-Origin-Resource-Policy", "cross-origin");
        return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
      }),
    );
  });
} else if (window.crossOriginIsolated) {
  // Isolated (by this worker or by the server): re-arm the one-reload guard for a later forced reload.
  try {
    sessionStorage.removeItem(`pyradplan-coi-reload:${new URL(".", document.currentScript.src).pathname}`);
  } catch {
    // storage blocked: nothing to re-arm
  }
} else if (window.crossOriginIsolated === false && window.isSecureContext && navigator.serviceWorker) {
  // Per scope: deployments side by side on one origin (/latest/, /develop/) each get their reload.
  const KEY = `pyradplan-coi-reload:${new URL(".", document.currentScript.src).pathname}`;
  const reload = () => {
    // One reload per tab session: if isolation still fails afterwards, run without it instead of looping.
    try {
      if (sessionStorage.getItem(KEY)) return;
      sessionStorage.setItem(KEY, "1");
    } catch {
      return;
    }
    window.location.reload();
  };
  navigator.serviceWorker.register(document.currentScript.src).then(
    (registration) => {
      // First visit: the worker takes control (clients.claim) once installed, then reload.
      navigator.serviceWorker.addEventListener("controllerchange", reload);
      // Already active but not controlling (forced reload, Shift+F5): reload to be controlled.
      if (registration.active && !navigator.serviceWorker.controller) reload();
    },
    (err) => console.warn("Cross-origin isolation service worker failed to register:", err),
  );
}

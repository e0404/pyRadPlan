# Publishing the web version (decision record)

Status 2026-11-18, branch `internal/web-pyodide`. Question: can the browser version of pyRadPlan
(`docs/development/web_prototype.md`) be hosted on GitHub Pages? Nothing was published, no
repository or Pages setting was touched; everything below was tested locally against a static
server that mimics Pages (see "What was tested").

## Result

**Yes.** The production build runs from a sub-path on a host that sends no COOP/COEP headers, and
with a small service worker it is cross-origin isolated again, so the Stop button works.

- Sub-path hosting works with Vite's `base`. A relative base (`--base=./`) makes one build work
  under `/<repo>/latest/`, `/<repo>/v0.10/`, any deeper path and a custom domain.
- Without any workaround the app boots and runs the TG119 pipeline; only Stop is unavailable
  (disabled, with a tooltip). That degradation is already implemented.
- With `web/pages/coi-serviceworker.js` (about 50 lines, first visit reloads once) the page is
  `crossOriginIsolated`, `SharedArrayBuffer` exists and Stop interrupts a dose calculation in
  60-70 ms, as with real headers.
- Size is not a problem for Pages: 50.7 MB site (42.9 MB gzipped), limit 1 GB. The Pyodide
  packages (numpy, scipy, ...: 20.9 MB) can stay on jsDelivr.
- The pyRadPlan repository does not use Pages today (docs are on Read the Docs), so
  `https://e0404.github.io/pyRadPlan/` is free. Versioned folders (`/latest/`, `/v0.10/`) are the
  awkward part: a Pages deployment replaces the whole site.

## What was tested

Scratch copy of `web/` (no files in the worktree touched), production build, served by a Node
static server that sends what GitHub Pages sends (no COOP/COEP, `Access-Control-Allow-Origin: *`,
`Cache-Control: max-age=600`, ETag, `.wasm` as `application/wasm`, `.mjs` as `text/javascript`,
`.whl` as `application/octet-stream`; checked against live github.io sites) under a prefix, driven by
Playwright with the installed Edge. Each run boots the real Pyodide backend, loads TG119, runs
Calc. Dose Influence and Optimize (default plan), and where isolated also Stop on a 9-beam plan.

| Run | Build | Served at | Isolated | Result |
|---|---|---|---|---|
| 1 | `--base=/pyRadPlan/app/`, no service worker | `/pyRadPlan/app/`, no headers | no | boots, pipeline completes, Stop disabled with tooltip, no console errors |
| 2 | `--base=/pyRadPlan/app/`, with service worker | `/pyRadPlan/app/`, no headers | yes (after one reload at 0.2 s) | pipeline completes, Stop enabled while busy, stops in 61 ms, no errors |
| 3 | `--base=./`, with service worker | `/pyRadPlan/v0.10/` | yes | same, Stop 73 ms |
| 4 | `--base=./`, with service worker | `/x/y/z/` | yes | boots |
| 5 | `--base=./`, with service worker | server sends COOP/COEP itself | yes, service worker not registered | boots (the script defers to real headers) |
| 6 | `--base=./`, warm second visit, persistent profile | as 2 | yes | 3 server requests instead of 36, boot 9.9 s instead of 12.0 s |

Boot to a usable session is 10-14 s and the default-plan dose influence + optimization 8-11 s on
this machine when it is idle, as with the dev server. Other work on the machine made single runs
3-5 times slower in every mode (service worker, server headers, no isolation: 10 s and 48 s both
occurred in each), so there is no measurable speed difference from isolation or from the worker;
treat timings as indicative. After the last change to the service worker, the two smoke runs of
`web/e2e/pages.mjs` (with worker plus Stop, and without worker, `--no-isolation`) passed. Not run:
Firefox, Safari, mobile, a real github.io origin.

## Feasibility on GitHub Pages

### Sub-path (`https://<org>.github.io/<repo>/`)

The frontend builds all its URLs from `import.meta.env.BASE_URL` (worker via `new URL(..,
import.meta.url)`, `pyodide-runtime/`, `pyrp/manifest.json`, wheel URLs, assets), so it needs no
code change. Verified with an absolute base (`/pyRadPlan/app/`) and a relative one (`./`). The
relative base is preferable: the artifact does not know where it lives. Two details:

- On Windows with Git Bash, MSYS rewrites `--base=/x/` into `C:/Program Files/Git/x/`; set
  `MSYS_NO_PATHCONV=1` or use PowerShell. Linux CI is unaffected.
- With a relative base the URL must end in a slash (`.../latest/`). Pages redirects a directory
  URL without it (assumed, not tested against github.io).

There is no client-side routing (no `pushState`, no hash routes; state is in the page and URL query
parameters), so the SPA-404 problem does not exist and no `404.html` trick is needed. The app uses
no `localStorage`, cookies or IndexedDB, which matters because all project sites of one
organization share the origin `<org>.github.io`.

### Cross-origin isolation

Pages cannot set headers, and `SharedArrayBuffer` needs COOP `same-origin` + COEP. It is used for
one thing: the Pyodide interrupt buffer behind Stop.

| Option | Stop | Cost | Verdict |
|---|---|---|---|
| (a) Do nothing | Disabled (tooltip "Stopping needs SharedArrayBuffer"); a long run cannot be cancelled, reload loses the session | none; verified (run 1) | acceptable fallback, not as the default |
| (b) Service worker adds COOP/COEP | Works, 60-70 ms (runs 2, 3) | one reload on the first visit; needs HTTPS or localhost; not available in private windows of some browsers (then falls back to (a)); the worker proxies every response of its scope | recommended |
| (c) Stop = terminate and restart the worker | Works everywhere | about 10-13 s boot and the whole workspace is lost (it lives in Python memory); needs a confirmation and ideally an auto-saved pickle to restore | possible extra fallback, not implemented |
| (d) Cooperative polling without SAB | Not feasible alone: the worker is blocked inside Python and cannot receive messages; a synchronous XHR to a service worker would work, but that needs a service worker anyway | | rejected |

Details of (b), `web/pages/coi-serviceworker.js` (modelled on the MIT-licensed
[coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker), rewritten smaller):

- The same file is the classic `<script>` that registers itself and the worker. It re-serves every
  response in its scope with COOP `same-origin`, COEP `require-corp`, CORP `cross-origin`. Scope is
  the directory of the file, so `/latest/` and `/v0.10/` get independent workers.
- **`require-corp` vs `credentialless`.** The only cross-origin downloads are the jsDelivr package
  wheels, fetched in CORS mode, and jsDelivr answers with `Access-Control-Allow-Origin: *` and
  `Cross-Origin-Resource-Policy: cross-origin` (checked with curl, and all 16 loaded in runs 2-5).
  So the stricter `require-corp` works and no browser needs `credentialless` support. If a
  cross-origin resource without CORP/CORS is ever added (fonts, images, badges), it breaks under
  COEP; vendoring it fixes that.
- **Reload loop guard.** The script reloads once per tab session after the worker takes control;
  if isolation still fails it stays unisolated instead of looping. The guard is re-armed when a
  page is isolated, so a forced reload (Shift+F5, which bypasses the worker) reloads once more.
- First visit shows the splash, reloads after 0.2 s, and starts booting; the aborted first request
  is harmless (`net::ERR_ABORTED`, filtered in `e2e/pages.mjs`).
- Build wiring: an additive plugin in `web/vite.config.ts`, active only with `PYRADPLAN_COI=1`
  (injects the script tag with the build's base, emits the file). The default build, the dev server
  and `npm run preview` are unchanged. Servers that send the headers (nginx, Cloudflare, Netlify)
  should build without it.
- Not verified: Firefox, Safari (isolation with a service worker, private browsing, Safari
  evicting service workers after seven days without use, which would only cost one extra reload).

### Size and limits

Measured on `dist/` (34 files, base-independent):

| Part | Size | gzipped | Note |
|---|---|---|---|
| `pyrp/pyradplan-*.whl` | 32.4 MB | 32.4 MB | about 31 MB of it is bundled `.mat` base data; does not compress |
| other `pyrp/` wheels (shim + 14 pure) | 4.0 MB | | pydicom 2.4 MB |
| `pyodide-runtime/` | 13.5 MB | 6.3 MB | `pyodide.asm.wasm` 9.6 MB (3.5 MB gz), stdlib zip 2.5 MB |
| app (JS, CSS, logos) | 0.73 MB | 0.23 MB | main chunk 0.58 MB |
| **site** | **50.7 MB** | **42.9 MB** | |
| jsDelivr, at run time | 20.9 MB (16 files) | | scipy 13.9, numpy 2.9, h5py 1.6, pydantic-core 1.3 |

A cold first visit transfers about 64-72 MB (43 MB from Pages with gzip, which Pages applies to
`.wasm`, `.mjs`, `.json` and, observed, `.whl`; 21 MB from jsDelivr).

GitHub's documented limits ([Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)):
published site at most 1 GB; source repository 1 GB recommended; deployment times out after 10
minutes; soft bandwidth limit 100 GB per month; soft limit of 10 builds per hour (does not apply
to builds with Actions); no per-file limit stated (the 100 MB limit is a git push limit, and the
site is built in CI, not committed); one Pages site per repository; commercial hosting is not
allowed. The artifact for custom workflows must be a tar below 10 GB without links.

- 100 GB / 43 MB is about 2,300 cold visits a month from Pages (about 1,500 if the jsDelivr
  packages were vendored too). Returning visitors hit the browser cache. Fine for a research and
  teaching tool; a course with hundreds of simultaneous first visits is still far below the limit.
- One full copy is 50.7 MB, so about 19 versions fit into 1 GB. Splitting the `.mat` data out of
  the wheel (step 2 in `web_prototype.md`) would shrink a copy to about 20 MB.
- **Keep the packages on jsDelivr** for a public Pages deployment: it saves 21 MB per copy and
  bandwidth, is immutable-cached for a year (`max-age=31536000`) and works with COEP. Vendor them
  (the app has `?packages=<url>`; the harness already caches them in `web/pyodide/.cache/packages`)
  for institute or offline hosting, where nothing may leave the network.

Caching on Pages: `Cache-Control: max-age=600`, ETag, Last-Modified, `Access-Control-Allow-Origin:
*`, no `immutable`, no way to change it. Hashed Vite assets and versioned wheel names are therefore
only cached for ten minutes, after which each file costs a conditional request answered with 304
(run 6: 36 requests cold, 3 warm inside the window). The boot time hardly changes warm (12.0 s vs
9.9 s): the time goes into instantiating Pyodide and installing the wheels, not the network. One
pitfall, from reasoning and not tested: a page with a cached `index.html` can request a lazily
loaded chunk of the previous build right after a deployment; it 404s until the tab is reloaded.
The service worker could precache everything for offline use (about 70 MB in the Cache API); the
current worker deliberately does not, because cache invalidation across versions is its own
project and Safari evicts script-writable storage.

### MIME types, Jekyll, 404

Observed on live github.io sites: `.wasm` is `application/wasm` (needed for streaming
compilation), `.mjs` is `text/javascript`, `.json` JSON, `.whl` `application/octet-stream`.
Pyodide downloads wheels with `fetch`, so the type does not matter. `.nojekyll` is not needed:
Jekyll only runs for branch-based publishing, and `upload-pages-artifact` v4+ drops dotfiles from
the artifact anyway (v5 has `include-hidden-files`). A 404 returns Pages' own 404.html or none;
the app has no routes that need a fallback.

## Network traffic and privacy

Across runs 1-5 (boot, TG119, dose influence, optimization, Stop) the browser contacted exactly
two hosts: the deployment's own origin and `cdn.jsdelivr.net` (Pyodide package wheels). Reading
`web/src`, the only `fetch` is the wheel manifest; there is no analytics, telemetry, `sendBeacon`,
WebSocket or cookie. Loaded CT/CST/plan files are read into WebAssembly memory in a worker and
leave the browser only through the user's own downloads ("Save"). Precisely:

- Pages sees the visitor's IP and the files requested (GitHub documents that IPs are logged).
- jsDelivr sees the IP, the package file names and the origin as referrer. Vendoring the packages
  removes it.
- Static links (GitHub, documentation) are plain anchors and only load on click.
- **URL overrides.** `?pyodide=`, `?packages=` and `?pyrp=` accept any URL, so a crafted link could
  make the app run foreign code in a tab holding a user's data. Harmless on localhost; for a public
  deployment they should be ignored unless same-origin or in a development build (patch below).
  Under COEP most foreign hosts would be refused anyway, but that is not a control.
- The page already carries the disclaimer: title "(NOT FOR CLINICAL USE!)", a banner in the
  header, and About ("not a medical device and must not be used for clinical treatment planning").
  The splash has none; a first-visit acknowledgement would be cheap.
- Licenses: Help > Licenses lists the installed packages' license files from inside Pyodide. The
  JavaScript bundle (React, zustand) has no license file yet, and the Pyodide runtime (MPL-2.0)
  and the pyRadPlan BSD-3 `LICENSE` are not shipped as files next to the app. A `NOTICE` /
  third-party file generated at build time is needed before a public release.

## Pages and the existing docs

State checked with the public API on 2026-11-18: `e0404/pyRadPlan` has `has_pages: false`; the docs
are on Read the Docs (`pyradplan.readthedocs.io` redirects to `docs.pyradplan.org`); the
organization has no `e0404.github.io` site, but `e0404/matRad` has a project site at
`e0404.github.io/matRad/`. So today there is no conflict. A repository has one Pages site; if
the docs ever move to Pages, or a project page is wanted, the options are:

| Option | Notes |
|---|---|
| Web app in a sub-directory of the one site artifact (`/app/`, `/docs/`) | one repository, one deployment job; every deployment must rebuild everything |
| Separate repository (`e0404/pyRadPlan-web`) | independent Pages site and deploy rights; needs a mirror or a token to publish from this build; URL `e0404.github.io/pyRadPlan-web/` |
| Custom domain (`app.pyradplan.org` CNAME to `e0404.github.io`) | root URL, no sub-path; per repository, so it also needs the separate repository if the docs site is on Pages; DNS ownership unknown (the docs domain suggests the project has one) |

Because the base is relative, the build is identical in all three.

## Deployment workflow (draft)

`.github/workflows/web-pages.yml`, `workflow_dispatch` only, so merging it runs nothing. Steps:
checkout with tags (for the wheel version), Python 3.13 and Node 24, wheels (`npm run build` in
`web/pyodide`), frontend with `PYRADPLAN_COI=1 npm run build -- --base=./`, assembly of
`_site/<channel>/` plus a landing page, a smoke test (`web/e2e/pages-server.mjs` serves the site as
Pages does, `web/e2e/pages.mjs` boots the real backend, runs TG119, tests Stop and fails if any host
besides the app and jsDelivr is contacted), `configure-pages`, `upload-pages-artifact`, and a
deploy job with the `github-pages` environment (`pages: write`, `id-token: write`).

- Pinned to the current majors: `actions/configure-pages@v6`, `upload-pages-artifact@v5`,
  `deploy-pages@v5` (latest releases on 2026-11-18; the GitHub guide still shows v5/v4/v4). The
  inputs used are the documented ones; the workflow itself was not run (needs a runner and the
  Pages setting), only its YAML parsed and its commands run locally.
- Needs once: Settings > Pages > Source "GitHub Actions". The `github-pages` environment, by default,
  only allows the default branch to deploy; a tag or release branch needs an environment rule.
- **One channel per deployment.** The artifact replaces the entire site, so a `/v0.10/` deployment
  would delete `/latest/`. Ways to keep versions, from simple to complete:
  1. One live version (`latest`), as drafted.
  2. A build job that checks out the last N release tags and assembles all of them in one artifact
     (N wheel builds, about 5 min each; the smoke test only on the newest).
  3. Attach each version's built site as a release asset (a 43-50 MB zip) and let the deploy job
     download the last N and the new one; old versions are never rebuilt. Not implemented, not
     tested; the wheel and frontend are built from different dependency pins over time, so storing
     the result is more reproducible than rebuilding.
  4. Publish from a `gh-pages` branch: old folders persist, but the repository grows by about 40 MB
     per version.
- Previews per pull request are not a Pages feature (none found in the documentation, assumed);
  options are a `pr-<n>/` folder in a `gh-pages` branch with a third-party action (not evaluated) or
  one of the hosts below.

## Alternatives

| Host | Headers (COOP/COEP) | Versions and previews | Limits that matter | Fit |
|---|---|---|---|---|
| **GitHub Pages** | none; service worker needed (tested) | one site; folders need a scheme (above); no PR previews | 1 GB, 100 GB/month soft, one site per repo | free, public, same place as the code; weakest for versions |
| GitLab Pages (internal GitLab) | no per-project `_headers` found in the docs; instance admins can add headers for all Pages (`gitlab_pages['headers']`, Omnibus); the service worker works the same | `pages.path_prefix` parallel deployments are Premium/Ultimate; `pages-internal` already deploys the internal docs from `main` | no site size limit stated in the user docs | right for an internal-only build behind the institute login (Pages access control); same service-worker caveat |
| Cloudflare Pages | `_headers` file: real COOP/COEP | automatic previews per branch and PR, immutable deployment URLs | **25 MiB per file: today's 32.4 MB wheel does not fit**; 20,000 files; 500 builds/month free | best technically once the `.mat` data is split out of the wheel |
| Netlify | `_headers` file in the publish directory | deploy previews and branch deploys | not checked | same as Cloudflare, without the file limit I found |
| nginx at the institute | any header; brotli; `immutable` for hashed files; JSON/wasm types | any directory scheme (`/latest/`, `/v0.10/`), previews by CI to `/pr-n/` | none; you run it | best for intranet and for data that must not touch the internet (vendor the packages; `?packages=` and `?pyodide=` already exist) |
| PWA (manifest + precaching service worker), on any of the above | needs the same headers or worker | each version is its own scope | about 70 MB of Cache API storage; browsers may evict it | offline and instant second start; not a hosting option and not free (cache invalidation); only worth it after the data split |

For the radiotherapy context all options are static: no server sees patient data. The difference
is who sees visitor IPs (GitHub, jsDelivr), whether a hospital network can reach jsDelivr (often not:
then vendor the packages and host internally), and whether a public page with a medical-looking tool
needs more than the disclaimer (the research-use notice and a license page, above).

## Recommendation and next steps

1. **Public demo on GitHub Pages is feasible now** at `https://e0404.github.io/pyRadPlan/latest/`,
   built with the relative base and the service worker, packages on jsDelivr. Use the draft
   workflow, started by hand from a release tag. Keep the service worker out of builds for hosts
   that send headers.
2. **Before it goes public**, in this order: (a) restrict the `pyodide`/`packages`/`pyrp` URL
   overrides to development and same-origin; (b) third-party license/NOTICE file in the build and
   the `LICENSE` next to the app; (c) a short first-visit acknowledgement of "research and
   education only, not for clinical use"; (d) try Firefox and Safari on a real deployment of a
   private test (or the preview server) and decide about the Stop fallback (c).
3. **Versions.** Start with one live channel. If tagged versions are wanted, implement the release-asset
   scheme (option 3) or move to Cloudflare Pages or an institute server where every deployment has
   its own URL.
4. **Split the base data out of the wheel** (it is 31 of 50.7 MB, and it blocks Cloudflare Pages'
   25 MiB limit and every version copy).
5. Decide with the user: org-level (`e0404`) public hosting at all, a custom domain, and whether an
   internal GitLab Pages copy is wanted too (same build, no service worker if the admin sets headers).

## Verified and assumed

Verified: the sub-path and relative-base builds boot and run the TG119 pipeline without headers;
the service worker gives isolation and a working Stop in Edge, defers to real headers, and does
not loop; the hosts contacted; sizes and gzip sizes; MIME types, gzip, cache headers and absence of
COOP/COEP on live github.io sites; jsDelivr's CORS and CORP headers; GitHub, GitLab, Cloudflare and
Netlify limits and header facts as read from their documentation on 2026-11-18; the repository and
organization Pages state through the public API; action versions through the releases API; the
workflow's YAML parses.

Assumed or not tested: Firefox, Safari, mobile and a real github.io origin; that Pages redirects a
directory URL to the trailing slash; the stale-chunk problem after a deployment; that Pages has no
per-PR previews; the workflow end to end (needs Pages enabled and a runner); GitLab per-project
`_headers` (not found, not proven absent); Netlify limits.

## Files

- `docs/development/web_publishing.md`: this document.
- `.github/workflows/web-pages.yml`: draft deployment, manual trigger only.
- `web/pages/coi-serviceworker.js`: the service worker (not shipped unless `PYRADPLAN_COI=1`).
- `web/vite.config.ts`: additive plugin `coiServiceWorker` and its entry in `plugins`; one import
  extended with `readFileSync`.
- `web/e2e/pages-server.mjs`, `web/e2e/pages.mjs`: Pages-like static server and the smoke test used
  above (`node e2e/pages.mjs <url> [--stop] [--no-isolation]`).

Patch not applied (source owned by other work): in `web/src/api/index.ts`, `createBackend` should
read `pyodide`, `packages` and `pyrp` only when `import.meta.env.DEV` or the URL's origin equals
`location.origin`.

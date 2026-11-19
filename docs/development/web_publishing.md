# Publishing the web version (decision record)

Status 2026-11-19, branch `internal/web-pyodide`. Question: can the browser version of pyRadPlan
(`docs/development/web_prototype.md`) be hosted on GitHub Pages? Nothing was published, no
repository or Pages setting was touched; everything below was tested locally against a static
server that mimics Pages (see "What was tested").

**Decision (2026-11-19): two public channels**, `/latest/` (the newest release) and `/develop/`
(the develop branch), assembled into one Pages site by `.github/workflows/web-pages.yml`; see
"Two channels" below for how it works and what an administrator must do once.

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
no cookies or IndexedDB and exactly one `localStorage` entry (the acknowledgement of the
disclaimer, below), which matters because all project sites of one organization share the origin
`<org>.github.io`; its key starts with `pyradplan.`.

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
  The guard's `sessionStorage` key contains the worker's scope path, so `/latest/` and
  `/develop/` opened in the same tab each get their reload.
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
- **Disclaimer and privacy notice (done).** The text is the one the Qt About dialog already had,
  now in `pyRadPlan/app/about.json` (read by `pyRadPlan/app/_about.py` for Qt, embedded by
  `web/vite.config.ts` so the web shows it before the backend starts). On the first visit a modal
  (`web/src/components/help/NoticeDialog.tsx`) shows the disclaimer and the privacy facts and must
  be acknowledged; nothing (Pyodide, wheels, jsDelivr) is downloaded before that, which
  `web/e2e/pages.mjs` checks. Afterwards it stays reachable from the status bar ("NOT FOR CLINICAL
  USE · data stays in this browser"), Help > About (button "Privacy…") and a line on the splash.
  The acknowledgement is one `localStorage` entry whose key contains a hash of all texts, so any
  change of the text asks again; without storage (private windows, blocked site data) it is asked
  on every visit.
- **Licenses (done).** Help > Third-party Licenses lists the Python distributions from inside
  Pyodide (checked for the installed wheels: the Pyodide distribution packages such as numpy,
  scipy, h5py carry their texts; the SimpleITK stand-in is `pyradplan-sitk-shim`, BSD-3-Clause,
  its text says it is part of pyRadPlan and not SimpleITK; `array-api-compat` 1.15.0 ships no
  license file upstream and shows "No license text found"). The build writes
  `THIRD_PARTY_NOTICES.txt` and pyRadPlan's `LICENSE.txt` next to the app
  (`web/scripts/thirdPartyNotices.ts`, a Vite plugin): every npm package whose modules end up in a
  bundle (react, react-dom, scheduler, zustand today), the Pyodide runtime with CPython and
  Emscripten, and coi-serviceworker when the service worker is included, each with its license
  file copied verbatim. A package without a license file falls back to `web/licenses/<name>.txt`
  (Pyodide's npm package ships none; the texts there are copied from the upstream repositories at
  the versions used: Pyodide 314.0.8, CPython 3.14.2, Emscripten 5.0.3, coi-serviceworker), and
  the build fails when neither exists. The minifier drops the `@license` banners of React; the
  notice file carries their full texts. The licenses view links both files. Not enumerated: the
  system libraries Emscripten and Pyodide compile into `pyodide.asm.wasm` beyond CPython and
  Emscripten's own (Pyodide publishes no list).

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

## Two channels

Exactly two public deployments: `https://<org>.github.io/<repo>/latest/` (the newest release) and
`.../develop/` (the head of `develop`), with a landing page at `.../` that links both, says which
version and commit each is, and carries the disclaimer line. A Pages deployment replaces the
whole site, so every run of `.github/workflows/web-pages.yml` builds and assembles both channels
(option 2 of the earlier list, with N = 1 release; a release-asset scheme or a `gh-pages` branch
remain the ways to keep more versions).

How a run works (two jobs, `build` and `deploy`):

1. Triggers: push to `develop`, pull requests into `develop` or `main` (both with the path filter
   of `web.yml`: `web/`, `pyRadPlan/`, `pyproject.toml`, the workflow), `release: published` and
   `workflow_dispatch` from any branch. Deploying runs (develop, releases) share the concurrency
   group `web-pages-deploy` without cancelling, so deployments never overlap; every other ref has
   its own group, and a newer push to a pull request cancels the older build.
2. The **"develop" channel** is checked out into `src/develop`: the `develop` branch for a
   release, otherwise the ref under test (the develop head on a push, the pull request's merge
   commit, the branch of a manual run), so the changes being tested are what gets built. All
   history and tags (`fetch-depth: 0`, `fetch-tags: true`, plus `git fetch --tags`), because
   setuptools-scm, the labels and the release tag need them. The **newest release** is
   `gh release view` (GitHub's "latest" release, which excludes drafts and pre-releases), or, if
   that is not a `vX.Y.Z` tag, the highest `vX.Y.Z` tag; it is checked out with `git worktree add`
   into `src/latest`. Each channel is built from its own checkout: its own wheels
   (`web/pyodide/build.py` of that checkout) and its own frontend, so `/latest/` uses the web
   sources of the tag, not develop's.
3. `web/scripts/test_pages.mjs --install --latest src/latest --develop src/develop --prefix
   /<repo>/` runs `assemble_pages.mjs --build --coi`, which builds each checkout (`npm ci`, wheels,
   `npm run build -- --base=./` with `PYRADPLAN_COI=1` and `VITE_PYRADPLAN_CHANNEL`,
   `VITE_PYRADPLAN_VERSION` (`git describe --tags`), `VITE_PYRADPLAN_COMMIT`), copies each
   `web/dist` to `_site/<channel>/` and writes `_site/index.html` and `_site/channels.json`. The
   app shows "Web build: develop channel · v0.9.0-12-g… · commit …" in Help > About.
4. **Releases without `web/`** (all up to v0.9.0): `_site/latest/index.html` becomes a static page
   saying that this release does not contain the browser version yet, that it is first included
   in a later release, and linking to `/develop/`. The run does not fail.
5. Smoke test, in the same script: `web/e2e/pages-server.mjs` serves `_site` as Pages does under
   `/<repo>/`; `/`, `/latest/`, `/develop/`, the notices and `LICENSE.txt` are fetched;
   `web/e2e/pages.mjs` opens `/develop/` in a fresh profile, checks the first-visit notice and
   that nothing was loaded from jsDelivr before it was acknowledged, boots the real backend, runs
   TG119, tests Stop and fails if any host besides the site and jsDelivr is contacted.
6. The assembled site is uploaded as the artifact **`web-pages-site`** (7 days) on every run; on
   deploying runs also with `upload-pages-artifact`. The `deploy` job (`configure-pages`,
   `deploy-pages`, environment `github-pages`, `pages: write`, `id-token: write`) runs only for a
   push or manual run on `develop` and for published releases, and only when the repository
   variable `WEB_PAGES_ENABLED` is `'true'`. `configure-pages` sits in the deploy job because it
   fails while Pages is disabled. Merging the workflow therefore deploys nothing until the
   administrator steps below are done, while pull requests already get the build and smoke test.

**What an administrator does once**, in this order:

1. Settings > Pages > Build and deployment > Source: **GitHub Actions**.
2. Settings > Environments > `github-pages` > Deployment branches and tags: allow the branch
   `develop` and the tag pattern `v*` besides the default branch (a push to develop runs on
   `refs/heads/develop`, a release on `refs/tags/vX.Y.Z`; by default only the default branch may
   deploy).
3. Settings > Secrets and variables > Actions > Variables: `WEB_PAGES_ENABLED` = `true`.
4. Start the workflow by hand once on `develop` (Actions > Web Pages > Run workflow) or push to
   develop.

Both channels share one origin (`<org>.github.io`), checked and decided:

- **Service workers.** Each channel's `coi-serviceworker.js` registers with its own URL, so its
  scope is its directory (`/<repo>/latest/`, `/<repo>/develop/`); the landing page and the
  placeholder register none. A worker only controls pages in its scope and only rewrites their
  responses, so the two do not interfere; the reload guard is namespaced by scope (above).
- **localStorage.** The only entry is the notice acknowledgement. It is **deliberately shared**
  by both channels: the key contains a hash of the texts, so the same text needs one
  acknowledgement, and a channel with a different text has a different key and asks on its own
  (no ping-pong between the channels). Settings are not persisted in the browser at all (they live
  in the Python session and are lost on reload), so there is nothing else to namespace.

### Testing the page build

All three run the same `web/scripts/test_pages.mjs` (assemble both channels, serve like Pages,
fetch, browser smoke test); none of them deploys anything.

- **Locally**, one command in `web/` (needs the tags locally, `npm install`, the installed
  Microsoft Edge; without `--skip-wheels` also `PYTHON` with pip and network to rebuild the
  wheels): `npm run pages:test -- --skip-wheels`. It clones the newest `vX.Y.Z` tag shallowly into
  `<repo>/_pages/latest`, builds the working tree as `develop` into `<repo>/_site` (both
  gitignored) and prints the result; open `http://localhost:8080/pyRadPlan/` while it runs, or
  serve `_site` afterwards with `node e2e/pages-server.mjs ../_site --prefix=/pyRadPlan/` (in Git
  Bash with `MSYS_NO_PATHCONV=1`, which otherwise rewrites `/pyRadPlan/` into a Windows path).
- **On GitLab**: the job `web-pages` in `.gitlab-ci.yml` runs on `internal/*` and `devops/*`
  branches with web-related changes (the rules of the other `web-*` jobs), with `GIT_DEPTH: "0"`
  and an explicit `git fetch --tags` so the release tag exists, builds the branch as `develop`,
  and keeps `_site/` as the job artifact for a week (Browse or Download on the job page). There is
  no GitLab Pages deployment.
- **On GitHub**: every pull request into `develop` or `main` that touches the web paths runs the
  `build` job; a manual run (Actions > Web Pages > Run workflow, pick the branch) builds any
  branch as `develop`. Download `web-pages-site` from the run's summary page and serve it locally
  with `pages-server.mjs` as above. Neither deploys.
- **A real deployment, safely**: in a scratch repository (a fork or a private copy under a
  personal account; free accounts publish Pages only from public repositories), set Settings >
  Pages > Source to "GitHub Actions", add `develop` to the `github-pages` environment rules, set
  the variable `WEB_PAGES_ENABLED` to `true`, and push this branch there as `develop` (with the
  tags, `git push --tags`). The site appears at `https://<account>.github.io/<scratch-repo>/` and
  exercises the real github.io origin (service worker, caching, sub-path) without touching the
  project's Pages. Delete the repository or switch Pages off afterwards.

Local test done for this document (2026-11-19): `npm run pages:test -- --skip-wheels` with v0.9.0
as `latest` (placeholder) and the working tree as `develop` passed: all five URLs 200, the notice
came before any jsDelivr request, TG119 dose influence + optimization, Stop in 35-80 ms, only
`localhost` and `cdn.jsdelivr.net` contacted, no browser errors.

Pinned like the other workflows to the current majors: `actions/checkout@v7`,
`setup-python@v7`, `setup-node@v7`, `upload-artifact@v7`, `configure-pages@v6`,
`upload-pages-artifact@v5`, `deploy-pages@v5`. Neither CI job has run on a runner yet; `actionlint`
passes for the workflow, both YAML files parse, and the commands were run locally.

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

1. **Public demo on GitHub Pages** at `https://e0404.github.io/pyRadPlan/latest/` and
   `.../develop/`, built with the relative base and the service worker, packages on jsDelivr, by
   the workflow above once the administrator steps are done. Keep the service worker out of builds
   for hosts that send headers.
2. **Before it goes public**: (a) URL overrides restricted to development and same-origin, done;
   (b) third-party notices and `LICENSE.txt` in the build, done; (c) first-visit acknowledgement
   of the disclaimer and privacy facts, done; (d) still open: try Firefox and Safari on a real
   deployment of a private test (or the preview server) and decide about the Stop fallback (c).
3. **Versions.** Two channels (decided). If tagged versions are wanted later, use the
   release-asset scheme or move to Cloudflare Pages or an institute server where every deployment
   has its own URL. `/latest/` shows a placeholder until the first release that contains `web/`.
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
workflow's YAML parses. 2026-11-19: the two-channel assembly with v0.9.0 (placeholder) and the
working tree, served under `/pyRadPlan/`, with the smoke test (notice before any CDN request,
TG119, Stop, hosts, notices deployed); `actionlint` on the workflow; the license rows under
Pyodide.

Assumed or not tested: Firefox, Safari, mobile and a real github.io origin; that Pages redirects a
directory URL to the trailing slash; the stale-chunk problem after a deployment; that Pages has no
per-PR previews; the GitHub workflow and the GitLab job end to end (need runners, and Pages
enabled for the deploy job); that the `github-pages` environment rules are as described; GitLab per-project
`_headers` (not found, not proven absent); Netlify limits.

## Files

- `docs/development/web_publishing.md`: this document.
- `.github/workflows/web-pages.yml`: the two-channel deployment.
- `web/scripts/assemble_pages.mjs`: builds and assembles the channels, the landing page and the
  placeholder; `web/scripts/test_pages.mjs` (`npm run pages:test`): assembly plus serving and smoke
  test, used locally and by both CIs (`web-pages` job in `.gitlab-ci.yml`).
- `web/pages/coi-serviceworker.js`: the service worker (not shipped unless `PYRADPLAN_COI=1`).
- `web/vite.config.ts`: plugins `coiServiceWorker`, `pyodideAssets` and the notices plugin; the
  About texts as `__PYRADPLAN_ABOUT__`.
- `web/scripts/thirdPartyNotices.ts`, `web/licenses/`: `THIRD_PARTY_NOTICES.txt` and its fallback
  license texts.
- `pyRadPlan/app/about.json`, `pyRadPlan/app/_about.py`: disclaimer, license note and description
  for both GUIs; `web/src/components/help/notice.ts`, `NoticeDialog.tsx`: the first-visit notice.
- `web/e2e/pages-server.mjs`, `web/e2e/pages.mjs`: Pages-like static server and the smoke test used
  above (`node e2e/pages.mjs <url> [--stop] [--no-isolation]`).

The URL overrides `pyodide`, `packages` and `pyrp` are honoured only in development or for URLs on
the page's own origin (`trustedSource` in `web/src/api/index.ts`).

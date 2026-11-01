# pyRadPlan in a web browser (container)

This directory builds a container image in which the unchanged Qt GUI runs on a virtual X
server, and [Xpra](https://xpra.org) streams its window to an HTML5 client in the browser.
There is no web port of the GUI: the browser only shows the screen and sends mouse and
keyboard input, so everything pyRadPlan computes runs in the container (CPU only; there is no
GPU in the image). It is meant for research and education, like the desktop installers.

| File | Purpose |
|---|---|
| `Dockerfile` | Ubuntu 24.04 with Qt's libraries, Xpra and pyRadPlan installed from the checkout. |
| `entrypoint.sh` | Starts Xpra with the command (by default `pyRadPlanGUI TG119`); the container exits when the GUI is closed. |
| `smoke_test.sh` | Checks a started container: web client, main window, clean exit. Run by both CI pipelines. |
| `Dockerfile.dockerignore` | Sends only the files the Dockerfile copies to the Docker daemon. |
| `compose.yaml` | A single session with a course folder and resource limits, for trying it out. |

## Build and run

From the repository root:

```bash
docker build -f packaging/container/Dockerfile -t pyradplan-web .
docker run --rm -p 10000:10000 -e XPRA_PASSWORD=choose-one pyradplan-web
```

Then open <http://localhost:10000>. The first start takes a while (the GUI imports the
numerical stack). Closing the GUI stops the container. `docker compose up --build` in this
directory does the same with the settings of `compose.yaml`, binding the web port to
`127.0.0.1` so the passwordless default is accessible only from the host. To serve other
machines, set `XPRA_PASSWORD` and explicitly change the port binding.

The build installs pyRadPlan from the working tree, so the image contains exactly the commit
it was built from; there is no separate version to pass. `--build-arg PYRADPLAN_EXTRAS=gui`
leaves numba out.

## Published images

The "Container" workflow on GitHub (`.github/workflows/container.yml`) builds the image and
checks it with `smoke_test.sh` (the web client answers, the GUI shows its main window, and the
container ends with exit code 0 when the GUI is closed). Branches (`main`, `rc/*`, `devops/*`)
are built and tested only. A release tag `vX.Y.Z` publishes the tested image as
`ghcr.io/e0404/pyradplan-web:X.Y.Z`, and as `:latest` except for pre-release tags such as
`v0.6.0rc1`.

The `container` job of the internal GitLab pipeline builds and tests the image the same way, for
branches that do not reach GitHub. It does not publish anything.

```bash
docker run --rm -p 10000:10000 -e XPRA_PASSWORD=choose-one ghcr.io/e0404/pyradplan-web:latest
```

The licenses of the Python packages in the image are listed in
`/usr/share/doc/pyradplan-web/THIRD_PARTY_NOTICES.txt` (also in the GUI's Help menu); the
Ubuntu and Xpra packages keep their copyright files in `/usr/share/doc`. A package pushed to
the GitHub registry for the first time may be private until its visibility is changed in the
package settings. The GitLab job needs a privileged runner for the `docker:dind` service.

## Settings

Everything is configured through environment variables, so the image needs no changes for a
deployment.

| Variable | Effect |
|---|---|
| `XPRA_PASSWORD` | Password the browser asks for. **Without it anyone who can reach the port can use the session** (the container warns). |
| `XPRA_PORT` | Port of the web client inside the container, 10000 by default. |
| `PYRADPLAN_GUI_UI_MODE=educational` | Reduced widgets for teaching. |
| `PYRADPLAN_BASE_DATA_PATH=/shared` | Shared base data folders: mount a folder with `phantoms/` (`.mat`, `.npz`) and, if wanted, `machines/` there and the cases are listed under File → Open Example Phantom. |
| `PYRADPLAN_DATA_DIR` | The user's data directory (`~/.pyradplan` of the `pyradplan` user by default); mount a volume there to keep settings and added base data between sessions. |
| `PYRADPLAN_AI_*`, provider API keys | AI features. A key in a shared image can be used by every session; leave them out if that is not wanted. |

A different command can be given instead of the default, for example `pyradplan-web
pyRadPlanGUI /shared/cases/Prostate.mat` to start with a case loaded.

## Several users

The image serves one session: each user needs a container of their own, and the
container is the isolation between them. Something has to start the containers and
authenticate the users; a JupyterHub with a remote desktop plugin, Apache Guacamole or a
Kubernetes job per session can do that. Set memory and CPU limits (`--memory`, `--cpus`):
a dose calculation of a realistic plan needs gigabytes.

## Limits

- Research and education only, like the rest of pyRadPlan: do not put patient data in a
  deployment that is not set up and secured for it. Xpra's own transport is not encrypted;
  put a reverse proxy with TLS in front of anything beyond a trusted network.
- No GPU, and Xpra's audio, webcam and printing are switched off in the entrypoint. Xpra's
  video encoders based on x264 are not installed; it uses its image encoders instead.
- The image is built for x86-64 only.

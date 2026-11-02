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
| `Dockerfile.hub` | The JupyterHub session image: the image above plus jupyter-server-proxy's standalone proxy. |
| `hub-session.sh` | Session server of the JupyterHub image; ends the session when the GUI is closed. |
| `jupyterhub-values.yaml` | Helm values for Zero to JupyterHub, installed into one namespace. |

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

The "Container" workflow on GitHub (`.github/workflows/container.yml`) builds the image and,
on top of it, the JupyterHub image, and checks both with `smoke_test.sh` (the web client
answers, the GUI shows its main window, and the container ends with exit code 0 when the GUI
is closed). Branches (`main`, `rc/*`, `devops/*`) are built and tested only. A release tag
`vX.Y.Z` publishes the tested images as `ghcr.io/e0404/pyradplan-web:X.Y.Z` and
`ghcr.io/e0404/pyradplan-hub:X.Y.Z`, and as `:latest` except for pre-release tags such as
`v0.6.0rc1`.

The `container` job of the internal GitLab pipeline builds and tests the images the same way,
for branches that do not reach GitHub. It does not publish anything.

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
| `XPRA_BIND` | Address the web client listens on, `0.0.0.0` by default. |
| `PYRADPLAN_GUI_UI_MODE=educational` | Reduced widgets for teaching. |
| `PYRADPLAN_BASE_DATA_PATH=/shared` | Shared base data folders: mount a folder with `phantoms/` (`.mat`, `.npz`) and, if wanted, `machines/` there and the cases are listed under File → Open Example Phantom. |
| `PYRADPLAN_DATA_DIR` | The user's data directory (`~/.pyradplan` of the `pyradplan` user by default); mount a volume there to keep settings and added base data between sessions. |
| `PYRADPLAN_AI_*`, provider API keys | AI features. A key in a shared image can be used by every session; leave them out if that is not wanted. |

A different command can be given instead of the default, for example `pyradplan-web
pyRadPlanGUI /shared/cases/Prostate.mat` to start with a case loaded.

## Several users

The image serves one session: each user needs a container of their own, and the
container is the isolation between them. Something has to start the containers and
authenticate the users. Set memory and CPU limits (`--memory`, `--cpus`): a dose calculation
of a realistic plan needs gigabytes.

### JupyterHub on Kubernetes

The JupyterHub image (`Dockerfile.hub`) lets a [Zero to JupyterHub](https://z2jh.jupyter.org)
installation start a pod with the GUI for every user who logs in. It runs jupyter-server-proxy's
standalone proxy in place of JupyterLab: the proxy logs the user in through the hub and
forwards the browser to the Xpra web client, which listens on localhost only. Users open the
hub, log in and get the GUI under `/user/<name>/`.

- At most `active_server_limit` sessions run at the same time; further users are told that
  the limit is reached and can try again later. There is no waiting list.
- Closing the GUI ends the session (`hub-session.sh` stops the server), and the hub offers to
  start a new one. A session whose browser tab has been closed for 30 minutes is stopped by
  the hub's culler.
- Each session starts with a fresh home directory; files are not kept between sessions.

`jupyterhub-values.yaml` installs the chart into one namespace without cluster-wide
permissions (no user scheduler, no image pullers, no privileged init container) and with
resource limits on every pod, as a namespace quota requires:

```bash
helm repo add jupyterhub https://hub.jupyter.org/helm-chart/
helm upgrade --install pyradplan-hub jupyterhub/jupyterhub --version 4.4.4 \
    --namespace <namespace> --values packaging/container/jupyterhub-values.yaml
```

Set the passwords, the host name of the ingress and the image tag in the file first. The
shared-password login lets anyone with the password log in under any name, which suits
courses and trials on a trusted network; the chart supports other authenticators (OAuth,
LDAP) for more. The JupyterHub version in the image matches the chart's hub; update both
together.

Without a hub, the image can be tried locally:

```bash
docker build -f packaging/container/Dockerfile.hub -t pyradplan-hub \
    --build-arg BASE_IMAGE=ghcr.io/e0404/pyradplan-web:latest packaging/container
docker run --rm -p 8888:8888 pyradplan-hub hub-session.sh --no-authentication --port=8888
```

Then open <http://localhost:8888>.

## Limits

- Research and education only, like the rest of pyRadPlan: do not put patient data in a
  deployment that is not set up and secured for it. Xpra's own transport is not encrypted;
  put a reverse proxy with TLS in front of anything beyond a trusted network.
- No GPU, and Xpra's audio, webcam and printing are switched off in the entrypoint. Xpra's
  video encoders based on x264 are not installed; it uses its image encoders instead.
- The image is built for x86-64 only.

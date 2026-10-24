# The pyRadPlan GUI in a web browser. The unchanged Qt application runs in this container on
# a virtual X server and Xpra streams its window to an HTML5 client; see README.md.
#
# Build from the repository root (the build installs pyRadPlan from the checkout):
#   docker build -f packaging/container/Dockerfile -t pyradplan-web .
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
# Extras of pyRadPlan to install; numba compiles the ray tracer, there is no GPU in the image.
ARG PYRADPLAN_EXTRAS=gui,numba

# The libraries of Qt's xcb platform plugin, a virtual X server and a font with the Greek
# letters of the GUI. Xpra comes from xpra.org, the package of Ubuntu 24.04 is from 2021.
# hadolint ignore=DL3008
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates curl python3 python3-venv \
        libgl1 libegl1 libglib2.0-0 libdbus-1-3 libfontconfig1 libxkbcommon-x11-0 \
        libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
        libxcb-shape0 libxcb-xkb1 \
        xvfb xauth fonts-dejavu-core \
 && curl -fsSL -o /usr/share/keyrings/xpra.asc https://xpra.org/xpra.asc \
 && printf '%s\n' 'Types: deb' 'URIs: https://xpra.org' 'Suites: noble' 'Components: main' \
        'Signed-By: /usr/share/keyrings/xpra.asc' > /etc/apt/sources.list.d/xpra.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends xpra xpra-x11 xpra-html5 xpra-codecs \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE /src/
COPY pyRadPlan /src/pyRadPlan
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir "/src[${PYRADPLAN_EXTRAS}]" \
 && rm -rf /src

# The license notices of everything in the virtual environment, as shipped with the desktop
# installers (the GUI's Help menu lists them too); the check fails the build on a GPL-only
# dependency. The Ubuntu and Xpra packages keep their copyright files in /usr/share/doc.
COPY packaging/briefcase/scripts/collect_licenses.py /tmp/collect_licenses.py
RUN /opt/venv/bin/python /tmp/collect_licenses.py \
        "$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
        --output-dir /usr/share/doc/pyradplan-web --check \
 && rm /tmp/collect_licenses.py

COPY packaging/container/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
 && useradd --create-home --uid 10001 pyradplan

# Qt draws with software OpenGL and without shared memory (there is no GPU, and the X server
# runs in the same container namespace but not necessarily with a large /dev/shm).
ENV PATH=/opt/venv/bin:$PATH \
    QT_QPA_PLATFORM=xcb \
    QT_X11_NO_MITSHM=1 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    PYRADPLAN_DATA_DIR=/home/pyradplan/.pyradplan \
    XPRA_PORT=10000

# Set by the CI builds. The source label is what links a package on the GitHub Container
# Registry to the repository.
ARG VERSION=dev
ARG REVISION=unknown
ARG SOURCE=https://github.com/e0404/pyRadPlan
LABEL org.opencontainers.image.title="pyRadPlan in a web browser" \
      org.opencontainers.image.description="The pyRadPlan GUI streamed to a web browser with Xpra. For research and education only, not for clinical use. License notices: /usr/share/doc/pyradplan-web" \
      org.opencontainers.image.source="${SOURCE}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"

USER 10001
WORKDIR /home/pyradplan
EXPOSE 10000
ENTRYPOINT ["entrypoint.sh"]
CMD ["pyRadPlanGUI", "TG119"]

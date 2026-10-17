# The image of the prediction API (`python -m demand serve`, src/demand/serving).
#
#   docker build -f docker/api.Dockerfile -t nyc-taxi-demand/api:local .
#
# (docker compose builds it for you; see docker-compose.yml.) The last argument, `.`, is
# the build context: the project folder, filtered by .dockerignore.
#
# What is in the image: Python, the libraries of docker/requirements-api.lock, this
# package and configs/config.yaml. What is not: the data and the model. The service reads
# the data from a folder mounted at /app/data and downloads the model from the MLflow
# server at startup, so one image serves any champion and any data snapshot, and a new
# model needs a restart, not a rebuild.
#
# Two stages ("multi-stage build"). The first, `builder`, installs everything into a
# virtual environment; the second, `runtime`, starts again from a clean Python image and
# copies only that environment across. What installing needs and running does not (pip's
# download cache, the build backend, the package's source tree) stays behind in the
# first stage and never reaches the image that runs.

# The base image, pinned twice. The tag names the version a person reads: Python
# 3.12.15 (the project requires 3.12) on Debian 13 "trixie", in the "slim" variant,
# which leaves out compilers and documentation (179 MB instead of about 1 GB). The digest
# after the @ is the checksum of that exact image: a tag can be moved to a rebuilt image
# (Docker Hub re-publishes it for every Debian security fix), a digest cannot. So this
# line builds the same base on every machine until someone changes it. To move to a newer
# base: `docker pull python:3.12-slim-trixie`, then copy the new tag and the digest that
# `docker images --digests` prints. Declared once, before the first FROM, so that the
# two stages cannot drift apart.
ARG PYTHON_IMAGE=python:3.12.15-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1


# ---------------------------------------------------------------------------------------
# Stage 1: install
# ---------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

# No .pyc files from this stage's own commands, and no "a newer pip exists" notice.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# A virtual environment in a folder of its own, so that the runtime stage can copy
# exactly what was installed, and nothing of the system Python, in one line. It must
# have the same path in both stages: a venv records its own path in its scripts.
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /build

# The libraries first, the package after, so that Docker's layer cache works for us.
# Docker reuses the result of a step as long as the files it copied have not changed. The
# lock changes rarely and takes minutes to install; the source changes all the time and
# takes seconds. In this order, a change to the code reruns only the last two steps.
COPY docker/requirements-api.lock ./

# --no-deps: the lock already lists every package, dependencies included, each at one
# version (compiled with requirements.lock as a constraint, so every version is the
# project's; the first lines of the file give the command). pip installs exactly that
# list and resolves nothing itself.
# --mount=type=cache keeps pip's download cache on the build machine between builds,
# outside the image, so a rebuild does not download 300 MB again.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps --requirement requirements-api.lock

# The package itself, as a normal (not editable) install: the code is copied into the
# venv, so the image does not depend on /build. --no-deps again: its dependencies are
# the lock above. The dev tools (pytest, ruff, mypy) are an extra that is not installed.
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps .


# ---------------------------------------------------------------------------------------
# Stage 2: run
# ---------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

# LightGBM is compiled against OpenMP (libgomp), which the slim image does not have; it
# fails to import without it. --no-install-recommends installs that library and nothing
# it merely suggests, and the package lists are deleted in the same step, so they never
# become part of a layer. The version is not pinned: Debian's stable release ships only
# security fixes for it, and the pinned base digest above already fixes the release.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# A user without privileges to run the service. A process that runs as root inside a
# container is root on the container's files, and a step closer to the host if the
# container is ever broken out of. A fixed number (10001, outside the range of normal
# users), so that Kubernetes can check `runAsNonRoot` (it can only check a number) and
# file permissions on mounted folders are predictable. No home folder and no login shell:
# the service needs neither.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --home-dir /nonexistent \
       --shell /usr/sbin/nologin app

# The installed environment from stage 1, and nothing else from it.
COPY --from=builder /opt/venv /opt/venv

# The service's working folder. Paths in the config are relative to it: the data is
# expected at /app/data (mounted at run time) and the config at /app/configs.
WORKDIR /app

# The config is part of the image: the zones and the feature settings must match the
# model, and they change together with the code. Owned by root and read-only to the
# service, like everything else in the image. Any value can still be overridden at run
# time with a DEMAND_<SECTION>__<KEY> variable (src/demand/config.py).
COPY configs/config.yaml configs/config.yaml

# PATH: `demand` and `python` are the venv's.
# PYTHONUNBUFFERED: log lines reach `docker logs` at once, not when a buffer fills.
# PYTHONDONTWRITEBYTECODE: the root filesystem may be read-only (docker-compose.yml);
#   pip already compiled the .pyc files at install time.
# MLFLOW_DISABLE_AGENT_HINT: MLflow otherwise prints an advertisement on import.
# DEMAND_MLFLOW__TRACKING_URI: where the model comes from. The config's default, a SQLite
#   file in the working folder, does not exist in this image, and the service refuses to
#   start without it; compose and Kubernetes point this at the MLflow server.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    DEMAND_MLFLOW__TRACKING_URI=http://mlflow:5000

# Everything from here on, and the container itself, runs as that user. By number, not
# by name: Docker and Kubernetes then know it is not root (0) without looking the name
# up inside the image.
USER 10001:10001

# Documentation only: the port the service listens on (serving.port in the config).
# Publishing it to the host is decided by `docker run -p` or docker-compose.yml.
EXPOSE 8000

# How Docker knows the container is healthy: every 30 s it runs this command inside the
# container, and three failures in a row mark it "unhealthy". It asks /ready, not
# /health: healthy here means "can answer a forecast", which is what compose waits for
# (Kubernetes, in phase 6, asks the two questions separately). urlopen raises on any
# status other than 2xx, so a 503 fails the check. Python, because the slim image has no
# curl and installing one only for this would add a program to the image. The startup
# period gives the model download and the data loading 90 s before failures count
# (it took 9 s warm and about 25 s cold); during it, the check runs every 5 s, so the
# container is marked healthy soon after it is ready. The port is hard-coded: a
# container started with another DEMAND_SERVING__PORT needs its own health check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --start-interval=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4)"]

# The exec form (a JSON list, no shell in between): the service is process 1 of the
# container and receives Docker's stop signal itself, so uvicorn shuts down cleanly
# instead of being killed after 10 s. One uvicorn worker per container; more load means
# more containers (replicas), not more workers in one.
CMD ["demand", "serve"]

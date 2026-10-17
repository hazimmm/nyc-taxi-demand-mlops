# The image of the batch side: the `demand` command line (train, promote, tune, backtest,
# and the data commands) and, in docker-compose.yml, the MLflow tracking server.
#
#   docker build -f docker/pipeline.Dockerfile -t nyc-taxi-demand/pipeline:local .
#
# It is built the same way as the API image, and docker/api.Dockerfile explains the
# shared steps line by line (the two stages, the pinned base, the lock, the cache). Only
# what differs is explained here:
#
#   * the libraries of docker/requirements-pipeline.lock: the `ml` extra (LightGBM,
#     MLflow, Optuna, SHAP, matplotlib). FastAPI and uvicorn come with it, because the
#     MLflow server needs them; only the Prometheus client of the API is missing;
#   * git, which training calls to record the commit a model was trained from;
#   * a home folder for the user, and two folders for the MLflow server's volumes.
#
# Why one image runs both the commands and the MLflow server: the server is MLflow
# itself, the same package at the same version (3.17.0, from the lock) that the commands
# use as a client. Taking it from this image guarantees that client and server match,
# runs it as a user without privileges (the official image, ghcr.io/mlflow/mlflow, runs as
# root and is 1.3 GB on its own), and leaves one image fewer to pull and keep up to date.
#
# DVC is not installed, and `dvc repro` does not run in a container. The data stages
# write data/, dvc.lock and the DVC cache into the project folder; from a container they
# would write into the checkout on the host as another user, through the slowest kind of
# mount Docker Desktop has. The data is built on the host with `dvc repro`, as before;
# the containers only read it. docker-compose.yml says how they reach it.

ARG PYTHON_IMAGE=python:3.12.15-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1


# ---------------------------------------------------------------------------------------
# Stage 1: install
# ---------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /build

COPY docker/requirements-pipeline.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps --requirement requirements-pipeline.lock

COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps .


# ---------------------------------------------------------------------------------------
# Stage 2: run
# ---------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

# libgomp1 for LightGBM, as in the API image. git, because `demand train` records the
# commit it was run from and whether files differed from it (src/demand/models/training.py),
# and the promotion gate refuses a model without a clean commit. git reads the project
# folder that docker-compose.yml mounts at /repo.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# Since version 2.35.2 git refuses to read a repository that belongs to another user, to
# protect against a planted .git folder. The mounted checkout belongs to whoever owns it
# on the host, never to the user below, so git is told to trust any folder. That is safe
# here because the only repository this container ever sees is the one it is given to
# read. --system writes /etc/gitconfig, which applies to every user of the image.
RUN git config --system --add safe.directory '*'

# The user, as in the API image (same number), but with a home folder: matplotlib (the
# SHAP plot) and MLflow keep small caches and settings under ~ and fail without one.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app \
       --shell /usr/sbin/nologin app

# The folders the containers of this image write to; docker-compose.yml mounts a named
# volume on each. /mlflow/backend and /mlflow/artifacts: the MLflow server's database and
# model files. /work/reports: the reports the commands write (backtest, training,
# promotion), which must not land in the read-only checkout. They are created here, owned
# by the user, because Docker fills a new named volume with what the image has at that
# path, owner included. Without them the volumes would be created empty and owned by
# root, and the user could not write to them.
RUN mkdir -p /mlflow/backend /mlflow/artifacts /work/reports \
    && chown -R app:app /mlflow /work

COPY --from=builder /opt/venv /opt/venv

# The working folder when the image is used on its own, with the config the image
# carries. docker-compose.yml runs the commands from the mounted checkout instead.
WORKDIR /app
COPY configs/config.yaml configs/config.yaml

# As in the API image. MPLBACKEND: draw the SHAP plot into a file; there is no screen.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MPLBACKEND=Agg \
    DEMAND_MLFLOW__TRACKING_URI=http://mlflow:5000

# By number, as in the API image.
USER 10001:10001

# `docker run IMAGE train` runs `demand train`: the arguments after the image name are
# appended to the entry point. With none, it prints the list of commands.
ENTRYPOINT ["demand"]
CMD ["--help"]

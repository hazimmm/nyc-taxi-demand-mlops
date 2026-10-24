# Kubernetes

This folder runs the project on Kubernetes: the prediction API, the MLflow server it
takes its model from, and the daily step of the replay as a scheduled job. It is the
same system as `docker-compose.yml`, written the way a cluster expects it. Locally the
cluster is [kind](https://kind.sigs.k8s.io/) ("Kubernetes in Docker"): one Docker
container on this machine that behaves like a Kubernetes node.

This file explains what is here and why each choice was made. The commands to bring
the cluster up, step by step, are at the end ("Running it").

**What was checked.** Every manifest here renders with `kubectl kustomize` and passes
`kubeconform`, a validator that checks each object against the official schemas of
Kubernetes 1.36. The numbers (memory, CPU, start times) were measured on the same images
under Docker, with the same restrictions the pods get (read-only root, no capabilities,
user 10001). Then every step of "Running it" was run on a kind cluster (kind v0.33.0,
Kubernetes 1.36.4); the outputs quoted there are from that run. Where a later run
disagrees, the run is right and this file needs fixing.

## The layout

```
k8s/
├── kind-cluster.yaml              the local cluster: one node, its image pinned
├── base/                          the project, the same on every cluster
│   ├── kustomization.yaml         namespace, labels, the settings (ConfigMap, Secret)
│   ├── namespace.yaml             the namespace, with "restricted" pod security
│   ├── data-pvc.yaml              the volume for the data and the replay's files
│   ├── mlflow.yaml                MLflow: its volume, Deployment and Service
│   ├── api.yaml                   the API: Deployment, Service, autoscaler
│   └── daily-step-cronjob.yaml    the daily step, as a CronJob
└── overlays/kind/                 what is particular to a kind cluster
    ├── kustomization.yaml         the base, with local image tags
    ├── pull-never.yaml            never pull the images: they are loaded into the node
    ├── suspend-daily-step.yaml    no runs by the clock; runs are started by hand
    ├── data-loader.yaml           a pod used once to copy the data in
    └── metrics-server/            the add-on that measures CPU for the autoscaler
        ├── kustomization.yaml
        └── components.yaml        the official v0.9.0 manifest, vendored unchanged
```

**Kustomize, base and overlay.** Kustomize is built into `kubectl` (`kubectl kustomize`
renders, `kubectl apply -k` applies). The *base* holds everything that does not depend
on where the project runs. An *overlay* takes the base and changes only what does: here,
the image tags and that images are never pulled. A second cluster (the CI cluster of
phase 7, with images in a registry) would be a second overlay next to `kind/`, and the
base would not change. No Helm: Helm's templates solve the problem of a chart that many
strangers install with many options. This project has one user and one cluster, and
plain YAML that a person can read is worth more.

## How compose maps to Kubernetes

| docker-compose.yml | Here | What changed |
|---|---|---|
| `mlflow` service | `mlflow.yaml` | volume is a PersistentVolumeClaim; its "jobs" processes are switched off |
| `api` service | `api.yaml` | probes instead of HEALTHCHECK; replicas set by an autoscaler |
| `scheduler` + `prefect` services | `daily-step-cronjob.yaml` | a CronJob runs one step and exits; no Prefect server |
| `pipeline` service (`train`, `promote`) | not here | these need a git checkout; see "The daily step and retraining" |
| bind mounts of `data/` | `data-pvc.yaml` + `data-loader.yaml` | the data is copied into a volume once |
| `prometheus`, `pushgateway`, `grafana` | not here | see "Monitoring" |

## The data

The API reads two tables at startup (the gold demand and the silver weather), and the
daily step reads those plus the gold feature table and writes the replay's forecasts and
its reports. In compose, all of that is a folder of the host mounted into the
containers. Here it is one PersistentVolumeClaim, `demand-data`, mounted at `/app/data`:
read-only in the API, read-write in the daily step.

The data gets in by being *copied*, once, through a pod that only holds the volume open
(`data-loader.yaml`): `kubectl cp` an archive in, unpack it, delete the pod. The
alternative was to mount the project folder of the host into the kind node
(`extraMounts` in `kind-cluster.yaml`). The copy was chosen because:

* the same commands work on any cluster, not only on kind;
* the cluster cannot write into the project folder: its replay is its own, as the
  compose stack's is (`data/predictions-stack`). The cluster's MLflow numbers its
  model versions independently, so its replay must not share files with any other;
* there is no Windows path to get right in the kind configuration;
* the data is about 3 MB, so the copy costs nothing.

**The limit of this choice.** The volume is `ReadWriteOnce`: one *node* may mount it
(any number of pods on that node). kind has one node, so it works. On a cluster with
several nodes, an API pod and the daily step could land on different nodes, and the
second would wait forever to attach the volume. Such a cluster needs storage that many
nodes can mount (`ReadWriteMany`: NFS, a cloud file share) or the data in object storage
(S3 and the like) read by the code. That is a change for when there is such a cluster.

## MLflow

The server is the one of compose: the pipeline image, `mlflow server` with a SQLite
backend and the model files on a volume, *served* by the server (clients send and fetch
files over HTTP, so no other pod needs the volume). What is new in Kubernetes:

* **One replica, strategy `Recreate`.** A rolling update starts the new pod before
  stopping the old one; two servers on one SQLite file can corrupt it. `Recreate` stops
  the old pod first, at the cost of a few seconds without a server.
* **`--allowed-hosts` lists the Service's names.** MLflow refuses requests whose `Host`
  header it does not know. Pods of the namespace call `http://mlflow:5000`; the longer
  forms (`mlflow.nyc-taxi-demand.svc.cluster.local:5000`) are listed too, and
  `localhost:5000` for `kubectl port-forward`. `/health` is exempt from the check
  (tested), which is why the probes pass with the pod's IP as their `Host`.
* **`MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`.** MLflow 3.17 starts nine background
  processes for its GenAI job features at every server start. Measured with
  `docker stats`: 1.76 GiB with them, 366 MiB without. The project uses none of them.
* **`enableServiceLinks: false`, on every pod.** By default Kubernetes gives each pod
  variables for every Service of its namespace. The Service `mlflow` produces
  `MLFLOW_PORT=tcp://10.96.x.x:5000`, and `mlflow server` reads `MLFLOW_PORT` as its
  port: without an explicit `--port` it stops with "'tcp://...' is not a valid integer"
  (reproduced with `docker run`). The flag removes that whole class of surprise.

The store starts empty. A model gets into it by training on the host, against this
server, and passing the same promotion gate as everywhere else ("Running it", step 7).

## The API

**Probes.** Kubernetes ignores the `HEALTHCHECK` of the image and asks its own three
questions:

* *startup*: `/health`, every 5 s, up to 24 times (2 minutes). The service opens its port
  only when the model and the data are loaded (3 to 10 s warm, about 25 s cold), so this
  is "has it started?". While it runs, the other two probes wait, so a slow cold start
  is not mistaken for a hung process.
* *liveness*: `/health` every 10 s; three failures restart the container.
* *readiness*: `/ready` every 5 s; a failing pod gets no traffic but is not restarted.

If the model cannot be loaded (no champion yet, MLflow down), the service exits with
code 3, and Kubernetes restarts it with growing pauses (`CrashLoopBackOff`). It comes up
by itself when the cause is gone; `kubectl rollout restart` saves the wait.

**Resources, from measurements.** On the API image under Docker, with one CPU:
250 MiB after startup, 286 MiB at the peak (the cgroup's `memory.peak`) with 16
clients in parallel; a request for three zones takes 46 ms, and one pod serves about
18 such requests a second at one full CPU. So: memory request 384Mi and limit 512Mi
(the limit is about 1.8 times the peak; past it the pod is killed, `OOMKilled`); CPU
request 500m and limit 1. `OMP_NUM_THREADS=1`: LightGBM would start one thread per CPU
of the *node*, and under a one-CPU limit they only queue (measured: same throughput,
20 MiB less).

**The autoscaler.** The HorizontalPodAutoscaler keeps between 2 and 4 pods and aims at
70 % of the CPU *request* (about 350m per pod on average). It needs metrics-server,
which kind does not ship; `overlays/kind/metrics-server` installs v0.9.0 with
`--kubelet-insecure-tls`, because kind's kubelets have self-signed certificates that
metrics-server cannot verify. That flag is for a laptop cluster only. The manifest is
vendored (committed as downloaded, its checksum in the kustomization) rather than
referenced by URL: rendering then needs no network, and a review sees exactly what is
installed. The Deployment has no `replicas` field: the autoscaler owns that number, and
a number in the file would be written back at every `kubectl apply`.

**Rolling updates** add one new pod before removing an old one (`maxUnavailable: 0`),
and a pod being deleted waits 5 s (`preStop`) so that the Service stops sending it
requests before it stops answering.

## Security

Every pod runs as user 10001 (`runAsNonRoot`), with a read-only root filesystem, no Linux
capabilities, no way to gain privileges, and the container runtime's default seccomp
profile. What a program must write goes to an `emptyDir` volume (`/tmp`, and the home
folder for the batch image) or to its PersistentVolumeClaim. No pod gets a token for
the Kubernetes API (`automountServiceAccountToken: false`): none of them talks to it.
The namespace enforces the "restricted" Pod Security Standard, so the API server
refuses any pod here that does not meet these rules, including pods added later.

Left out, on purpose: NetworkPolicies (which pod may talk to which; the next step for a
shared cluster), an Ingress (one person reaches the services with `kubectl
port-forward`), PodDisruptionBudgets (one node: there is no drain to survive).

## Settings and secrets

Settings that are not secret are a ConfigMap of `DEMAND_*` variables (`kustomization.yaml`
of the base), which override `configs/config.yaml` inside the images exactly as in
compose. Kustomize names it with a hash of its content (`demand-settings-htgkd2d88h`):
change a value and the name changes, and the Deployment rolls its pods to read it. With
a fixed name, the change would reach only pods that happen to start later.

There is no secret in this stack today: the MLflow server has no login, and Grafana,
the one thing with a password in compose, is not in the cluster. The Secret
`demand-secrets` is generated empty and already read by the API and the daily step, so
the wiring exists. To add one later, without committing it:

1. Put the values in `k8s/overlays/kind/.env.secrets`, one `NAME=value` per line. The
   name matches `.env.*` in `.gitignore`, so git never sees it (`git check-ignore -v`
   confirms).
2. Add to `overlays/kind/kustomization.yaml`:
   ```yaml
   secretGenerator:
     - name: demand-secrets
       behavior: merge
       envs: [.env.secrets]
   ```
3. `kubectl apply -k k8s/overlays/kind`. The pods restart with the new variables.

Rendering then needs that file, so a fresh clone (and CI) would need its own. A Secret
is only base64-encoded in the cluster, not encrypted; on a real cluster the values would
come from a secret manager instead.

## The daily step and retraining

The CronJob `daily-step` runs `demand step` once a night at 01:00 New York time: one
call of `run_days` (forecast the next simulated day, score, monitor, retrain if due),
then the pod exits. In compose a Prefect scheduler does this. A CronJob needs no Prefect
server, and one short-lived pod per run is what Kubernetes is built for.

* `concurrencyPolicy: Forbid` is the Kubernetes form of Prefect's `CANCEL_NEW`: a run that
  comes due while one is still going is skipped, not queued.
* `backoffLimit: 0` and `restartPolicy: Never`: a failed run is **not** retried. The
  replay's clock is the folder of forecasts, so a run that failed after writing its
  forecast has already moved the clock, and a retry would do the next day, not this one.
* In the kind overlay the CronJob is suspended: each run moves the simulated clock by a
  day, so runs are started by hand with `kubectl create job --from=cronjob/daily-step`.

**Retraining is refused in the cluster, by design.** The promotion gate accepts only a
model trained from a clean git commit with a known data version (`promotion.py`). The
images carry the code but no `.git` folder and no `dvc.lock`, so in a pod the commit and
the data version are "unknown", and when a retrain comes due, `check_clean_checkout`
refuses it before training anything. The run then fails with the gate's message, after
the day's forecast and monitoring were written; `kubectl get jobs` shows it as Failed,
and `kubectl logs` says why. Because the alert stands and no retrain is recorded, every
later run fails the same way until the end of the replay: the Failed Jobs are the alarm
"a retrain is due and cannot be done here". The gate is not loosened to make the job
green. Making retraining possible in the cluster needs the commit and the data version
baked into the image at build time, from a clean checkout. That belongs with phase 7,
where CI builds the images from a commit.

## Monitoring

Prometheus, the Pushgateway and Grafana stay in compose. The plan does not list them
for Kubernetes, and the autoscaler does not need them: it reads CPU from
metrics-server. Doing them properly in a cluster means the Prometheus Operator and its
custom resources (installed with Helm), which is a large stack for a laptop and teaches
little about this project. The API still serves `/metrics`, which a port-forward shows.
The daily step is run without `--push-to`, since there is no Pushgateway to push to;
its numbers are in the monitoring report on the data volume.

## Images and pinning

* The project's images are `nyc-taxi-demand/api:local` and
  `nyc-taxi-demand/pipeline:local`, built by `docker compose build` and copied into the
  node with `kind load docker-image`; `imagePullPolicy: Never` stops the node from ever
  pulling. A local image has no registry digest to pin (it gets one when pushed), so
  the tag plus "load exactly this image, never pull" is what fixes it. The base names
  the images without a tag; every overlay must set one.
* metrics-server: `v0.9.0@sha256:d986...05c0`, the tag and the digest of its
  multi-platform index.
* The kind node: `kindest/node:v1.36.4@sha256:099e...faed`, the image kind v0.33.0 built
  for Kubernetes 1.36.4, the same minor as `kubectl` here (1.36.1).
* kubeconform, used to validate: `ghcr.io/yannh/kubeconform:v0.8.0@sha256:faff...f64e`.

## Validating without a cluster

From Git Bash in the project folder:

```bash
kubectl kustomize k8s/overlays/kind > /tmp/kind.yaml
kubectl kustomize k8s/overlays/kind/metrics-server > /tmp/metrics-server.yaml
MSYS_NO_PATHCONV=1 docker run --rm -v "$(cygpath -w /tmp):/m:ro" -v "$(cygpath -w "$PWD/k8s"):/k8s:ro" \
  ghcr.io/yannh/kubeconform:v0.8.0@sha256:faffaf43f95aa6425306e1ab8d6fcad72acb9049158f38e574c085ea1ec0f64e \
  -strict -summary -kubernetes-version 1.36.4 /m/kind.yaml /m/metrics-server.yaml /k8s/overlays/kind/data-loader.yaml
```

It prints `Summary: 21 resources found in 3 files - Valid: 21, Invalid: 0, Errors: 0,
Skipped: 0`. kubeconform downloads the schemas from GitHub while it runs, so it needs
the network. `kubectl apply --dry-run=client` does not work without a cluster: even in
client mode kubectl asks the API server which kinds of object exist.

## Running it

Needs: Docker Desktop running, `kind` v0.33.0 installed, the images built (`docker compose
build`), and the data built on the host (`dvc repro`). Commands are for PowerShell in
the project folder. The compose stack must be down (`docker compose down`): both use
port 5000, and the two together need more memory than Docker Desktop has.

**1. Create the cluster** (about a minute; the node image is pulled the first time).

```powershell
kind create cluster --config k8s/kind-cluster.yaml
kubectl cluster-info --context kind-nyc-taxi-demand
```

It ends with `Set kubectl context to "kind-nyc-taxi-demand"`; `kubectl` now talks to it.

**2. Load the images into the node** (a few minutes: 1.45 GB and 2.22 GB).

```powershell
kind load docker-image nyc-taxi-demand/api:local nyc-taxi-demand/pipeline:local --name nyc-taxi-demand
```

Each image: `Image: "nyc-taxi-demand/api:local" with ID "sha256:..." not yet present on
node "nyc-taxi-demand-control-plane", loading...`. After a rebuild, load again.

**3. Install metrics-server and apply the project.**

```powershell
kubectl apply -k k8s/overlays/kind/metrics-server
kubectl apply -k k8s/overlays/kind
kubectl wait -n kube-system --for=condition=Available deployment/metrics-server --timeout=180s
kubectl wait -n nyc-taxi-demand --for=condition=Available deployment/mlflow --timeout=180s
```

The second command lists every object as `created` (`namespace/nyc-taxi-demand created`,
`configmap/demand-settings-<hash> created`, ...). The API pods now restart in a loop
(`kubectl get pods -n nyc-taxi-demand` shows `Error` or `CrashLoopBackOff`): there is no
data and no model yet, and `kubectl logs -n nyc-taxi-demand deployment/api` says which.
That is expected and fixes itself in step 5.

**4. Copy the data in.**

```powershell
kubectl apply -f k8s/overlays/kind/data-loader.yaml
kubectl wait -n nyc-taxi-demand --for=condition=Ready pod/data-loader --timeout=120s
tar -cf "$env:TEMP\demand-data.tar" -C data gold silver/weather
Push-Location $env:TEMP
kubectl cp demand-data.tar nyc-taxi-demand/data-loader:/tmp/demand-data.tar
Pop-Location
kubectl exec -n nyc-taxi-demand data-loader -- tar -xf /tmp/demand-data.tar -C /app/data
kubectl exec -n nyc-taxi-demand data-loader -- ls -lR /app/data
kubectl delete pod -n nyc-taxi-demand data-loader
Remove-Item "$env:TEMP\demand-data.tar"
```

The listing shows `gold/demand_hourly.parquet`, `gold/features.parquet` and
`silver/weather/weather_hourly.parquet`, owned by `app`. (`kubectl cp` is given a bare
file name from the temp folder because the `C:` of a Windows path can be read as the
name of a pod.)

**5. Seed the model**: train on the host against the cluster's MLflow and promote it
through the gate. The host has what the gate needs (a clean commit, `dvc.lock`), as
long as `git status` shows nothing: commit first, or the gate refuses the model. (With
changes you are not ready to commit, train from a clean copy instead: `git worktree add
--detach <folder> HEAD`, link its `data` to the project's with `mklink /J`, and run
the commands below in that folder with `$env:PYTHONPATH = "<folder>\src"`, so the code
that trains is the code of the commit.) Open a second PowerShell window for the
port-forward and leave it running:

```powershell
kubectl port-forward -n nyc-taxi-demand svc/mlflow 5000:5000
```

Then, in the first window (the reports go to a temporary folder, so `reports/` of the
project is not touched; training needs the cleaning report there):

```powershell
New-Item -ItemType Directory -Force "$env:TEMP\nyc-k8s-reports" | Out-Null
Copy-Item reports\silver_cleaning_report.json "$env:TEMP\nyc-k8s-reports\"
$env:DEMAND_MLFLOW__TRACKING_URI = "http://localhost:5000"
$env:DEMAND_PATHS__REPORTS_DIR = "$env:TEMP\nyc-k8s-reports"
.venv\Scripts\python -m demand train
.venv\Scripts\python -m demand promote
Remove-Item Env:DEMAND_MLFLOW__TRACKING_URI, Env:DEMAND_PATHS__REPORTS_DIR
kubectl rollout restart -n nyc-taxi-demand deployment/api
kubectl rollout status -n nyc-taxi-demand deployment/api --timeout=300s
```

`train` ends with `"model_mae": 24.106025, "baseline_mae": 27.70067`, the numbers of the
host's model: same commit, same data, same model. `promote` makes version 1 champion;
`rollout status` ends with `deployment "api"
successfully rolled out`, and `kubectl get pods -n nyc-taxi-demand` shows two `api` pods
`1/1 Running`. Stop the MLflow port-forward (Ctrl+C) when done.

**6. Call the API.** In the second window:

```powershell
kubectl port-forward -n nyc-taxi-demand svc/api 8000:8000
```

and in the first (`curl.exe`, because `curl` in Windows PowerShell is another command):

```powershell
curl.exe "http://localhost:8000/ready"
curl.exe "http://localhost:8000/predict?date=2025-05-14&zone=161&zone=237"
```

`{"status":"ready","model_version":1}`, then 48 forecasts (24 hours, two zones) from
`"model_version":1`.

**7. Check the autoscaler** (metrics appear about a minute after the pods start):

```powershell
kubectl get hpa -n nyc-taxi-demand
kubectl top pods -n nyc-taxi-demand
```

`TARGETS` shows `cpu: 1%/70%` or similar, `MINPODS 2`, `MAXPODS 4`, `REPLICAS 2`. While it
shows `<unknown>/70%`, metrics-server has no numbers yet; if that lasts, check
`kubectl logs -n kube-system deployment/metrics-server`. To watch it scale, put load on
it (with the port-forward of step 6 running) and run `kubectl get hpa -n nyc-taxi-demand
--watch` in another window:

```powershell
.venv\Scripts\python -c "import urllib.request as r, concurrent.futures as f; u='http://localhost:8000/predict?date=2025-05-14&zone=161&zone=237&zone=236'; list(f.ThreadPoolExecutor(8).map(lambda _: r.urlopen(u).read(), range(3000)))"
```

A port-forward sends everything to *one* pod, so the new pods get no traffic; the
average still rises above 70 % and the HPA adds pods, which is what this shows. Spread
traffic needs a client inside the cluster. In the test run, `TARGETS` went from `0%/70%`
to `100%/70%` within 30 s, a third pod started 15 s later, and the average settled at
`67%/70%` with `REPLICAS 3`. When the load stops, the HPA waits five minutes (its
default stabilisation window) before it removes the extra pod.

**8. Run the daily step once.**

```powershell
kubectl create job -n nyc-taxi-demand --from=cronjob/daily-step daily-step-manual-1
kubectl wait -n nyc-taxi-demand --for=condition=Complete job/daily-step-manual-1 --timeout=600s
kubectl logs -n nyc-taxi-demand job/daily-step-manual-1
```

The log is JSON lines; the last one's message should read like `day done at
2025-05-01T00:00:00-04:00: forecast 2025-05-02; alerts none; warnings none; retrain
no_alert`. Each further run (a new job name each time) forecasts the next day. If a run
fails, `kubectl wait` runs into its timeout and `kubectl get jobs -n nyc-taxi-demand`
shows it as Failed.

In the host's replay a retrain came due on 30 May. The cluster's model comes from the
same commit and data, and the same happens there. In the test run, one Job of 40 steps
(the CronJob's spec with `args: ["step", "--days", "40"]`) took 98 s. It showed
`alerts accuracy` and `retrain waiting` on the last days of May. Then it failed at the
step of 30 May with the gate's refusal: "A retrain is due, but the working directory is
not a clean git checkout and the data version is unknown, so the gate could not promote
the new model. Nothing was trained." From then on every run fails the same way, as
explained in "The daily step and retraining".

**9. Delete the cluster.**

```powershell
kind delete cluster --name nyc-taxi-demand
```

This removes the node and everything in it, including the cluster's MLflow store and
its replay. Nothing on the host is changed: the data was copied in, not mounted.

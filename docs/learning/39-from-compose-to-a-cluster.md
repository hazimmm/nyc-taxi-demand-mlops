# Chapter 39 – From Compose to a cluster: the objects of Kubernetes

Chapter 33 started the whole stack with one file on one machine. That is enough for a laptop. A real service runs on several machines. It must keep answering when one of them fails, and it needs more copies when many requests arrive. Docker Compose does none of this. Kubernetes does. This chapter starts Part VIII. It explains the objects of Kubernetes one at a time, with the files of the folder [`k8s/`](../../k8s/README.md) as the examples. Chapter 40 shows what happened when they ran.

## What you will learn

- What a cluster is, and the idea of a desired state.
- Pods, Deployments and Services, and how they replace a Compose service.
- Three probes, and requests and limits taken from measurements.
- An autoscaler that adds pods when the CPU is busy.
- A volume claim for the data, and its limit.
- Settings in a ConfigMap whose name changes with its content, and a Secret that is empty.
- The daily step as a CronJob, and why it is never retried.
- One base and one overlay with Kustomize, and a namespace that checks the rules.

## A cluster and its desired state

A **cluster** is a group of machines that Kubernetes treats as one. Each machine is a **node**. The **control plane** is the part that decides: it keeps the records, chooses a node for each container, and watches what happens. You talk to it through its **API server**, with the command-line tool `kubectl`.

You do not tell Kubernetes "start this container". You describe what should exist, as an **object**, such as "two copies of the API, from this image". A file of objects is a **manifest**, written in YAML, and `kubectl apply` sends it to the API server. Small programs of the control plane, called **controllers**, then act until what exists matches what you described. When a container dies, a controller sees one copy too few and starts a new one. This is the **desired state**: you write the goal, and the cluster keeps working towards it.

On this laptop the cluster is **kind**, short for "Kubernetes in Docker". kind runs a whole node as one Docker container. The file [`k8s/kind-cluster.yaml`](../../k8s/kind-cluster.yaml) asks for one node, from the image `kindest/node:v1.36.4`, pinned by digest (Chapter 32). It is 1.36, not kind's default 1.37, because `kubectl` here is 1.36.1, and the same minor version is the safe choice.

## From services to objects

Compose has one word, service, for a container with its settings. Kubernetes splits it into several objects. Figure 39.1 shows where each part of `docker-compose.yml` went.

| In `docker-compose.yml` | In `k8s/` | What changed |
|---|---|---|
| service `mlflow` | `base/mlflow.yaml` | the volume is a PersistentVolumeClaim; MLflow's background jobs are off |
| service `api` | `base/api.yaml` | probes instead of the image's health check; the number of copies is set by an autoscaler |
| services `scheduler` and `prefect` | `base/daily-step-cronjob.yaml` | a CronJob runs one step and exits; no Prefect server |
| service `pipeline` (`train`, `promote`) | not in the cluster | they need a git checkout (Chapter 40) |
| bind mounts of `data/` | `base/data-pvc.yaml` | the data is copied into a volume once |
| `prometheus`, `pushgateway`, `grafana` | not in the cluster | see "Left out on purpose" |

*Figure 39.1 – Compose against Kubernetes, from the table in [`k8s/README.md`](../../k8s/README.md).*

### Pods

A **Pod** is the smallest thing Kubernetes runs: one or more containers that share a network address and volumes. Here every pod has one container. A pod is never repaired: a new pod with a new name replaces it.

### Deployments

A **Deployment** keeps a number of identical pods running from one template. Each copy is a **replica**. The Deployment finds its pods by a **label**, a key and a value on an object, here `app.kubernetes.io/name: api`. A **selector** is the question "which pods have this label?". When you change the template, for example the image, the Deployment replaces the pods step by step. This is a **rolling update**.

The API's Deployment adds one new pod before it removes an old one (`maxSurge: 1`, `maxUnavailable: 0`). So the number of pods that answer never drops during an update. This is how a new champion reaches the service: `kubectl rollout restart deployment/api` starts new pods, and each loads the champion once at startup (Chapter 31).

MLflow's Deployment is different. It has exactly one replica and the strategy `Recreate`: the old pod stops before the new one starts. Its store is one SQLite file. A rolling update would run two servers on that file for a moment, and two writers can corrupt it. The cost of `Recreate` is a few seconds with no server.

### Services

Each new pod has a new address. A **Service** is a fixed name and address in front of all the pods that match a selector, and it spreads the requests over those that are ready. The Services are reached as `http://api:8000` and `http://mlflow:5000`, the names of Compose, so the settings need no change. Their type is `ClusterIP`: reachable only inside the cluster. From the laptop, `kubectl port-forward svc/api 8000:8000` opens a tunnel from port 8000 of your machine into the cluster.

A trap was found on the way. By default, Kubernetes gives every pod a variable for every Service, and the Service `mlflow` gives `MLFLOW_PORT=tcp://10.96.x.x:5000`. But `mlflow server` reads `MLFLOW_PORT` as its own port, and stops with "'tcp://...' is not a valid integer" (reproduced with `docker run`). So every pod sets `enableServiceLinks: false`, and the server gets an explicit `--port=5000`.

## Three probes

Kubernetes ignores the `HEALTHCHECK` of the image (Chapter 32). It asks its own questions, called **probes**: automatic requests to the container at fixed intervals. Chapter 31 introduced two of them. The API has three.

| Probe | Asks | Every | Gives up after | On failure |
|---|---|---|---|---|
| startup | `/health` | 5 s | 24 failures (2 minutes) | the container is restarted |
| liveness | `/health` | 10 s | 3 failures (30 s) | the container is restarted |
| readiness | `/ready` | 5 s | 2 failures | the pod gets no requests, and is not restarted |

*Figure 39.2 – The probes of the API, from [`base/api.yaml`](../../k8s/base/api.yaml).*

Why a startup probe? The service opens its port only after it has loaded the model and the data: 3 to 10 seconds warm, about 25 seconds cold. While the startup probe runs, the other two wait, so a slow start is not killed as a hung process. If the model cannot be loaded at all, the service exits with code 3 (Chapter 31), and the Deployment starts it again with growing pauses: the state **CrashLoopBackOff** of Chapter 33.

## Requests, limits, and numbers that were measured

Each container states its **resources**. The **request** is what the cluster reserves for it on a node. The **limit** is the most it may use. CPU is counted in thousandths of a CPU: `500m` is half a CPU. A container that wants more CPU than its limit is slowed down. A container that uses more memory than its limit is killed, and Kubernetes shows the reason `OOMKilled`, "out of memory".

The numbers were not guessed. Under Docker, with one CPU and the restrictions of the pods, the API used 250 MiB after startup and 286 MiB at its peak with 16 clients. One pod served about 18 requests for three zones a second at one full CPU.

| Container | CPU request | CPU limit | Memory request | Memory limit |
|---|---|---|---|---|
| api | 500m | 1 | 384Mi | 512Mi |
| mlflow | 100m | 1 | 512Mi | 1Gi |
| daily-step | 500m | 1 | 512Mi | 1Gi |

*Figure 39.3 – Resources. The API's memory limit is about 1.8 times its measured peak.*

`OMP_NUM_THREADS=1` came from a measurement too. LightGBM starts one thread per CPU of the node, and under a limit of one CPU they only wait for each other. With one thread the throughput was the same, and the memory 20 MiB lower.

### MLflow's hidden gigabyte

The MLflow server was measured too, and the first number was a surprise: 1.76 GiB while doing nothing. MLflow 3.17 starts nine background processes for its "jobs", features for generative AI that this project never uses. With the variable `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` the server used 366 MiB. 1.76 GiB is more than the limit of 1Gi in Figure 39.3, so without the variable the pod would most likely be killed as `OOMKilled`; that was not tried. The same variable is now set in `docker-compose.yml`, so the Compose stack saves the memory as well.

## An autoscaler

A **HorizontalPodAutoscaler**, or **HPA**, changes the number of replicas of a Deployment from a measurement. "Horizontal" means more copies, not bigger ones. The API's HPA keeps between 2 and 4 pods and aims at an average CPU use of 70 %.

The percentage is of the *request*, not of a whole CPU. 70 % of 500m is 350m. Every 15 seconds the HPA reads the CPU use of each pod and computes, in a simplified form:

```text
average     = CPU used by all pods / CPU requested by all pods
wanted pods = current pods × average / 70 %, rounded up
```

So with 2 pods and an average of 105 %, it wants 2 × 105 / 70 = 3 pods. Chapter 40 shows this rule at work, with a result that surprises at first.

Two pods at least, so that one can restart while the other answers; four at most, because on one laptop more pods only fight for the same CPUs. The Deployment has no `replicas` field, on purpose: the HPA owns that number.

The HPA reads the CPU use from **metrics-server**, an add-on that measures every pod. kind does not ship it. [`overlays/kind/metrics-server`](../../k8s/overlays/kind/metrics-server/kustomization.yaml) installs version 0.9.0 from its official file, downloaded once and committed unchanged with its checksum. This is **vendoring**: nothing is fetched at install time, and a review sees exactly what is installed. One flag is added, `--kubelet-insecure-tls`, because kind's nodes use certificates that metrics-server cannot check. It is for a laptop only.

## The data on a volume

The nodes of a cluster do not see the folders of your machine. So the data lives on a **PersistentVolumeClaim**, or **PVC**: a request for storage that outlives every pod. The cluster creates a **PersistentVolume** that fits it; on kind, a folder inside the node. The PVC `demand-data` (1 GiB) is mounted at `/app/data`: read-only in the API, read-write in the daily step. MLflow has its own, `mlflow-store`. The data is copied in once (Chapter 40), so the cluster's replay stays apart from your files.

The access mode is **ReadWriteOnce**: one *node* may mount the volume, not one pod. kind has one node, so the API pods and the daily step share it. On several nodes, the second pod to need it could wait forever. Such a cluster needs storage that many nodes can mount (`ReadWriteMany`) or the data in object storage. This limit is not solved here.

What a program writes that is not data goes into an **emptyDir**, a folder that lives as long as its pod, such as `/tmp` for the model download.

## Settings and secrets

Settings that are not secret go into a **ConfigMap**: an object of names and values. Here it holds three `DEMAND_*` variables. They override `configs/config.yaml` inside the images, exactly as in Compose (Chapter 7):

```yaml
configMapGenerator:
  - name: demand-settings
    literals:
      - DEMAND_MLFLOW__TRACKING_URI=http://mlflow:5000
      - DEMAND_PATHS__DATA_DIR=/app/data
      - DEMAND_PATHS__REPORTS_DIR=/app/data/reports
```

Kustomize (below) adds a hash of the content to the name, so the object is called `demand-settings-htgkd2d88h`. Change one value, and the name changes. The Deployment then points at a new name, so its template has changed, and it rolls its pods by itself. With a fixed name, only the pods that happened to start later would read the new value.

A **Secret** is the same idea, for passwords and keys. This stack has none today, so `demand-secrets` is generated empty, and the pods already read it. A later secret goes into a file matching `.env.*`, which git ignores ([`k8s/README.md`](../../k8s/README.md) gives the steps). A Secret is only **base64**-encoded, a form anyone can decode, not encrypted.

## The daily step as a CronJob

A **Job** runs a pod until it finishes, and records whether it succeeded. A **CronJob** creates a Job on a cron schedule (Chapter 38). The CronJob `daily-step` has the schedule `"0 1 * * *"` and `timeZone: America/New_York`, the clock of the data.

Each run calls a new command, `step`, which was added to [`cli.py`](../../src/demand/cli.py) for this. With the type hints shortened:

```python
@app.command()
def step(days: int = 1, push_to: str | None = None) -> None:
    """Run the daily step (forecast, monitor, retrain if due) once, without Prefect."""
    _run_stage("demand.flows.orchestration", "run_days", days, push_to)
```

It calls `run_days` of Chapter 38, the function that Prefect's flow calls too, so no logic exists twice, and a new test checks that it passes its values on. The pod exits when the step is done, and no Prefect server is needed.

Two settings carry the rules of Chapter 38.

- `concurrencyPolicy: Forbid`. A run that comes due while another is still going is skipped. This is Prefect's `CANCEL_NEW`.
- `backoffLimit: 0` and `restartPolicy: Never`. A failed run is not retried, neither as a new pod nor in the same pod. The clock is the folder of forecasts. A run that failed after it wrote its forecast has moved the clock, and a retry would do the next day.

In the kind overlay the CronJob is **suspended**: the schedule starts no run, because each run moves the simulated clock by a day. A suspended CronJob still works as a template: `kubectl create job --from=cronjob/daily-step daily-step-manual-1` starts one run by hand.

## One base and one overlay

**Kustomize** builds the final manifests from a starting set and a list of changes. It is part of `kubectl`: `kubectl kustomize` prints the result, and `kubectl apply -k` applies it. The **base** holds what is the same on every cluster. An **overlay** takes the base and changes only what differs on one cluster. The kind overlay changes three things.

1. The image tags: `nyc-taxi-demand/api:local` and `nyc-taxi-demand/pipeline:local`, the tags that `docker compose build` makes.
2. `imagePullPolicy: Never` on every container, because the images are copied into the node and exist in no registry.
3. `suspend: true` on the CronJob.

Another cluster would be a second overlay, and the base would not change. Helm, the best-known tool for this, was not used: its templates serve packages that many strangers install with many options, and one user with one cluster is better served by plain YAML. A local image has no registry digest to pin (Chapter 32). What fixes it is that `kind load` copies exactly this machine's image into the node, and `Never` stops the node from fetching another.

## A namespace with rules

A **namespace** is a named group of objects inside a cluster. Everything of the project is in `nyc-taxi-demand`. Its label turns on the **Pod Security Standard** "restricted". The API server then refuses any pod in it that could run as root, keep Linux capabilities, gain privileges, or run without a seccomp profile. **Seccomp** is a Linux filter of the requests that a process may make to the kernel; `RuntimeDefault` is the standard filter of the container runtime.

Every pod in `k8s/` passes, with the settings of Chapter 33 and a little more.

| Chapter 33, Compose | Here, every pod |
|---|---|
| user 10001 from the image | `runAsNonRoot: true`, `runAsUser: 10001` |
| `read_only: true` | `readOnlyRootFilesystem: true` |
| `cap_drop: ["ALL"]` | `capabilities: drop: ["ALL"]` |
| `no-new-privileges:true` | `allowPrivilegeEscalation: false` |
| Docker's default filter | `seccompProfile: RuntimeDefault` |
| – | `automountServiceAccountToken: false` |

*Figure 39.4 – The same protections, now checked by the cluster.*

The last line is new: no pod gets a token to talk to the API server, because none needs one. The label turns "we wrote it that way" into "the cluster checks it", also for pods added later.

## Left out on purpose

- **No NetworkPolicy**, a rule of which pod may talk to which. Today every pod of the namespace can call every other one. On a shared cluster, this is the next step.
- **No Ingress**, the object that brings requests from outside to a Service. One person uses `kubectl port-forward`.
- **No monitoring in the cluster.** Prometheus, the Pushgateway and Grafana stay in Compose; in a cluster they would need the Prometheus Operator, a large stack for a laptop. The HPA reads metrics-server, and the daily step runs without `--push-to`.
- **No PodDisruptionBudget**, which limits how many pods may be stopped at once for maintenance. One node has none to survive.

## Try it yourself

This needs no cluster. Render the kind overlay and list its objects.

```powershell
kubectl kustomize k8s/overlays/kind | Select-String "^kind:"
```

You should see eleven lines: `Namespace`, `ConfigMap`, `Secret`, two of `Service`, two of `PersistentVolumeClaim`, two of `Deployment`, then `CronJob` and `HorizontalPodAutoscaler`. Now look for the generated name and the images, in the overlay and in the base.

```powershell
kubectl kustomize k8s/overlays/kind | Select-String "demand-settings-|image:"
kubectl kustomize k8s/base | Select-String "image:"
```

The first command should show `demand-settings-htgkd2d88h` three times (the ConfigMap itself, and its use in the API and in the CronJob) and three images, each ending in `:local`. The second should show the same three images without a tag. That is the whole difference: the base names the images, and the overlay chooses which ones. Last, look at the new command.

```powershell
.venv\Scripts\python.exe -m demand step --help
```

You should see `Run the daily step (forecast, monitor, retrain if due) once, without Prefect.` and the two options `--days`, with `[default: 1]`, and `--push-to`.

## Key takeaways

- In Kubernetes you describe the desired state as objects, and controllers work until the cluster matches it.
- A Compose service becomes a Deployment of pods and a Service in front of them. The API rolls one pod at a time; MLflow, with one SQLite file, is recreated.
- Three probes: startup protects a slow start, liveness restarts a hung process, readiness only stops traffic.
- Requests and limits come from measurements, which found 1.4 GiB that MLflow used for nothing. The HPA keeps 2 to 4 API pods at 70 % of their CPU request.
- The data is on a ReadWriteOnce claim, which works on one node only.
- The daily step is a CronJob that calls `demand step`: never two runs at once, never a retry.
- One base and one overlay for kind, in a namespace that refuses any pod that breaks the "restricted" rules.

## Exercises

**1.** A colleague adds `replicas: 2` to the API's Deployment, "to be explicit". What happens on a busy afternoon?

<details>
<summary>Answer</summary>

Suppose the HPA has scaled to 4 pods because the CPU is busy. Then someone runs `kubectl apply` for another change. The file says 2, so the Deployment drops to 2 pods while the load is high. A few seconds later the HPA sees a high average and scales up again. The file and the autoscaler fight over one number, and every apply removes capacity at the worst time. Only one owner may set the number, and here that is the HPA.

</details>

**2.** Why does the liveness probe ask `/health` and not `/ready`?

<details>
<summary>Answer</summary>

A failed liveness probe restarts the container. A restart helps only when the process itself is broken. "Not ready" can have a cause outside the pod, and a restart does not fix that. It only throws away a process that would answer again soon, and it adds a slow start. `/health` needs nothing but the process, so it fails only when a restart can help. A pod that is not ready only stops getting requests, which is the right reaction.

</details>

**3. Interview question.** "How would you move a service from Docker Compose to Kubernetes?"

<details>
<summary>Points of a good answer</summary>

- Map each part: Deployment and Service for the service, PersistentVolumeClaims for volumes, ConfigMaps and Secrets for settings, a CronJob for scheduled work.
- Probes for start, life and readiness; requests and limits from measurements, not guesses.
- An autoscaler that owns the replica count, with a minimum of two.
- The same protections as in the container, enforced by the namespace.
- One base and one overlay per cluster, so differences are small and visible.
- Name what is left out (network rules, ingress, storage for several nodes) instead of hiding it.

</details>

---

[Previous: Chapter 38 – A clock that starts the day: scheduling with Prefect](38-a-clock-that-starts-the-day.md) | [Contents](README.md) | [Next: Chapter 40 – Running it on kind: what happened](40-running-it-on-kind.md)

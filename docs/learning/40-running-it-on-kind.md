# Chapter 40 – Running it on kind: what happened

Chapter 39 described the objects. A file that describes a cluster proves little until a cluster runs it. On 24 October 2026 the lead engineer created a kind cluster and followed every step of the **runbook** in [`k8s/README.md`](../../k8s/README.md): the list of commands to run, in order, with what each should print. This chapter tells what happened, with the real outputs. Some failures were expected, one was a trap of the shell, and one was an alarm that was designed on purpose. The chapter closes Part VIII.

## What you will learn

- What was checked before any cluster existed.
- Why the API pods crashed after the first `apply`, and why that was correct.
- How the data got into the cluster, and a trap of Git Bash.
- How a model was trained for the cluster without loosening the gate.
- What the autoscaler did under load, and the arithmetic behind a surprising result.
- How the daily step ran as a Job, and why forty days ended in a failure that is an alarm.
- What is still not done.

## Checked before running

Every manifest was rendered with `kubectl kustomize` and checked with **kubeconform**, a tool that compares each object with the official schemas of Kubernetes 1.36. It printed `Valid: 21, Invalid: 0`. Why not `kubectl apply --dry-run=client`? Even in that mode, `kubectl` asks the API server which kinds of object exist, so it needs a cluster.

The tests of the project passed too: 1004 tests, with a coverage of 99.93 %, and ruff and mypy found nothing. But a schema check says only that a file is well formed, not that the system works. For that, a cluster had to run it.

## A cluster in one minute

kind v0.33.0 was installed, and the checksum of the download was compared with the published one (`4b22adaa…c9fc`) before it ran (Chapter 15). `kind create cluster --config k8s/kind-cluster.yaml` took about a minute, and the node ran Kubernetes 1.36.4.

The images were rebuilt first. The old images had been built before the command `step` existed, and a CronJob that runs them would fail with "No such command 'step'". Then both images were copied into the node:

```powershell
kind load docker-image nyc-taxi-demand/api:local nyc-taxi-demand/pipeline:local --name nyc-taxi-demand
```

The node has its own image store and does not see Docker Desktop's images. `kind load` copied 1.45 GB and 2.22 GB.

## Apply, and a service that waits

metrics-server was applied first, and then the project:

```powershell
kubectl apply -k k8s/overlays/kind
```

The command printed one line per object: the namespace, the ConfigMap `demand-settings-htgkd2d88h`, the Secret `demand-secrets-46f8b28mk5`, the Services `api` and `mlflow`, the PVCs `demand-data` and `mlflow-store`, the Deployments `api` and `mlflow`, the CronJob `daily-step` and the HPA `api`. Each line ended in `created`.

Right after that, `kubectl get pods -n nyc-taxi-demand` showed the API pods in `Error`, and soon in `CrashLoopBackOff`. The MLflow store and the data volume were empty, so the service exited with code 3, as Chapter 31 designed it. Kubernetes waits 10 seconds before the first restart and doubles the pause each time, up to five minutes: the `Restarting (3)` of Compose (Chapter 33). Nothing had to be fixed in the API. It needed data and a model.

## The data, copied in

The data is copied into the volume once. A small pod, [`data-loader`](../../k8s/overlays/kind/data-loader.yaml), does nothing but hold the volume open. The data is packed into a tar archive, copied into the pod with `kubectl cp`, and unpacked into `/app/data`. Then the pod is deleted. The listing in the pod showed three files, each owned by the user `app`:

| File | Bytes |
|---|---|
| `gold/demand_hourly.parquet` | 491,793 |
| `gold/features.parquet` | 2,533,792 |
| `silver/weather/weather_hourly.parquet` | 51,345 |

*Figure 40.1 – The data in the cluster: about 3 MB.*

The owner is `app` because the pod sets `fsGroup: 10001`: Kubernetes makes the volume writable for that group when it mounts it.

### A trap of Git Bash

The engineer ran this step from Git Bash, not from PowerShell as in the runbook. The copy failed:

```text
pods "nyc-taxi-demand\data-loader;C" not found
```

The command was `kubectl cp ... nyc-taxi-demand/data-loader:/tmp/demand-data.tar`. Git Bash is built on **MSYS**, a layer that lets Unix tools run on Windows. When it starts a Windows program, MSYS rewrites every argument that looks like a Unix path into a Windows path. `...:/tmp/...` looked like a list of paths, so MSYS turned it into `nyc-taxi-demand\data-loader;C:\...`. `kubectl` then read everything up to the colon as the name of a pod. The fix is a variable that turns the rewriting off: `MSYS_NO_PATHCONV=1`. PowerShell does not rewrite arguments, so the runbook works there as written.

The lesson: when a tool says that a name does not exist, read the name it printed.

## A model for the cluster, through the same gate

The cluster's MLflow starts empty, as the store of Compose did (Chapter 33). The images cannot train a model that the gate would accept: the gate needs a clean git commit and a known data version (Chapter 29), and an image has no `.git` folder. So the model is trained on the host, against the cluster's MLflow. A port-forward in a second window opens the way:

```powershell
kubectl port-forward -n nyc-taxi-demand svc/mlflow 5000:5000
```

Training then runs with `DEMAND_MLFLOW__TRACKING_URI=http://localhost:5000`.

There was a problem. The working folder had changes that were not committed: the files of this phase. `git status --porcelain` was not empty, so the gate would refuse any model trained there. There were two wrong ways out. One was to commit unfinished work just to train. The other was to loosen the gate. The engineer did neither.

Instead, the model was trained from a clean copy of the last commit, `111a03e`. A **git worktree** is a second working folder of the same repository, checked out at a commit you choose.

```powershell
git worktree add --detach <folder> HEAD
```

`--detach` means that no branch is checked out, only the commit. Three more things made the copy complete.

- Its `data` folder became a **junction** to the project's data, made with `mklink /J`. A junction is a Windows link: the folder points to another folder, so the 3 MB were not copied, and `dvc.lock` of the commit described exactly these files.
- `PYTHONPATH` pointed to the worktree's `src`. The editable install of Chapter 5 points to the `src` of the main folder, which had the uncommitted changes. With `PYTHONPATH` set, Python imported the code of the commit, so the code that trained was the code that the record names.
- The reports went to a temporary folder, so `reports/` of the project was not touched.

Then `train` and `promote` ran. The training took 103.56 seconds, and the record in the cluster's MLflow said:

| Field | Value |
|---|---|
| version | 1 |
| `git_dirty` | `false` |
| data version | `7bacf1bf80cf4612d40552e74ef11cc5.dir` |
| training rows | 53,280 |
| model MAE | 24.106025 |
| baseline MAE | 27.70067 (the model is 12.977 % better) |

*Figure 40.2 – The cluster's first model version.*

`promote` printed: `Promoted ... There was no champion, so only the baseline had to be beaten.`

The MAE is exactly that of the host's model and of the Compose stack's model (Chapter 33). This is the third time: the same commit and the same data gave the same model.

Was the gate bypassed? No. The rule says that a model must come from a clean commit with a known data version. That was true: the record says `git_dirty: false`, and the commit and the data version can be checked by anyone. The engineer changed *where* the model was trained, not the rule.

## The API answers

The API pods would have come up by themselves at their next restart; `kubectl rollout restart deployment/api` saved the wait. The rollout finished with two API pods `1/1 Running`. The old pod had restarted 5 times while it waited for a model. Through a port-forward to the Service `api`:

```text
GET /ready
{"status":"ready","model_version":1}
GET /predict?date=2025-05-14&zone=161&zone=237
48 forecasts, issued_at 2025-05-13T00:00:00-04:00
first: zone 161, hour 00:00, forecast 47.33
```

48 forecasts are 24 hours for two zones. The first one is the 47.33 of Figure 33.1.

`kubectl top pods`, which reads metrics-server, showed the API pods at 229 MiB and 241 MiB, and MLflow at 409 MiB. These fit the measurements of Chapter 39: about 250 MiB for the API, and 366 MiB for MLflow with its jobs off, measured with an empty store. This one held a model. The HPA first showed `<unknown>/70%`, because metrics-server had no numbers yet. A minute later it showed real ones.

## The autoscaler under load

A small Python command sent 3,000 requests, 8 at a time, each for three zones, through the port-forward. The HPA was read every 15 seconds:

| Reading | TARGETS | REPLICAS |
|---|---|---|
| 1 | 0%/70% | 2 |
| 2 | 5%/70% | 2 |
| 3 | 100%/70% | 2 |
| 4 | 100%/70% | 3 |
| 5 | 100%/70% | 3 |
| 6 | 67%/70% | 3 |
| 7 | 67%/70% | 3 |
| 8 | 66%/70% | 3 |

*Figure 40.3 – The HPA under load, one reading every 15 seconds.*

A third pod started, as planned. But look at the numbers. Why 100 %, and why did the average fall to 67 % when the new pod did no work?

A port-forward does not spread requests. It picks one pod when it starts and sends everything there. So one pod got all the requests, and it ran at about one CPU, its limit. Its request is 500m, so that pod was at 1000m / 500m = 200 % of its own request. The other pod was nearly idle. The HPA uses the average over all pods:

```text
2 pods: average = (1000m + about 0m) / (500m + 500m) = 100 %
        wanted  = 2 × 100 / 70 = 2.86, rounded up: 3 pods
3 pods: average = (1000m + 0m + 0m) / (500m × 3) = 67 %
        wanted  = 3 × 67 / 70 = 2.87, rounded up: 3 pods
```

The average fell because the third pod added 500m of request, not because it took work. It got no requests through the port-forward. The busy pod was as busy as before. The last readings stayed at 3 pods, because 67 % is below 70 % and the rule still rounds up to 3. Readings 4 and 5 still showed 100 %, probably because the new pod had no measurement yet.

So this test proves that the HPA reacts and adds a pod. It does not prove that more pods carry more load. Real clients would reach the pods through the Service, which spreads the requests, and then each of three pods would do a third of the work. A test of that needs a client inside the cluster, and it was not run.

When the load stops, the HPA waits five minutes before it removes a pod. This is its default **stabilisation window**: a pause that stops a short quiet moment from shrinking the number of pods. The scale-down was not timed for this book.

## The daily step, once

The CronJob is suspended in the kind overlay, so a run was started by hand:

```powershell
kubectl create job -n nyc-taxi-demand --from=cronjob/daily-step daily-step-manual-1
```

The Job was `Complete` after 18 seconds. The last line of its log was:

```text
day done at 2025-05-01T00:00:00-04:00: forecast 2025-05-02; alerts none; warnings none; retrain no_alert
```

The replay in the cluster starts on 2 May, as on the host (Chapter 34). Its files are on the data volume, apart from the host's replay and the stack's.

## Forty days, and an alarm by design

What happens when a retrain comes due in the cluster? To find out, the engineer ran forty days in one Job. `kubectl create job ... --dry-run=client -o json` builds the Job from the CronJob and prints it, without sending it to the cluster. In that file the arguments `["step"]` were changed to `["step", "--days", "40"]`, and the file was applied.

The Job ran for 98 seconds. Day after day went as on the host:

- In May, `input_drift` warnings appeared, as in Chapter 36.
- On 28 and 29 May the log said `alerts accuracy; retrain waiting`. The reason: the retrain "needs the champion's scored forecasts of 2025-05-01 to 2025-05-28; 1 of these 28 days have none". 1 May is never forecast (Chapter 34), so the retrain waited (Chapter 37).
- At the step of 30 May, the retrain was due. The run stopped with:

```text
Error: A retrain is due, but the working directory is not a clean git checkout and the data version is unknown, so the gate could not promote the new model. Nothing was trained. Commit the code, build the data with `dvc repro`, and run the step again.
```

`kubectl get jobs` showed the Job as `Failed`. Because of `backoffLimit: 0`, Kubernetes did not retry it. A retry would have moved the clock one more day, as Chapter 38 explained.

The alert and the retrain day are the same as in the host's replay: an accuracy alert, and a retrain due on 30 May. On the host the retrain then trained and was rejected (Chapter 37). In the cluster it was refused before training, by the function `check_clean_checkout`. The pod has no `.git` folder and no `dvc.lock`, so the commit and the data version are unknown.

This failure is the design, not a bug. The alert stays, and no retrain is recorded. So, by design, every later run would fail in the same way, and the list of Failed Jobs becomes the alarm: "a retrain is due, and it cannot be done here". This book did not run those later days. The gate is not loosened to make the Job green. The real fix is to bake the commit and the data version into the image when it is built from a clean checkout. That belongs to Phase 7, where CI builds the images from a commit.

## Deleting the cluster

```powershell
kind delete cluster --name nyc-taxi-demand
```

This removed the node and everything in it: the cluster's MLflow store, its model, and its replay. Nothing on the host changed, because the data had been copied in, not mounted.

## What is still not done

- **One node.** The ReadWriteOnce volumes would not work for the API and the daily step on several nodes (Chapter 39).
- **Retraining in the cluster is refused**, until Phase 7 bakes the commit and the data version into the image.
- **The load test went through one pod.** It showed the HPA adding a pod, not three pods sharing the work.
- **Left out on purpose** (Chapter 39): monitoring stays in Compose, and there is no NetworkPolicy, no Ingress, and no real Secret.

## Part VIII, closed

Phase 6 put the system on Kubernetes. Chapter 39 described the objects, and this chapter ran them on kind. The API came up after its expected crash loop. The model was trained from a clean commit and was the same model a third time. The HPA added a pod under load. And the daily step ended in the failure that was designed as its alarm.

Phase 7 builds the images in CI from a commit, which also makes retraining in a cluster possible.

## Try it yourself

This needs Docker Desktop, kind v0.33.0, the images (`docker compose build`) and the data (`dvc repro`). Stop the Compose stack first with `docker compose down`: both use port 5000, and together they need more memory than Docker Desktop has. Create the cluster, load the images, and apply.

```powershell
kind create cluster --config k8s/kind-cluster.yaml
kind load docker-image nyc-taxi-demand/api:local nyc-taxi-demand/pipeline:local --name nyc-taxi-demand
kubectl apply -k k8s/overlays/kind/metrics-server
kubectl apply -k k8s/overlays/kind
```

The first command should end with `Set kubectl context to "kind-nyc-taxi-demand"`, and the last one should list eleven objects as `created`. Wait a minute, and look at the pods and the reason.

```powershell
kubectl get pods -n nyc-taxi-demand
kubectl logs -n nyc-taxi-demand deployment/api
```

You should see the `mlflow` pod `Running` and two `api` pods in `Error` or `CrashLoopBackOff`, and a log that ends with a message like the one of Chapter 33: no model is registered under `taxi-demand`. To go on, follow steps 4 to 8 of "Running it" in [`k8s/README.md`](../../k8s/README.md). When you are done, delete the cluster.

```powershell
kind delete cluster --name nyc-taxi-demand
```

## Key takeaways

- A schema check proves the files are well formed. Only a cluster proves the system works.
- The API's crash loop after the first `apply` was correct: no data and no model, so exit code 3, and it came up once both existed.
- The data was copied into a volume, not mounted, so the cluster cannot change the host's files. Git Bash rewrites path-like arguments; `MSYS_NO_PATHCONV=1` stops it.
- The model was trained from a clean worktree of the last commit. The gate was not loosened, and the same commit and data gave the same MAE, 24.106025, a third time.
- The HPA averages CPU over all pods. Through a port-forward, one pod at 200 % of its request gave an average of 100 %, a third pod was added, and the average fell to 67 % with no work moved.
- Forty days in one Job reached the same alert and the same retrain day as on the host. The refused retrain made the Job fail, with no retry: an alarm by design, fixed only when the image carries its commit.

## Exercises

**1.** Suppose a client inside the cluster spreads its requests evenly, and the API needs 1200m of CPU in total. The HPA starts with 2 pods. How many pods does it end with?

<details>
<summary>Answer</summary>

With 2 pods: 1200m / 1000m = 120 %, and 2 × 120 / 70 = 3.43, rounded up to 4. With 4 pods: 1200m / 2000m = 60 %, and 4 × 60 / 70 = 3.43, rounded up to 4. So it stays at 4, the maximum. Three pods would not be enough: 1200m / 1500m = 80 %, above the target. Now the load is spread, so each pod does 300m of real work.

</details>

**2.** A colleague sees the Failed Job of 30 May and sets `backoffLimit: 3`, "so that it tries again". What happens?

<details>
<summary>Answer</summary>

The forecast of the day is written before the retrain check, so the clock has already moved. Each retry does the next day, and is refused again for the same reason, because the image still has no commit. One run would move the clock by up to four days and still fail. The cause is a fact of the image, not a passing problem, so a retry cannot help. It is the same lesson as Exercise 1 of Chapter 38.

</details>

**3.** Another colleague wants a green Job, and proposes a setting that makes `check_clean_checkout` treat an unknown commit as clean inside the cluster. Why is that wrong, and what is the right fix?

<details>
<summary>Answer</summary>

The gate would then promote models whose record says nothing true about their code or data. Nobody could rebuild them or explain them later, which is the reason the rule exists (Chapter 29). A green Job would hide the alarm. The right fix keeps the rule and supplies what it asks for: CI builds the image from a clean commit and writes the commit and the data version into it (Phase 7). The engineer did the same for the first model: the place of training changed, not the rule.

</details>

**4. Interview question.** "A pod is in CrashLoopBackOff. How do you find out why?"

<details>
<summary>Points of a good answer</summary>

- `kubectl describe pod` for the events, the last state, the exit code and the reason, such as `OOMKilled` or a failed probe.
- `kubectl logs`, and `kubectl logs --previous` for the container that crashed before the current one.
- Read the exit code: here code 3 means "no model or no data", a cause outside the pod.
- Check what the pod needs: the image (`ErrImageNeverPull` on kind), settings, volumes, the services it calls.
- Fix the cause, not the restart: the loop ends by itself once the cause is gone, and `kubectl rollout restart` saves the wait.

</details>

---

[Previous: Chapter 39 – From Compose to a cluster: the objects of Kubernetes](39-from-compose-to-a-cluster.md) | [Contents](README.md) | [Next: Appendix A – Stack map](appendix-a-stack-map.md)

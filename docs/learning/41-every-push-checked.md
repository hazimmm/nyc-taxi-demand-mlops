# Chapter 41 – Every push checked: continuous integration and known vulnerabilities

Until now, every check of this project ran by hand, on one Windows laptop. Chapter 13 called this a manual step: before a push, someone had to remember to run the tests. This chapter starts Part IX and Phase 7. It adds **continuous integration**: a service that runs every check on a fresh machine for every push, and says "green" or "red". On the way, the checks met Linux for the first time and found a test that failed only sometimes, and a new audit found known security problems in eleven libraries. Both are told here with the real outputs of 24 October 2026.

At the time of writing, the workflow has not yet run on GitHub. This chapter describes the workflow and the checks that were run on the local machine; the first run on GitHub is the real test.

## What you will learn

- What continuous integration is, and the words of GitHub Actions: workflow, job, step, runner, action and matrix.
- How a test that failed only sometimes was made to fail every time, and then fixed.
- Why every action is pinned by a commit hash, and what a supply-chain attack is.
- CVE, GHSA and PYSEC: the names of known vulnerabilities, and where they are kept.
- What pip-audit and Trivy found, and why nothing was upgraded on the same day.
- One file of accepted findings, read by two scanners, where every entry expires.
- Why the images are built without a layer cache.

## What CI is

**Continuous integration**, or **CI**, means that every change is checked automatically, on a clean machine, as soon as it is pushed. "Integration" is the moment your change meets the shared code. "Continuous" means it happens for every change, not once a month.

Why a clean machine? On your laptop, many things are there by accident: a library you installed last year, a file that git ignores, a setting in your shell. A test can pass because of them. A clean machine has only what the repository describes. If the checks pass there, they pass because of the repository, not because of your laptop. So a red result means "this commit is not fit to merge", and a green result means the same for everyone.

This project uses **GitHub Actions**, the CI service of GitHub. Its words are worth learning one by one.

- A **workflow** is one YAML file in the folder `.github/workflows/`. Here there is one: [`ci.yml`](../../.github/workflows/ci.yml).
- Its `on:` block says when it runs. Here: on a push to `dev` or `main`, on every **pull request** (a request to merge one branch into another), and by hand (`workflow_dispatch`).
- A workflow has **jobs**. Each job runs on its own **runner**: a virtual machine that GitHub starts for this one job and deletes afterwards. Jobs run at the same time unless one waits for another.
- A job is a list of **steps**, run in order. A step is either a shell command (`run:`) or an **action** (`uses:`): a reusable step that someone published in a repository, such as `actions/checkout`, which copies the repository onto the runner.
- A **matrix** runs the same job several times with different values. Here it builds two images with one job description.

The workflow has three jobs. Figure 41.1 shows them.

| Job | Runs in | What it does | Fails when |
|---|---|---|---|
| `checks` | the Python image of the Dockerfiles | installs `git` and `libgomp1`, installs `requirements.lock` with uv, then runs `ruff format --check`, `ruff check`, `mypy` and `pytest --cov` | any of the four fails, or coverage is below 80 % |
| `audit` | the same Python image | runs `tools/audit_dependencies.py`: pip-audit on the three lock files | a known vulnerability is found that is not accepted |
| `images` (matrix `api`, `pipeline`) | the runner itself | builds the image as `docker compose build` does, and scans it with Trivy | Trivy finds a HIGH or CRITICAL vulnerability with a fix that is not accepted |

*Figure 41.1 – The three jobs of [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml).*

The four commands of `checks` are exactly the "Checks" section of the [project README](../../README.md). CI does not have its own idea of "good"; it runs what a developer runs.

## First, the checks on Linux

The project had only ever been tested on Windows. A runner is Linux. So before any workflow was written, the lead engineer made a clean clone and ran all the checks inside the pinned image `python:3.12.15-slim-trixie`, the base of both Dockerfiles (Chapter 32). ruff and mypy passed. pytest did not:

```text
1 failed, 1003 passed
FAILED tests/unit/flows/test_orchestration.py::test_the_flow_runs_each_day_as_a_task_without_a_prefect_server
```

The engineer ran this one test alone. It passed. A second full run passed too. A test that passes and fails on the same code is a **flaky test**.

### Make it fail every time

It is tempting to say "it passed the second time" and move on. That is a mistake. A flaky test is a bug that has not been found yet. Sometimes the bug is in the test, sometimes in the code, and you cannot know which until you find it. And if the team learns that red can mean "just run it again", they will also run it again when red means a real bug.

So the first goal was not to fix the test, but to make it fail on purpose. What was different in the failed run? The full suite runs many heavy tests. The test that failed starts Prefect without a server, in ephemeral mode (Chapter 38): Prefect starts a temporary server, and that server first builds its database. Chapter 38 measured this start: 17 seconds on the laptop. A busy machine could be slower.

The idea, then: a slow, busy machine. The engineer ran the test in a container limited to 4 CPUs (`docker run --cpus 4`), with 4 extra processes that did nothing but loop, to keep those CPUs busy. The test ran 6 times. It failed 6 times of 6, always with the same message:

```text
RuntimeError: Timed out while attempting to connect to ephemeral Prefect API server.
```

Now the failure was not random any more. It had a cause that could be named.

### The cause and the fix

Prefect waits for its temporary server for a fixed time, the setting `PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS`. Its default is 20 seconds. On the laptop, with 12 CPUs, the start took about 17 seconds: only about 3 seconds to spare. A GitHub runner has far fewer CPUs. Under load the start took longer than 20 seconds, and Prefect gave up.

The code under test was not wrong. The limit belonged to the test's environment. So the fix is in the test fixture `prefect_home` of [`test_orchestration.py`](../../tests/unit/flows/test_orchestration.py), which already sets up Prefect for the tests:

```python
with temporary_settings(
    updates={
        PREFECT_HOME: home,
        PREFECT_SERVER_DATABASE_CONNECTION_URL: database,
        PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS: 120,
    },
    restore_defaults={PREFECT_API_URL},
):
    yield home
```

Two minutes instead of 20 seconds. Under exactly the same load, the test then passed 6 times of 6, in 32 to 43 seconds each. That is the proof: the same condition that failed every time now passes every time. The stack and the cluster are not affected, because they do not use ephemeral mode: Compose has a real Prefect server, and the CronJob runs without Prefect (Chapter 39).

Look at the order of the work: first make the failure happen every time, then find the cause, then fix the cause, then show that the same condition now passes. A retry would have made the run green, and hidden the 3 seconds of margin until the next slower machine.

## The workflow, part by part

Some settings of the workflow apply to all jobs.

```yaml
permissions:
  contents: read

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
```

Every job gets a **token**, a temporary password that lets it talk to GitHub. `permissions: contents: read` means this token can read the repository and do nothing else: it cannot push code, change a release or comment. If a step is ever tricked into running bad code, the token it can steal is nearly useless. The `concurrency` block says: when you push again to the same branch, cancel the older run that is still going. Its result would be about an old commit, so it only wastes time.

Each job has `timeout-minutes` (30, 15 and 30), so a job that hangs is stopped instead of running for GitHub's default of six hours.

### The job `checks`

The job runs inside a container, from the same image as the Dockerfiles, pinned by the same digest. So the tests run on the Python and the Debian that the images run on. Then four steps.

1. `apt-get install git libgomp1`. git comes before the checkout, so that the checkout is a real repository: some training tests make commits in a temporary repository. `libgomp1` is the library that LightGBM needs for its threads, as in the images.
2. `actions/checkout`, with `persist-credentials: false`. By default, checkout leaves the token in the folder's git settings, where any later step could read it. Nothing here needs git to talk to GitHub again, so the token is not kept.
3. `uv pip install --system -r requirements.lock`, then the project itself with `-e . --no-deps`. The same lock as on the laptop (Chapter 6), so CI tests the same versions.
4. The four checks of the README.

### The job `images`

```yaml
strategy:
  fail-fast: false
  matrix:
    image: [api, pipeline]
```

`${{ matrix.image }}` is `api` in one copy of the job and `pipeline` in the other. `fail-fast: false` means that when one copy fails, the other still finishes, so one run shows the state of both images.

Docker's own actions build `docker/${{ matrix.image }}.Dockerfile` with `load: true` (keep the image in the runner's Docker) and `push: false` (send it nowhere). Then Trivy scans it. Why without a layer cache is explained near the end of the chapter.

## Pinning what CI runs

A workflow runs code that other people wrote: the actions and the scanner. If one of them is changed by an attacker, your CI runs the attacker's code, with your token, on your code. An attack on the tools and libraries that a project uses, instead of on the project itself, is a **supply-chain attack**. This is not theory. In March 2025, the popular action `tj-actions/changed-files` was changed so that its release tags pointed at bad code, and many workflows that used it by tag printed their secrets into their public logs before it was found.

The defence is the idea of Chapter 32 again. A tag such as `v7.0.1` is a name, and the owner (or an attacker) can move it to other code. A **commit hash** is the checksum of one exact version of the code, and it cannot be moved. So every action is pinned by its full commit hash, with the readable version in a comment:

```yaml
- uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
```

This is **commit hash pinning**. The three actions are `actions/checkout` v7.0.1, `docker/setup-buildx-action` v4.4.1 and `docker/build-push-action` v7.4.0, each a release at least two weeks old: the project's rule for any new release (Chapter 38), because a bad release is usually found and pulled within days.

Trivy is not used through an action at all. It runs from its official image, `aquasec/trivy:0.75.0`, pinned by digest. A third-party action would be one more piece of someone else's code between the workflow and the scanner, and the image is the scanner itself.

The workflow file was then checked with **actionlint** 1.7.12, a linter for GitHub Actions workflows. It finds wrong keys, wrong expressions and typing mistakes, and it runs **shellcheck**, a linter for shell scripts, on every `run:` step. It reported no findings.

## Known vulnerabilities: the words first

A **vulnerability** is a mistake in software that an attacker can use. When one is found in a public library, it is published, so that every user can learn about it. Each published vulnerability gets an id.

- **CVE**, "Common Vulnerabilities and Exposures": the oldest and most common list. An id looks like `CVE-2026-114145`.
- **GHSA**, a "GitHub Security Advisory": GitHub's own list. An id looks like `GHSA-83vg-56qc-22m7`.
- **PYSEC**: the list of the Python Packaging Authority, for packages on PyPI. An id looks like `PYSEC-2026-4320`.

One vulnerability often has ids in several lists. The other ids are its **aliases**. Above, the CVE and the GHSA id are the same problem in gitpython.

A **vulnerability database** collects these records with the versions they affect. **OSV.dev** joins many of them in one place. pip-audit asks PyPI's vulnerability service, which uses the same records; Trivy downloads its own database, built from these sources. A record also has a **severity**: LOW, MEDIUM, HIGH or CRITICAL, a rough measure of how bad the worst case is. And it has a **fix version**: the first release without the problem, if one exists yet.

The project uses two scanners, because they look at different things.

- **pip-audit** reads a list of Python packages, here the three lock files: `requirements.lock`, `docker/requirements-api.lock` and `docker/requirements-pipeline.lock`.
- **Trivy** reads a finished image: the Debian packages of the operating system and the Python libraries installed in it.

## What the audit found

pip-audit on the three lock files found known vulnerabilities in 11 packages.

| Package | Version | Findings | The problem, and its reach here | Fix released |
|---|---|---|---|---|
| certifi | 2026.7.22 | 1 | trusts a root certificate whose intermediate CAs leaked their keys; used for HTTPS to the data sources | 20 Oct |
| uv | 0.13.0 | 1 | the same root in its own certificate bundle; a development tool | 20 Oct |
| gitpython | 3.2.0 | 1 | a submodule can be cloned into `.git/hooks`; MLflow and DVC only read the repository | 14 Oct |
| mlflow | 3.17.0 | 1 | path traversal in the FileStore backend; the project's servers use SQLite | 21 Oct |
| pillow | 12.3.0 | 37 | decoder bugs on crafted images; it only writes the SHAP plot | 16 Oct |
| filelock | 4.1.1 | 6 | races in lock files; needs an attacker who can already write files | 17 Oct |
| virtualenv | 21.14.7 | 7 | shell syntax in a path or prompt; a development tool | 17 Oct |
| setuptools | 84.0.0 | 1 | a crafted wheel on Windows; only pinned wheels from PyPI are installed | 19 Oct |
| pypdf | 6.20.1 | 7 | crafted PDFs that use much memory; only reads the PDF of this book | 21 Oct |
| nltk | 3.10.3 | 1 | paths outside its data folder; comes with Evidently, never loads a model | no fix |
| diskcache | 5.6.3 | 1 | unpickles its cache folder; DVC's cache inside the project | no fix |

*Figure 41.2 – The findings of 24 October 2026, from the statements in [`.trivyignore.yaml`](../../.trivyignore.yaml).*

Some words of the table. A **certificate authority** (CA) signs the certificates that prove a website is who it says; a *root* CA is one that your software trusts directly, and certifi is the list of them. **Path traversal** means a name such as `../../secret` that leads a program out of the folder it should stay in. A **crafted** file is one made on purpose to trigger a bug. A **race** is a bug that appears only when two things happen in a certain order. To **unpickle** is to load a Python pickle file, which can run any code.

The column "its reach here" is the important one. A vulnerability is a danger only if an attacker can reach it in *this* project. pillow has 37 findings, all in code that reads images. This project never opens an image it did not make: pillow comes with matplotlib, which only *writes* the SHAP plot (Chapter 28). Every row was judged in this way, one by one.

## A rule against a finding

Now look at the last column. Every fix was released between 13 and 21 October. On 24 October, none of them was two weeks old. The project's rule said: do not take them yet. The scanners said: you have known problems. Both are right, and the engineer had to choose.

One way out was to upgrade everything that day and break the rule. But the rule exists for a reason. A release can itself be the attack: a stolen password, a bad version on PyPI, found and pulled a few days later. A project that upgrades on the first day takes that risk eleven times at once, to fix problems that, row by row, it cannot be hit by.

So the rule was kept, and each finding was **accepted for a limited time**. An **accepted risk** is a known problem that a team decides, in writing, to live with, and gives a reason. When the decision has a fixed end, it is a **time-boxed exception**. Here is one entry of [`.trivyignore.yaml`](../../.trivyignore.yaml):

```yaml
- id: CVE-2026-53995
  statement: "Path traversal through a trace ID in the FileStore tracking backend. The project's MLflow servers use the SQLite backend, and none is reachable from outside the machine or the cluster. Fixed in 3.17.2 (21 Oct)."
  expired_at: 2026-11-04
```

Where does 4 November come from? The last fixes, mlflow 3.17.2 and pypdf 6.21.2, were released on 21 October. 21 October plus 14 days is 4 November. On that day every fix is old enough, and the upgrade is made as a separate change. The two findings with no fix, nltk and diskcache, are reviewed on 1 December. The file has 66 entries in all: the 64 findings of pip-audit, and two more pillow findings that only Trivy reported.

The important property is what happens when the day comes. The entry stops applying, the finding comes back, and CI turns red. Nobody has to remember. Nothing stays accepted by being forgotten. A test, `test_every_entry_of_the_project_file_has_a_reason_and_an_expiry_day` in [`test_audit_dependencies.py`](../../tests/unit/test_audit_dependencies.py), makes sure that no one can add an entry without a statement or without a day.

## One file, two scanners

Two scanners with two lists of exceptions would drift apart: someone accepts a finding in one and forgets the other. So there is one file. Trivy reads `.trivyignore.yaml` by itself, including `expired_at`. pip-audit has no such file, only the option `--ignore-vuln ID`. The small program [`tools/audit_dependencies.py`](../../tools/audit_dependencies.py) connects them. It reads the same file and passes on only the entries whose day has not come:

```python
def accepted_ids(ignore_file: Path, today: dt.date) -> list[str]:
    """The ids of ``ignore_file`` whose ``expired_at`` day has not been reached."""
    ...
        if expires is None:
            raise ValueError(f"{entry['id']} in {ignore_file.name} has no expired_at day")
        ...
        if today < expires:
            ids.append(entry["id"])
    return ids
```

`today < expires`: on the day itself, the entry no longer counts, the same as in Trivy. Counted on the real file:

| Day | Accepted ids |
|---|---|
| 24 October to 3 November | 66 |
| 4 November to 30 November | 2 |
| from 1 December | 0 |

*Figure 41.3 – How the accepted list shrinks by itself.*

Two details had to be checked, not assumed.

- **Does pip-audit match aliases?** The file mostly uses CVE ids, and pip-audit often reports a finding under its own id. The engineer ignored the gitpython finding once by its GHSA id, `GHSA-83vg-56qc-22m7`, and once by its CVE id, `CVE-2026-114145`. Both times the finding was ignored. So pip-audit matches aliases.
- **Do both tools know the id?** For certifi, Trivy reported `GHSA-6q48-6c7w-9gw2`, and pip-audit's own id was `PYSEC-2026-4320`. Trivy matches an entry by the id it reports itself, and pip-audit also matches aliases. So the file uses the GHSA id, which both tools find.

With the file in place, the three lock files printed:

```text
== requirements.lock
No known vulnerabilities found, 65 ignored
== docker\requirements-api.lock
No known vulnerabilities found, 40 ignored
== docker\requirements-pipeline.lock
No known vulnerabilities found, 47 ignored
```

The numbers differ because each lock has a different set of packages. pip-audit also prints a warning: "users are encouraged to fully hash their pinned dependencies". The lock files pin each version, but not the checksum of each file, so a different file under the same version number would not be noticed. That is a known limit, not solved in this push.

## The images, and a package that a cache kept old

Trivy 0.75.0 scanned both images for HIGH and CRITICAL findings, with `--ignore-unfixed`: a finding without a fix is left out, because nothing can be done about it but wait. Before the ignore file existed, the API image had 31 findings: 25 HIGH and 6 CRITICAL. All of them were in Python packages (gitpython, certifi, mlflow and pillow), none in a Debian package. With the ignore file, the API image scanned clean.

The pipeline image had Python findings too, and one more in a Debian package:

```text
libbrotli1   1.1.0-2+b7   fixed in 1.1.0-2+deb13u1
```

Debian had published a security update. But why was the old version in the image? The Dockerfile installs its Debian packages like this:

```dockerfile
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*
```

No version is pinned (Chapter 33). So a fresh build installs the newest packages of Debian 13, security updates included. The local build was not fresh: Docker reused a cached layer of this step from an earlier build (Chapter 32), with the packages of that day. A fresh build would get the fixed version.

This is why the job `images` uses no layer cache, on purpose. With a cache, the scan would report problems that a fresh build has already fixed, and, worse, the image would really keep them. Without a cache, each run builds what a release would build today. The cost is time: every run installs everything again.

## What is not done yet

- **The first run on GitHub.** Everything in this chapter ran on the local machine, in the same images. The runner may still surprise.
- **The upgrade of 4 November**, when the fixes are two weeks old. It is a separate change, and CI will demand it on that day.
- **The images are built and scanned, not published**, and they do not carry their git commit yet. So retraining in the cluster is still refused (Chapter 40).
- **Lock files without hashes**, as pip-audit's warning says.
- **actionlint and kubeconform run by hand**, not in CI.

## Try it yourself

Run the audit as CI runs it. It needs a network, because pip-audit asks a vulnerability database.

```powershell
.venv\Scripts\python.exe tools\audit_dependencies.py
```

After about half a minute you should see the three blocks shown in the section "One file, two scanners": `No known vulnerabilities found, 65 ignored`, then `40 ignored` and `47 ignored`, each after two lines of the warning about hashes. If a new vulnerability was published since this book was written, you will see it instead, and the exit code is 1. That is CI doing its job.

Now see the expiry at work, without waiting for November. Ask the tool's own function which ids it would pass on, on three days.

```powershell
.venv\Scripts\python.exe -c "import datetime as dt, importlib.util; s=importlib.util.spec_from_file_location('a','tools/audit_dependencies.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); [print(d, len(m.accepted_ids(m.IGNORE_FILE, dt.date.fromisoformat(d)))) for d in ['2026-11-03','2026-11-04','2026-12-01']]"
```

You should see `2026-11-03 66`, `2026-11-04 2` and `2026-12-01 0`: the numbers of Figure 41.3. Last, run the tests of the tool.

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/test_audit_dependencies.py -q
```

You should see `4 passed`.

## Key takeaways

- CI runs the same checks as the README on a clean machine for every push and pull request. A workflow has jobs, each job runs on a fresh runner, and a job is a list of steps.
- A flaky test is a bug that has not been found. Make it fail every time, find the cause, fix the cause, and show that the same condition now passes. Here: a 20-second start limit that a busy machine with few CPUs could not meet.
- Pin every action by its full commit hash, give the token only `contents: read`, and do not keep credentials after the checkout. Tags can move; hashes cannot.
- A finding matters when an attacker can reach it in your project. Each of the 11 packages was judged on that.
- The two-week rule was kept. Each finding was accepted until a fixed day, with a reason, in one file that both scanners read. When the day comes, CI turns red by itself.
- Build images for a scan without a layer cache, or the scan reports the past.

## Exercises

**1.** A test fails in CI about once in fifty runs. A colleague installs a plugin that runs every failed test again up to three times. CI is green again. What is wrong with this, and what would you do instead?

<details>
<summary>Answer</summary>

The retry hides the failure; it does not explain it. The cause could be in the test, or a real race in the code that users will also meet. After the retry, nobody sees it any more, and a real bug that fails sometimes is now hidden too. Instead, find what was different in the failed run (load, order of tests, time, network), build a condition that makes the test fail every time, as the engineer did with 4 CPUs and 4 busy loops, find the cause, fix it, and show that the same condition now passes.

</details>

**2.** On 2 November a new finding appears in pyarrow, and CI turns red. A fix was released on 1 November. You read the record, and the problem is in a part of pyarrow that this project never calls. What goes into `.trivyignore.yaml`, and what is its `expired_at` day?

<details>
<summary>Answer</summary>

An entry with the finding's id (one that both scanners know, such as the CVE or GHSA id), a statement that says why the project cannot be hit ("the problem is in X; the project only uses pyarrow to read and write its own Parquet files"), and the fixed version with its date. The day is 1 November plus 14 days: `expired_at: 2026-11-15`. If the problem *could* reach the project, accepting it would be wrong; then the upgrade is made at once, as a written exception to the two-week rule.

</details>

**3.** A colleague wants to add a layer cache to the job `images`, "to make CI faster". What would the scan report then, and what would the images contain?

<details>
<summary>Answer</summary>

The step with `apt-get install` would be reused from an older build, with the Debian packages of that day. The scan would report problems, such as the old `libbrotli1`, that a fresh build has already fixed, so CI would be red for no reason. And if a scan is ignored because "a fresh build fixes it", the images that are built with the cache really do keep the old packages. The time saved is paid for with a scan of the past.

</details>

**4. Interview question.** "Your dependency scanner reports a critical vulnerability in a library you use. What do you do?"

<details>
<summary>Points of a good answer</summary>

- Read the record: which versions, which part of the library, and is there a fix version.
- Decide if it can reach you: do you call that code, with input an attacker controls? Use the "via" chain of the lock file to see why the library is there.
- If it can reach you, upgrade now, and say so if that breaks a rule such as a minimum age for releases.
- If it cannot, accept it in writing, with the reason and an expiry day, in a file that every scanner reads, so that the check fails again by itself.
- Make the upgrade a separate, tested change, and pin actions and tools so that the fix itself cannot become an attack.

</details>

---

[Previous: Chapter 40 – Running it on kind: what happened](40-running-it-on-kind.md) | [Contents](README.md) | [Next: Appendix A – Stack map](appendix-a-stack-map.md)

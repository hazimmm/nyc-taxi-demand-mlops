# Architecture

The system produces a **day-ahead forecast of hourly taxi pickups per NYC zone**. This page shows every tool in the stack and how data moves from ingestion to production.

## High-level flow

```mermaid
flowchart TB
    subgraph SRC["Sources"]
        TLC["NYC TLC<br/>yellow-taxi trip Parquet"]
        OM["Open-Meteo<br/>hourly weather"]
    end

    subgraph DATA["Data pipeline · DuckDB + pandas · versioned by DVC"]
        direction LR
        RAW[("Raw / bronze<br/>as downloaded + manifest")]
        SILVER[("Silver<br/>cleaned trips + weather")]
        GOLD[("Gold<br/>hourly demand per zone<br/>+ feature table")]
        RAW -->|"clean"| G1{{"Pandera gate"}} --> SILVER
        SILVER -->|"aggregate + features"| G2{{"Pandera gate"}} --> GOLD
    end

    subgraph TRAIN["Training"]
        direction LR
        BASE["Seasonal-naive<br/>baseline"]
        LGBM["LightGBM<br/>+ Optuna tuning"]
        BT["Rolling-origin<br/>backtest"]
        SHAP["SHAP<br/>explainability"]
        BASE --> BT
        LGBM --> BT --> SHAP
    end

    subgraph REG["MLflow · experiment tracking + model registry"]
        direction LR
        CHAL["new version"]
        GATE{{"Promotion gate<br/>beats champion and baseline"}}
        CHAMP["champion"]
        REJ["challenger"]
        CHAL --> GATE
        GATE -->|passes| CHAMP
        GATE -->|fails| REJ
    end

    subgraph PROD["Production · Docker images on Kubernetes (kind) · Compose in development"]
        direction LR
        API["FastAPI /predict<br/>stateless · HPA-scaled"]
        BATCH["Batch forecast job<br/>day-ahead · simulated clock"]
        PRED[("Predictions<br/>forecasts + actuals")]
        BATCH --> PRED
    end

    subgraph MON["Monitoring"]
        direction LR
        EVI["Evidently<br/>drift + accuracy"]
        PROM["Prometheus"]
        GRAF["Grafana dashboards"]
        EVI --> PROM --> GRAF
    end

    RETRAIN["Retrain flow<br/>cooldown · rollback"]
    USER(["API client"])

    TLC --> RAW
    OM --> RAW
    GOLD --> TRAIN
    TRAIN -->|"params · metrics · artefacts"| CHAL
    CHAMP -->|"load model"| API
    CHAMP -->|"load model"| BATCH
    GOLD -->|"features"| API
    GOLD -->|"features"| BATCH
    USER -->|"request"| API
    API -->|"request metrics"| PROM
    PRED --> EVI
    GOLD -->|"reference data"| EVI
    EVI -->|"threshold breached"| RETRAIN
    RETRAIN --> TRAIN
```

## Cross-cutting layers

```mermaid
flowchart LR
    subgraph ORCH["Orchestration · Prefect"]
        direction LR
        I["ingest"] --> S["silver"] --> G["gold"] --> T["train"] --> F["forecast"] --> M["monitor"] --> R["retrain"]
    end

    subgraph CICD["Delivery · GitHub Actions"]
        direction LR
        Q["ruff · mypy · pytest"] --> SEC["pip-audit · Trivy"] --> B["build images"] --> P["push to GHCR"] --> K["kind smoke test"]
    end

    subgraph DEMO["Public demo"]
        HF["Streamlit on Hugging Face Spaces<br/>bundled champion model + gold sample"]
    end
```

## Stack by concern

| Concern | Tool | Where it appears |
|---|---|---|
| Data source | NYC TLC trip records, Open-Meteo | Sources |
| Processing | DuckDB, pandas | Data pipeline |
| Data validation | Pandera | Gates after silver and gold |
| Data versioning | DVC | Raw, silver and gold layers |
| Modelling | LightGBM, Optuna, SHAP | Training |
| Tracking and registry | MLflow | Registry, promotion gate |
| Orchestration | Prefect | Schedules every stage |
| Serving | FastAPI, Pydantic | Production |
| Monitoring | Evidently, Prometheus, Grafana | Monitoring |
| Packaging | Docker, Docker Compose | Production (development) |
| Deployment | Kubernetes (kind), Kustomize | Production |
| CI/CD | GitHub Actions, GHCR | Delivery |
| Demo | Streamlit, Hugging Face Spaces | Public demo |

## Design decisions

| Decision | Reason |
|---|---|
| **Day-ahead forecast, 48-hour horizon.** Every pickup feature is at least 48 h old. | The forecast for the whole of tomorrow is issued at 00:00 today. Its last hour, 23:00, is then 47 hours away, so nothing newer than 48 hours before each target hour may be used. |
| **Simulated clock.** Train on January to April 2025, replay May and June day by day as "production". | TLC publishes each month about two months late, so there is no live feed to monitor. |
| **Weather forecasts as features**, not observed weather. | Observed weather for a future hour is information the model would not have. |
| **One feature function** shared by training, batch forecasting and the API. | Removes training/serving skew. |
| **Zones chosen from the training period only**, then frozen in the config. | Picking them from the full dataset would leak the test period. |
| **Stateless API.** Forecasts and actuals are written by the batch job. | The API can scale to several replicas without file-write conflicts. |
| **Two monitors:** drift at once, accuracy when actuals arrive. | Ground truth is always late. |
| **Gated retraining** with cooldown and rollback. | An automatically retrained model must not replace a better one. |
| **DVC for the pipeline definition, Prefect for scheduling**, both calling the same functions. | Reproducibility in development, automation in production, no duplicated logic. |

## Contracts

- **Configuration:** `configs/config.yaml`, loaded by `src/demand/config.py`. Environment variables prefixed `DEMAND_` override any value.
- **Data:** column names and Pandera schemas for every layer live in `src/demand/schemas.py`.
- **Entry points:** `python -m demand <command>`, defined in `src/demand/cli.py`.

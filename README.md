# Finemed PharmaAI — Demand Intelligence Platform

![Tests](https://github.com/GradientDescent-git/Finemed_PharamaAI/actions/workflows/test.yml/badge.svg)

Finemed PharmaAI is an enterprise-grade demand intelligence, ML forecasting, and conversational analytics platform designed for pharmaceutical distribution and inventory planning.


---

## 1. Product & Architecture Overview

Finemed PharmaAI transforms historical ERP database exports (`.DAT` files) into deterministic demand series, applies validation-backed machine learning forecasting models (TSB + Chronos-2 P50), publishes immutable versioned production artifacts, and provides both interactive UI dashboards and grounded conversational AI query interfaces.

```text
                        ┌───────────────────────────┐
                        │       Finemed UI          │
                        │                           │
                        │ Assistant                 │
                        │ Overview                  │
                        │ Forecasts                 │
                        │ Rankings                  │
                        │ Data Pipeline             │
                        │ Operations                │
                        │ Settings                  │
                        └────────────┬──────────────┘
                                     │
                                     │ HTTPS / API
                                     ▼
                        ┌───────────────────────────┐
                        │        FastAPI API        │
                        │                           │
                        │ Auth                      │
                        │ Request validation        │
                        │ Chat API                  │
                        │ Forecast API              │
                        │ Ranking API               │
                        │ Pipeline API              │
                        │ Health / readiness        │
                        └────────────┬──────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
     ┌────────────────┐    ┌────────────────────┐  ┌────────────────┐
     │ Conversation   │    │ Forecast           │  │ Pipeline/Admin │
     │ Orchestrator   │    │ Intelligence       │  │ Controller     │
     │                │    │                    │  │                │
     │ Intent Router  │    │ Medicine Resolver  │  │ Upload         │
     │ LLM Router     │    │ Query Service      │  │ Validation     │
     │ Context        │    │ Repository         │  │ Pipeline       │
     └────────┬───────┘    └─────────┬──────────┘  └───────┬────────┘
              │                      │                     │
              ▼                      ▼                     ▼
     ┌────────────────┐    ┌────────────────────┐  ┌────────────────┐
     │ Claude / LLM   │    │ Production         │  │ Monthly        │
     │                │    │ Forecast Artifacts │  │ Data Pipeline  │
     │ Interpretation │    │                    │  │                │
     │ ONLY           │    │ latest.parquet     │  │ Raw → Bronze   │
     └────────────────┘    │ routing tables     │  │ → Silver       │
                           │ metadata           │  │ → Gold         │
                           └────────────────────┘  └───────┬────────┘
                                                           │
                                                           ▼
                                                   ┌────────────────┐
                                                   │ Forecasting    │
                                                   │ Models         │
                                                   │                │
                                                   │ TSB            │
                                                   │ Chronos-2 P50  │
                                                   │ Routing        │
                                                   └────────────────┘
```

---

## 2. Core Principles & Governance

1. **Validated Forecasting Methodology**:
   - Routing uses validation evidence only; holdout metrics never influence production routing.
   - Chronos-2 validation advantage $\ge 30\%$ routes to Chronos-2 P50. Otherwise route to TSB.
   - Active and Stale medicines enter model forecasting. Dormant medicines receive deterministic zero-demand records.
2. **Deterministic Intelligence**:
   - The deterministic query service remains 100% authoritative for numbers, rankings, and model decisions.
   - LLM layers format natural-language text and parse user intent. The LLM never invents business metrics or bypasses deterministic query services.
3. **Immutable & Atomic Publication**:
   - Candidate artifacts are written to versioned directories (`production_forecasts/<run_id>/forecast.parquet`).
   - `latest.parquet` is promoted only after complete validation succeeds. Failed runs never overwrite active forecasts.

---

## 3. Directory Layout

```text
Finemed_PharmaAI/
├── data/                       # Medallion data directories
│   ├── 01_raw/                 # Staged monthly DAT exports (INVOICE.DAT, MEDIMAST.DAT, etc.)
│   ├── 02_bronze/              # Raw converted tables
│   ├── 03_silver/              # Cleaned warehouse & aggregated series
│   ├── 04_silver/              # Daily & monthly demand series
│   └── 05_gold/                # Production forecast artifacts & latest.parquet
├── docs/                       # Architecture documentation
├── logs/                       # Application & execution logs
├── scripts/                    # CLI execution entrypoints
├── src/
│   └── finemed_ai/             # Main Python package
│       ├── api/                # FastAPI application, static frontend & schemas
│       ├── automation/         # Monthly pipeline runners & run status locking
│       ├── config/             # Environment and directory paths
│       ├── demand_forecasting/ # TSB, Chronos-2, routing logic & ForecastStore
│       ├── forecast_intelligence/ # Medicine resolver, query service & LLM orchestrator
│       ├── validation/         # File, schema, duplicate & data quality validators
│       └── warehouse/          # Dimensional modeling & facts
├── tests/                      # Pytest unit, integration, and security tests
├── Dockerfile                  # Production container definition
├── docker-compose.yml          # Container deployment compose file
├── pyproject.toml              # Project dependencies & build metadata
└── README.md                   # System documentation & runbooks
```

---

## 4. API Endpoints

### Health & Readiness
- `GET /health`: Liveness & dependency status (`status`, `forecast_store_loaded`, `chat_available`).
- `GET /ready`: Readiness probe for load balancers (returns `200` when store is loaded).
- `GET /version`: System version and forecasting model details.

### Conversational Intelligence
- `POST /chat`: Grounded natural-language query endpoint (`X-API-Key` required). Parses employee intent and returns deterministic forecast answers.

### Forecast & Ranking Services
- `GET /forecast/{medicine_id}`: Detailed 30-day forecast for a specific medicine.
- `GET /forecast/{medicine_id}/summary`: Aggregated summary and selected model info.
- `GET /forecast/top?n=10`: Top $N$ medicines by predicted demand.
- `GET /forecast/trend?direction=increasing&n=10`: Demand trend filtering.
- `GET /forecast/uncertain?n=10`: Uncertainty and prediction interval analysis.

### Monthly Pipeline & Operations (`X-Admin-Token` required)
- `POST /pipeline/upload`: Upload monthly DAT ZIP archive and stage package.
- `POST /pipeline/validate`: Validate staged DAT files without triggering pipeline execution.
- `POST /pipeline/run`: Trigger asynchronous monthly forecasting pipeline.
- `GET /pipeline/status/{run_id}`: Fetch status and stage breakdown for a run ID.
- `GET /pipeline/latest`: Return status and manifest of the latest pipeline execution.
- `GET /operations/summary`: Aggregate API status, store freshness, total medicines, and execution lock.
- `POST /refresh`: Reload in-memory `ForecastStore` after manual artifact updates.

---

## 5. Web Interface (Dark Finemed UI)

Accessible via `http://localhost:8000/`:

1. **Assistant**: Conversational interface for natural-language questions ("What is the forecast for Otacare?", "Which model predicted medicine 0001?", "Why was that model selected?").
2. **Overview**: Live dashboard summarizing active medicines, forecast horizon, daily average, and model distribution.
3. **Forecasts**: Deep-dive medicine lookup with daily predictions, quantiles (P10, P50, P90), and SVG demand charts.
4. **Demand Rankings**: Ranked lists of highest and lowest demand medicines (with medicine names and codes).
5. **Data Pipeline**: 5-step operational interface for package upload, validation, pipeline progress tracking, and store reload.
6. **Operations**: System health monitor displaying API status, store freshness (`HEALTHY`, `STALE`), and active container state.
7. **Settings**: Configuration panel for `CLIENT_API_KEY`, `ADMIN_TOKEN`, session management, and API connection testing.

---

## 6. Operator Monthly Update Runbook

Follow these steps to process new monthly ERP data:

1. **Access Data Pipeline Page**: Open the Finemed UI and navigate to **Data Pipeline** (or use administrative API endpoints).
2. **Upload Source Package**:
   - Provide the month identifier (e.g. `2025-07`).
   - Select the monthly `.zip` package containing all 8 DAT files (`INVOICE.DAT`, `INVDET.DAT`, `MEDIMAST.DAT`, `PURCHASE.DAT`, `COMPUR.DAT`, `SUPMAST.DAT`, `SFILE.DAT`, `TFILE.DAT`).
   - Click **Upload & Start Pipeline**.
3. **Validation & Pipeline Execution**:
   - The package is unpacked into `data/01_raw/<month>/` securely.
   - File presence, encoding, duplicates, and schemas are audited.
   - Bronze and Silver demand series are rebuilt.
   - TSB and Chronos-2 model predictions and validation-backed routing policies execute.
4. **Atomic Publication**:
   - Candidate forecasts are verified in `data/05_gold/demand_forecasting/production_forecasts/<run_id>/`.
   - On full validation success, `latest.parquet` is atomically updated.
5. **Application Refresh**:
   - The UI automatically reloads the in-memory `ForecastStore` and invalidates stale conversation sessions.

---

## 7. Local Development & Testing

### Virtual Environment Setup
```powershell
# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\Activate.ps1

# Run test suite
.venv\Scripts\python.exe -m pytest -q
```

---

## 8. Docker Deployment

### Build & Run Container
```bash
# Build image
docker-compose build

# Start production container
docker-compose up -d

# Verify container health
docker-compose ps
```

Container endpoints default to `http://localhost:8000`.

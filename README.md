# FinemedAI

### Production-Oriented Pharmaceutical Demand Forecasting & Intelligence Platform

> An end-to-end machine learning system built on real pharmaceutical operational data to transform historical pharmaceutical transactions into medicine-level demand forecasts using rigorous time-series validation, model comparison, and production-oriented engineering.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue)
![Docker](https://img.shields.io/badge/Docker-Containerization-blue)
![Pytest](https://img.shields.io/badge/Pytest-Testing-yellow)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-success)

</p>

<p align="center">

<b>79 Medicines</b> • <b>25.58% WAPE</b> • <b>Rolling Backtesting</b> • <b>TSB vs Chronos-2</b> • <b>FastAPI</b> • <b>PostgreSQL</b> • <b>Docker</b>

</p>

---

# 📌 Overview

FinemedAI is a pharmaceutical demand forecasting system designed to transform multi-year operational data into validated medicine-level demand forecasts.

The project focuses on the complete machine learning workflow:

```text
Raw Pharmaceutical Data
        ↓
Data Extraction
        ↓
Cleaning & Transformation
        ↓
Schema & Data Quality Validation
        ↓
Calendar Continuity
        ↓
Medicine-Level Demand Construction
        ↓
Time-Series Backtesting
        ↓
Model Evaluation
        ↓
TSB / Chronos-2 / Hybrid Experiments
        ↓
Evidence-Based Model Selection
        ↓
Forecast Service
        ↓
API / Production Layer
```

Rather than treating forecasting as a single notebook experiment, FinemedAI focuses on reproducible data preparation, leakage-aware evaluation, model diagnostics, forecasting experiments, validation, and production-oriented software architecture.

---

# 🎯 The Business Problem

Pharmaceutical demand forecasting is not a standard regression problem.

Medicine demand can be:

* Intermittent
* Sparse
* Highly variable
* Different across products
* Sensitive to changes in demand behaviour
* Vulnerable to the consequences of systematic underforecasting

Poor forecasts can negatively affect downstream planning. In pharmaceutical operations, persistent underforecasting can increase the risk of stock shortages, while excessive forecasting can contribute to inefficient planning.

The objective of FinemedAI is to create a reproducible forecasting system capable of:

1. Processing historical pharmaceutical operational data.
2. Constructing reliable medicine-level demand time series.
3. Evaluating multiple forecasting approaches.
4. Preventing future-data leakage during evaluation.
5. Comparing models using business-relevant forecasting metrics.
6. Selecting forecasting strategies based on empirical evidence.
7. Serving forecasts through a production-oriented application layer.

---

# 🏆 Key Results

The core forecasting experiments evaluated **79 medicines** using rolling validation and a separate locked holdout evaluation.

| Metric                         |                  Result |
| ------------------------------ | ----------------------: |
| Medicines evaluated            |                  **79** |
| Best holdout WAPE              |             **25.577%** |
| TSB holdout WAPE               |             **25.577%** |
| Hybrid holdout WAPE            |             **26.013%** |
| Chronos-2 P50 holdout WAPE     |             **27.724%** |
| Validation methodology         | **Rolling Backtesting** |
| Classical forecasting approach |                 **TSB** |
| Foundation model evaluated     |    **Amazon Chronos-2** |

## Key Finding

> A more sophisticated model does not automatically produce a better forecasting system.

On the final locked holdout evaluation:

```text
TSB               → 25.577% WAPE
Hybrid            → 26.013% WAPE
Chronos-2 P50     → 27.724% WAPE
```

TSB achieved the strongest overall holdout performance.

The production decision was therefore driven by empirical evaluation rather than model novelty.

---

# 🧠 The Core ML Story

A central question behind FinemedAI was:

> Can a modern time-series foundation model outperform a classical intermittent-demand forecasting approach on real pharmaceutical medicine demand?

To investigate this, multiple forecasting approaches were evaluated using chronological, leakage-aware backtesting.

```text
                     Historical Demand
                            │
                            ▼
                  Rolling Backtesting
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
        Classical Forecasting       Foundation Model
               TSB                    Chronos-2
              │                           │
              └─────────────┬─────────────┘
                            │
                            ▼
                    Model Evaluation
                            │
                            ▼
                     Error Analysis
                            │
                            ▼
                    Locked Holdout
                            │
                            ▼
                  Production Decision
```

The objective was not to prove that one type of model was universally superior.

The objective was to determine which approach performed best for the actual forecasting problem.

---

# 🔍 A Critical Diagnostic Finding

During experimentation, Chronos-2 showed systematic underforecasting on the pharmaceutical demand data.

Instead of automatically deploying the foundation model, the project investigated the behaviour through additional experiments and diagnostics.

The experimentation workflow included:

* Rolling backtesting
* Context-length experiments
* Forecast error analysis
* Probabilistic forecasting
* Calibration experiments
* Bias analysis
* Model robustness analysis
* Model selection experiments
* Routing experiments
* Hybrid and ensemble experiments
* Locked holdout evaluation

The final comparison showed:

| Model         | Holdout WAPE |
| ------------- | -----------: |
| 🥇 **TSB**    |  **25.577%** |
| Hybrid        |      26.013% |
| Chronos-2 P50 |      27.724% |

> The model with the strongest empirical performance was preferred over the model with the highest architectural complexity.

---

# 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                  PHARMACEUTICAL OPERATIONAL DATA             │
│                                                              │
│                 Multi-Year Historical Records                │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                       DATA PIPELINE                          │
│                                                              │
│  Extraction → Cleaning → Transformation → Validation         │
│                                                              │
│  • Schema Validation                                         │
│  • Data Quality Checks                                       │
│  • Calendar Continuity                                       │
│  • Demand Construction                                       │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                 MEDICINE-LEVEL TIME SERIES                   │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                    FORECASTING ENGINE                        │
│                                                              │
│       ┌─────────────┐              ┌─────────────┐           │
│       │     TSB     │              │  Chronos-2  │           │
│       └──────┬──────┘              └──────┬──────┘           │
│              │                            │                  │
│              └─────────────┬──────────────┘                  │
│                            ▼                                 │
│                   Model Evaluation                           │
│                            │                                 │
│                 WAPE + Diagnostics                           │
│                            │                                 │
│                            ▼                                 │
│                    Model Selection                           │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                  FORECAST SERVICE LAYER                      │
│                                                              │
│        Forecast API • Validation • Metadata                  │
│                                                              │
│        Persistence • Forecast Artifacts • Automation         │
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
                        APPLICATION / CLIENT
```

---

# ⚙️ Data Engineering Pipeline

The forecasting workflow begins with historical pharmaceutical operational records.

```text
Raw Operational Files
        │
        ▼
Data Extraction
        │
        ▼
Schema Validation
        │
        ▼
Cleaning & Transformation
        │
        ▼
Data Quality Checks
        │
        ▼
Calendar Continuity
        │
        ▼
Demand Construction
        │
        ▼
Medicine-Level Time Series
```

The pipeline is responsible for transforming operational data into forecasting-ready demand series.

Key responsibilities include:

* Data extraction
* Data transformation
* Schema validation
* Data quality checks
* Missing-value handling
* Calendar continuity checks
* Demand aggregation
* Medicine-level time-series construction
* Forecast artifact validation

---

# 📈 Evaluation Methodology

## Why Not Use Random Train-Test Splits?

Time-series forecasting requires preserving chronological order.

A random train-test split can introduce future information into the training process and produce misleading evaluation results.

FinemedAI therefore uses chronological forecasting evaluation.

```text
Historical Data

│
├──────────── Train ────────────► Forecast ───► Evaluate
│
├────────────────── Train ──────► Forecast ───► Evaluate
│
├──────────────────────── Train ─► Forecast ───► Evaluate
│
└─────────────────────────────── Locked Holdout
                                      │
                                      ▼
                                Final Evaluation
```

The evaluation design aims to:

* Preserve temporal ordering
* Reduce future-data leakage
* Evaluate models across multiple historical periods
* Separate validation from final holdout testing
* Support reproducible model comparison

---

# 🤖 Forecasting Approaches

## TSB

TSB was evaluated as a classical intermittent-demand forecasting approach.

This is particularly relevant for medicine demand where time series can include intermittent or zero-demand periods.

TSB ultimately achieved the strongest overall performance on the final holdout evaluation.

```text
TSB Holdout WAPE → 25.577%
```

---

## Amazon Chronos-2

Amazon Chronos-2 was evaluated as a modern time-series foundation model.

The experimentation explored:

* Context length
* Point forecasting
* Probabilistic forecasting
* Quantile predictions
* Calibration
* Forecast bias
* Model robustness

Chronos-2 provided a strong modern benchmark and probabilistic forecasting capabilities.

However, empirical evaluation revealed systematic underforecasting on this dataset and weaker overall holdout performance compared with TSB.

```text
Chronos-2 P50 Holdout WAPE → 27.724%
```

---

## Hybrid Forecasting

Hybrid and ensemble approaches were also evaluated to determine whether combining forecasts could improve performance.

```text
TSB               → 25.577% WAPE
Hybrid            → 26.013% WAPE
Chronos-2 P50     → 27.724% WAPE
```

The hybrid approach improved relative to Chronos-2 but did not outperform TSB on the final holdout.

---

# 🧪 Experimentation Framework

The project uses a structured experimentation workflow rather than relying on a single model run.

```text
Demand Data
    │
    ├── Classical Backtesting
    │
    ├── Chronos-2 Backtesting
    │
    ├── Context Optimization
    │
    ├── Calibration Experiments
    │
    ├── Model Robustness Analysis
    │
    ├── Model Selection
    │
    ├── Routing Experiments
    │
    ├── Routing Threshold Optimization
    │
    └── Adaptive Ensemble Experiments
            │
            ▼
      Model Comparison
            │
            ▼
      Locked Holdout
            │
            ▼
      Production Decision
```

This experimentation process allows the forecasting strategy to be selected using observed performance rather than assumptions.

---

# 🧭 Model Selection and Routing

Medicine demand characteristics can differ significantly across products.

A single forecasting approach may not always be optimal for every time series.

FinemedAI therefore explores model selection and routing strategies based on:

* Historical forecasting performance
* Rolling backtest results
* Forecast error
* Systematic bias
* Demand characteristics
* Model robustness

This architecture supports the idea that forecasting strategies should be evaluated at the appropriate level of granularity rather than assuming one model will always be optimal.

---

# 🚀 Production-Oriented Architecture

FinemedAI extends beyond model experimentation by structuring forecasting functionality into reusable services.

The production-oriented layer includes components for:

* Forecast generation
* Forecast routing
* Forecast services
* API integration
* Data validation
* Forecast artifact handling
* Pipeline automation
* Testing

The exact deployment configuration depends on the environment and should be verified against the repository implementation.

---

# 🔌 API

The project includes an API layer for exposing forecasting functionality.

A typical interaction flow is:

```text
Client
   │
   ▼
Forecast Request
   │
   ▼
API Layer
   │
   ▼
Forecast Service
   │
   ▼
Model Selection / Routing
   │
   ▼
Forecast Generation
   │
   ▼
Validation
   │
   ▼
Response
```

> Refer to the API source and interactive documentation for the exact endpoints and request schemas implemented in the repository.

---

# 🗂️ Project Structure

The repository is organized around modular data, forecasting, validation, API, and automation components.

```text
Finemed_PharmaAI/
│
├── src/
│   └── finemed_ai/
│       │
│       ├── api/
│       │
│       ├── automation/
│       │
│       ├── demand_forecasting/
│       │   ├── backtest.py
│       │   ├── evaluation.py
│       │   ├── model_selection.py
│       │   ├── production_forecast_router.py
│       │   ├── production_forecast_service.py
│       │   └── ...
│       │
│       ├── pipeline/
│       ├── validation/
│       └── warehouse/
│
├── config/
├── data/
├── docs/
├── notebooks/
├── scripts/
├── tests/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# 🧪 Testing

The project includes automated tests for core functionality.

Run the test suite:

```bash
pytest -q
```

A previously verified project state included:

```text
37 passed
```

Testing covers relevant functionality including:

* Forecasting logic
* Data validation
* Forecast artifact validation
* Pipeline behaviour
* Model selection logic

The current test count should always be verified before presenting a release or portfolio version.

---

# 💻 Local Development

## 1. Clone the Repository

```bash
git clone https://github.com/GradientDescent-git/Finemed_PharamaAI.git
cd Finemed_PharamaAI
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create and configure the environment file based on the repository configuration.

If an example environment file is available:

```bash
cp .env.example .env
```

Do not commit secrets, passwords, or production credentials.

---

## 5. Run the Application

Run the application using the project's configured API entry point.

For example:

```bash
uvicorn src.finemed_ai.api.main:app --reload
```

If your package structure or API entry point differs, use the command defined by the repository.

---

# 🐳 Containerization

If Docker support is configured in the repository, the application can be run using the provided Docker configuration.

Typical workflow:

```bash
docker build -t finemed-ai .
docker run -p 8000:8000 finemed-ai
```

If Docker Compose is configured:

```bash
docker compose up --build
```

Always verify the exact deployment command against the repository configuration.

---

# 🔬 Key Technical Decisions

## Why TSB?

TSB was retained as the strongest forecasting approach based on empirical evaluation.

```text
TSB WAPE            → 25.577%
Chronos-2 P50 WAPE  → 27.724%
```

The model decision was based on observed performance rather than model complexity.

---

## Why Evaluate a Foundation Model?

Time-series foundation models offer promising generalization capabilities and probabilistic forecasting.

Chronos-2 was therefore evaluated as a serious forecasting candidate rather than being adopted without validation.

The experiments showed that performance must be measured against the specific demand characteristics of the problem.

---

## Why Rolling Backtesting?

Rolling backtesting simulates repeated historical forecasting scenarios while preserving chronological order.

This provides a more realistic evaluation framework than randomly splitting a time series.

---

## Why Separate Validation and Holdout?

Model development and model selection should not repeatedly optimize against the final evaluation period.

A separate holdout provides a more reliable estimate of how the selected approach performs on unseen future data.

---

## Why Model Routing?

Different medicines can exhibit different demand characteristics.

The project therefore explores whether model performance can vary across medicine-level series and whether routing or selection strategies can improve forecasting decisions.

---

# 📉 What Did Not Work

A key strength of the project is that failed or weaker experiments were retained as part of the engineering and research process.

## Chronos-2 Systematic Underforecasting

Chronos-2 initially appeared promising as a foundation forecasting approach.

However, diagnostics identified systematic underforecasting behaviour.

This led to further experiments involving:

* Calibration
* Context optimization
* Forecast bias analysis
* Model robustness analysis
* Routing strategies
* Threshold optimization
* Hybrid and ensemble approaches

The final decision remained evidence-based.

> A newer or more complex model should not automatically replace a simpler model unless evaluation demonstrates a meaningful improvement.

---

# 📊 Data and Forecast Validation

Reliable forecasting depends on more than model accuracy.

The project includes validation-oriented components for:

* Input data quality
* Schema consistency
* Calendar continuity
* Time-series construction
* Forecast artifacts
* Pipeline outputs

The objective is to reduce the risk of generating forecasts from invalid or inconsistent data.

---

# 🔮 Future Roadmap

Potential future improvements include:

* [ ] Automated retraining
* [ ] Forecast drift detection
* [ ] Expanded uncertainty calibration
* [ ] Cloud deployment
* [ ] Forecast monitoring dashboard
* [ ] Automated data ingestion
* [ ] Model registry
* [ ] Expanded model routing
* [ ] Natural-language analytics interface
* [ ] LLM tool-calling layer for structured forecasting queries

These items are future improvements and are not represented as currently implemented functionality.

---

# 📊 Business Value

FinemedAI converts historical pharmaceutical operational data into evaluated medicine-level demand forecasts.

The system is designed to support demand planning through:

* Reproducible data pipelines
* Automated validation
* Leakage-aware evaluation
* Empirical model comparison
* Evidence-based model selection
* Production-oriented forecasting services

No unsupported financial savings or business impact claims are made.

---

# 🛠️ Technology Stack

## Machine Learning and Forecasting

* Python
* Time-Series Forecasting
* Intermittent Demand Forecasting
* TSB
* Amazon Chronos-2
* Rolling Backtesting
* Model Selection
* Model Routing
* Probabilistic Forecasting

## Data Engineering

* Pandas
* NumPy
* ETL Pipelines
* Data Validation
* Data Quality Checks
* Time-Series Construction

## Application Layer

* FastAPI
* REST APIs
* PostgreSQL

## Engineering and MLOps

* Docker
* Pytest
* GitHub Actions
* CI/CD
* Automated Testing
* Forecast Artifact Validation

---

# 📚 Key Learnings

This project reinforced several machine learning engineering principles:

1. **Complex models do not automatically outperform simpler models.**
2. **Time-series evaluation must preserve chronological ordering.**
3. **Future-data leakage can invalidate forecasting results.**
4. **A strong baseline is essential when evaluating advanced models.**
5. **Error analysis is as important as headline metrics.**
6. **Model selection should be driven by empirical evidence.**
7. **Production ML requires data validation, testing, reproducibility, and software engineering in addition to model development.**

---

# 👤 Author

**Vigneshwaran M**

Aspiring Machine Learning Engineer focused on production-oriented machine learning systems, time-series forecasting, and applied AI.

* GitHub: https://github.com/GradientDescent-git
* LinkedIn: https://www.linkedin.com/in/vigneshwaran-m-a12b4635a/

---

# ⭐ Project Status

FinemedAI has completed its core demand forecasting experimentation and validation work.

Current engineering focus includes:

* Production hardening
* Forecast service refinement
* API and deployment improvements
* Operational reliability
* Monitoring and observability
* Continued model evaluation

---

# Final Perspective

FinemedAI is designed as more than a notebook-based machine learning experiment.

The project follows the complete path:

```text
Raw Pharmaceutical Operational Data
        ↓
Validated Data Pipeline
        ↓
Medicine-Level Demand Construction
        ↓
Leakage-Aware Backtesting
        ↓
Classical and Foundation Model Evaluation
        ↓
Error Analysis
        ↓
Locked Holdout Evaluation
        ↓
Evidence-Based Model Selection
        ↓
Forecast Services
        ↓
Production-Oriented Engineering
```

> **Do not deploy a model because it is the most sophisticated. Deploy the model that performs reliably under rigorous evaluation for the problem being solved.**

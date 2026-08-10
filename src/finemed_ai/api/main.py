from __future__ import annotations
 
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
 
from contextlib import asynccontextmanager
 
import shutil
import zipfile
import uuid

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from finemed_ai.automation.run_status import RunStatus
 
from finemed_ai.demand_forecasting.store import ForecastNotFoundError, ForecastStore
from finemed_ai.llm.orchestrator import Orchestrator
from finemed_ai.llm.tools import ForecastTools
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finemed_api")
 
FORECAST_OUTPUT_DIR = Path(os.environ.get("FORECAST_OUTPUT_DIR", "data/05_forecasts"))
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")
# Import the SAME path constants the ETL/forecasting pipeline actually
# uses -- NOT independently redefined env-var-configurable versions. Two
# separate constants pointing at "the same" default path is exactly how
# this endpoint and run_pipeline() ended up disagreeing about where raw
# files live during testing. One source of truth, always in agreement.
from finemed_ai.config.paths import RAW_DATA_DIR
from finemed_ai.config.settings import Settings as _Settings
SILVER_DEMAND_PATH = _Settings.DEMAND_FILE

RUN_STATUS = RunStatus(Path(os.environ.get("RUN_STATUS_FILE", "data/run_status.json")))

# Exact file set required per month -- matches
# finemed_ai.validation.validation_config.REQUIRED_FILES. Kept as an
# explicit constant here (not imported) so this endpoint validates the
# upload BEFORE touching the database layer, which needs a working
# Postgres connection just to import -- see the deferred-import note on
# the endpoint below.
REQUIRED_MONTHLY_FILES = [
    "INVOICE.DAT", "INVDET.DAT", "MEDIMAST.DAT", "PURCHASE.DAT",
    "COMPUR.DAT", "SUPMAST.DAT", "SFILE.DAT", "TFILE.DAT",
]

# Shared API key for staff/founder access. Simple by design: this is a
# small team, not an enterprise needing per-user login -- a single shared
# key checked on every client-facing request is the right amount of
# security for this scale, without building a full auth system nobody
# asked for. Upgrade to per-user auth later if the team grows or the
# client wants individual access logs.
CLIENT_API_KEY = os.environ.get("CLIENT_API_KEY")
 
_store: Optional[ForecastStore] = None
_orchestrator: Optional[Orchestrator] = None
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store, _orchestrator
    _store = ForecastStore(FORECAST_OUTPUT_DIR / "latest.parquet")
 
    try:
        _orchestrator = Orchestrator(tools=ForecastTools(_store))
    except Exception:
        logger.exception(
            "Failed to initialize Orchestrator (likely missing ANTHROPIC_API_KEY). "
            "/chat will return 503 until this is fixed; /forecast and /health still work."
        )
        _orchestrator = None
 
    yield  # app runs here
 
    logger.info("Shutting down FineMed API")
 
 
app = FastAPI(
    title="FineMed Pharma AI",
    description="Demand forecasting + LLM Q&A over Chronos-2 forecasts.",
    version="1.0.0",
    lifespan=lifespan,
)
 
# CORS_ALLOWED_ORIGINS: comma-separated list, e.g.
# "https://finemed-dashboard.yourdomain.com,http://localhost:3000"
# No wildcard here -- this serves real client business data, not a public demo.
_cors_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",")
_cors_origins = [o.strip() for o in _cors_origins if o.strip()]
if not _cors_origins:
    logger.warning(
        "CORS_ALLOWED_ORIGINS not set -- no browser-based frontend origin is "
        "allowed yet. Set this env var to your frontend's URL once it exists."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Minimal staff/founder-facing frontend, served directly by this API --
# no separate hosting, no build step, no teammate frontend dependency.
# A fuller frontend can replace this later without touching any endpoint
# below; the JS here only talks to /health, /chat, /forecast/*, same as
# any other frontend would.
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(_STATIC_DIR / "index.html")
 
 
def get_store() -> ForecastStore:
    if _store is None:
        raise HTTPException(status_code=503, detail="Forecast store not initialized")
    if _store.is_stale():
        _store.reload()
    return _store
 
 
def get_orchestrator() -> Orchestrator:
    if _orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Chat is unavailable — ANTHROPIC_API_KEY not configured on the server.",
        )
    return _orchestrator
 
 
def require_admin(x_admin_token: Optional[str] = Header(default=None)) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN not configured on server")
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")


def require_client_auth(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    Guards every staff/founder-facing endpoint (/chat, /forecast/*).
    This is real client business data over the open internet now -- not a
    public portfolio demo -- so every data-returning endpoint needs this,
    not just /refresh.
    """
    if not CLIENT_API_KEY:
        raise HTTPException(status_code=503, detail="CLIENT_API_KEY not configured on server")
    if x_api_key != CLIENT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
 
 
# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
 
class ChatRequest(BaseModel):
    question: str
    conversation_history: Optional[List[Dict]] = None
 
 
class ChatResponse(BaseModel):
    answer: str
 
 
class RefreshResponse(BaseModel):
    status: str
    detail: str
 
 
# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
 
@app.get("/health")
def health():
    store = _store
    return {
        "status": "ok",
        "forecast_store_loaded": store is not None and not store._df.empty if store else False,
        "medicines_available": len(store.list_medicine_ids()) if store else 0,
        "chat_available": _orchestrator is not None,
    }
 
 
@app.get("/forecast/{medicine_id}", dependencies=[Depends(require_client_auth)])
def get_forecast(medicine_id: str, store: ForecastStore = Depends(get_store)):
    try:
        result = store.get(medicine_id)
    except ForecastNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return result.model_dump(mode="json")
 
 
@app.get("/forecast/{medicine_id}/summary", dependencies=[Depends(require_client_auth)])
def get_forecast_summary(medicine_id: str, store: ForecastStore = Depends(get_store)):
    try:
        result = store.get(medicine_id)
    except ForecastNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return result.to_summary().model_dump(mode="json")
 
 
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_client_auth)])
def chat(req: ChatRequest, orchestrator: Orchestrator = Depends(get_orchestrator)):
    answer = orchestrator.ask(req.question, req.conversation_history)
    return ChatResponse(answer=answer)
 
 
@app.post("/refresh", response_model=RefreshResponse, dependencies=[Depends(require_admin)])
def refresh(background_tasks: BackgroundTasks):
    """
    Kicks off a new monthly forecast batch run in the background.
 
    NOTE: this runs Chronos-2 inference for every medicine — it is heavy
    and belongs on a worker with adequate CPU/RAM (or GPU), not the same
    small instance serving /chat and /forecast. On a real deployment, point
    this at a separate worker service or a queue (e.g. a Celery/RQ task) —
    the synchronous background-task version here is fine for a portfolio
    demo but will block your web dyno's other requests while it runs.
    """
    from finemed_ai.demand_forecasting.pipeline import run_monthly_forecast
 
    silver_path = Path(os.environ.get("SILVER_DEMAND_PATH", "data/04_silver/demand_daily.parquet"))
    if not silver_path.exists():
        raise HTTPException(status_code=400, detail=f"Silver demand source not found: {silver_path}")
 
    def _run():
        manifest = run_monthly_forecast(silver_path, FORECAST_OUTPUT_DIR)
        if _store:
            _store.reload()
        logger.info("Background refresh complete: run_id=%s", manifest.run_id)
 
    background_tasks.add_task(_run)
    return RefreshResponse(status="accepted", detail="Forecast refresh started in background.")
 

# ---------------------------------------------------------------------------
# Self-serve monthly upload
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    status: str
    run_id: str
    detail: str


class PipelineStatusResponse(BaseModel):
    run_id: Optional[str] = None
    month: Optional[str] = None
    stage: Optional[str] = None
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    error: Optional[str] = None


def _run_full_monthly_chain(month: str) -> None:
    """
    Runs ETL -> demand prep -> forecast, in order, updating RUN_STATUS at
    each stage. Deferred imports (matching /refresh's existing pattern):
    these modules touch the database at import time, so importing them at
    the top of main.py would make the whole API fail to start if Postgres
    isn't reachable. Importing here means only THIS background task fails,
    not the entire server.
    """
    try:
        RUN_STATUS.update("etl")
        from finemed_ai.pipeline.run_pipeline import run_pipeline
        run_pipeline()

        RUN_STATUS.update("demand_prep")
        from finemed_ai.demand_forecasting.data_preparation import prepare_demand_data
        prepare_demand_data()

        RUN_STATUS.update("forecasting")
        from finemed_ai.demand_forecasting.pipeline import run_monthly_forecast
        manifest = run_monthly_forecast(SILVER_DEMAND_PATH, FORECAST_OUTPUT_DIR)

        if _store:
            _store.reload()

        RUN_STATUS.update("done")
        logger.info(
            "Monthly chain complete for %s: %d/%d medicines forecasted (run_id=%s)",
            month, manifest.medicines_succeeded, manifest.medicines_requested, manifest.run_id,
        )
    except Exception as e:
        logger.exception("Monthly chain failed for month=%s", month)
        RUN_STATUS.fail(str(e))


@app.post(
    "/admin/upload-monthly-data",
    response_model=UploadResponse,
    dependencies=[Depends(require_admin)],
)
def upload_monthly_data(
    background_tasks: BackgroundTasks,
    month: str,
    file: UploadFile = File(...),
):
    """
    Founder/admin uploads a ZIP containing this month's 8 required .DAT
    files. We validate the ZIP contains exactly what's needed BEFORE
    kicking off the (multi-minute) pipeline, so a bad upload fails in
    seconds with a clear message instead of after running for a while.

    `month` should match your raw-data folder naming convention (e.g.
    "2026-08" or whatever your existing data/01_raw/<folder> names use --
    check an existing folder name if unsure).
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a single .zip file containing the 8 .DAT files.")

    month_dir = RAW_DATA_DIR / month
    if month_dir.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Month '{month}' already exists at {month_dir}. "
                    f"Delete it first if you're intentionally re-uploading.",
        )

    tmp_zip_path = RAW_DATA_DIR / f"_upload_{uuid.uuid4().hex[:8]}.zip"
    tmp_zip_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with tmp_zip_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        with zipfile.ZipFile(tmp_zip_path) as zf:
            names_in_zip = {Path(n).name for n in zf.namelist() if not n.endswith("/")}
            missing = set(REQUIRED_MONTHLY_FILES) - names_in_zip
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"ZIP is missing required files: {sorted(missing)}",
                )

            month_dir.mkdir(parents=True, exist_ok=True)
            for required_name in REQUIRED_MONTHLY_FILES:
                matching = [n for n in zf.namelist() if Path(n).name == required_name]
                with zf.open(matching[0]) as src, (month_dir / required_name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    finally:
        tmp_zip_path.unlink(missing_ok=True)

    run_id = uuid.uuid4().hex[:12]
    RUN_STATUS.start(run_id, month)
    background_tasks.add_task(_run_full_monthly_chain, month)

    return UploadResponse(
        status="accepted",
        run_id=run_id,
        detail=f"Files received for {month}. Processing started in the background -- "
               f"check /admin/pipeline-status for progress (this can take several minutes).",
    )


@app.get(
    "/admin/pipeline-status",
    response_model=PipelineStatusResponse,
    dependencies=[Depends(require_admin)],
)
def pipeline_status():
    data = RUN_STATUS.read()
    if data is None:
        return PipelineStatusResponse()
    return PipelineStatusResponse(**data)

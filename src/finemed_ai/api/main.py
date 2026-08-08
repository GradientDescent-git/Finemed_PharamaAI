from __future__ import annotations
 
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
 
from contextlib import asynccontextmanager
 
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
from finemed_ai.demand_forecasting.store import ForecastNotFoundError, ForecastStore
from finemed_ai.llm.orchestrator import Orchestrator
from finemed_ai.llm.tools import ForecastTools
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finemed_api")
 
FORECAST_OUTPUT_DIR = Path(os.environ.get("FORECAST_OUTPUT_DIR", "data/05_forecasts"))
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")
 
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
 
# Tighten this to your actual frontend origin before going live.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
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
 
 
@app.get("/forecast/{medicine_id}")
def get_forecast(medicine_id: str, store: ForecastStore = Depends(get_store)):
    try:
        result = store.get(medicine_id)
    except ForecastNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return result.model_dump(mode="json")
 
 
@app.get("/forecast/{medicine_id}/summary")
def get_forecast_summary(medicine_id: str, store: ForecastStore = Depends(get_store)):
    try:
        result = store.get(medicine_id)
    except ForecastNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return result.to_summary().model_dump(mode="json")
 
 
@app.post("/chat", response_model=ChatResponse)
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
 
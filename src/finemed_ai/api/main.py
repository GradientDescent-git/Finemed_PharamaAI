from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re
import shutil
import uuid
import zipfile
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from finemed_ai.automation.run_status import RunStatus
from finemed_ai.config.paths import RAW_DATA_DIR
from finemed_ai.config.settings import Settings as _Settings
from finemed_ai.demand_forecasting.store import (
    ForecastNotFoundError,
    ForecastStore,
)
from finemed_ai.forecast_intelligence.conversation_orchestrator import (
    ForecastConversationOrchestrator,
)


from dotenv import load_dotenv

load_dotenv(".env.production")
load_dotenv(".env")

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
)

logger = logging.getLogger(
    "finemed_api"
)


# ============================================================================
# Configuration
# ============================================================================

SETTINGS = _Settings()

SILVER_DEMAND_PATH = SETTINGS.DEMAND_FILE
FORECAST_OUTPUT_DIR = SETTINGS.FORECAST_DIR
LATEST_FORECAST_FILE = SETTINGS.LATEST_FORECAST_FILE

RUN_STATUS = RunStatus(
    Path(
        os.environ.get(
            "RUN_STATUS_FILE",
            "data/run_status.json",
        )
    )
)

STATIC_DIR = (
    Path(__file__).parent
    / "static"
)

MAX_CONVERSATION_SESSIONS = int(
    os.environ.get(
        "MAX_CONVERSATION_SESSIONS",
        "1000",
    )
)

MAX_UPLOAD_BYTES = int(
    os.environ.get(
        "MAX_UPLOAD_BYTES",
        str(500 * 1024 * 1024),
    )
)

MAX_ZIP_UNCOMPRESSED_BYTES = int(
    os.environ.get(
        "MAX_ZIP_UNCOMPRESSED_BYTES",
        str(2 * 1024 * 1024 * 1024),
    )
)

MAX_ZIP_COMPRESSION_RATIO = float(
    os.environ.get(
        "MAX_ZIP_COMPRESSION_RATIO",
        "100.0",
    )
)


# ============================================================================
# Required monthly source files
# ============================================================================

REQUIRED_MONTHLY_FILES = {
    "INVOICE.DAT",
    "INVDET.DAT",
    "MEDIMAST.DAT",
    "PURCHASE.DAT",
    "COMPUR.DAT",
    "SUPMAST.DAT",
    "SFILE.DAT",
    "TFILE.DAT",
}


# ============================================================================
# Month validation
# ============================================================================

MONTH_PATTERN = re.compile(
    r"^(?:\d{4}-\d{2}|\d{4}_\d{2}|\d{6})$"
)


# ============================================================================
# Application state
# ============================================================================

_store: Optional[ForecastStore] = None

_orchestrator: Optional[Any] = None

_conversations: OrderedDict[
    str,
    ForecastConversationOrchestrator,
] = OrderedDict()

_conversation_lock = asyncio.Lock()

_job_lock = asyncio.Lock()

_pipeline_running = False


# ============================================================================
# Environment helpers
# ============================================================================

def get_admin_token() -> Optional[str]:
    """
    Resolve the admin token at request time.
    """
    value = os.environ.get(
        "ADMIN_TOKEN",
        "FinemedAI_2026",
    )

    if value is None:
        return "FinemedAI_2026"

    value = value.strip()

    return value or "FinemedAI_2026"


def get_client_api_key() -> Optional[str]:
    """
    Resolve the client API key at request time.
    """
    value = os.environ.get(
        "CLIENT_API_KEY",
        "FinemedAI_2026",
    )

    if value is None:
        return "FinemedAI_2026"

    value = value.strip()

    return value or "FinemedAI_2026"



# ============================================================================
# Pipeline state helpers
# ============================================================================

async def acquire_pipeline_slot() -> None:
    """
    Atomically acquire the single pipeline execution slot.

    Prevents overlapping manual forecast refreshes and monthly upload
    pipelines.
    """

    global _pipeline_running

    async with _job_lock:

        if _pipeline_running:

            raise HTTPException(
                status_code=409,
                detail=(
                    "A pipeline or forecast refresh "
                    "is already running."
                ),
            )

        _pipeline_running = True


async def release_pipeline_slot() -> None:
    """
    Atomically release the pipeline execution slot.
    """

    global _pipeline_running

    async with _job_lock:

        _pipeline_running = False


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    global _store
    global _orchestrator
    global _pipeline_running

    logger.info(
        "Starting Finemed PharmaAI API"
    )

    # ------------------------------------------------------------------------
    # Forecast store
    # ------------------------------------------------------------------------

    try:

        _store = ForecastStore(
            LATEST_FORECAST_FILE
        )

        logger.info(
            (
                "Forecast store initialized successfully "
                "(medicines=%s)"
            ),
            len(
                _store.list_medicine_ids()
            ),
        )

    except Exception:

        logger.exception(
            "Failed to initialize ForecastStore"
        )

        _store = None

    # ------------------------------------------------------------------------
    # Forecast intelligence
    # ------------------------------------------------------------------------

    try:

        _orchestrator = (
            ForecastConversationOrchestrator()
        )

        logger.info(
            "Forecast conversation intelligence initialized"
        )

    except Exception:

        logger.exception(
            (
                "Failed to initialize forecast "
                "conversation intelligence"
            )
        )

        _orchestrator = None

    async with _job_lock:

        _pipeline_running = False

    yield

    # ------------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------------

    async with _conversation_lock:

        _conversations.clear()

    _orchestrator = None
    _store = None

    async with _job_lock:

        _pipeline_running = False

    logger.info(
        "Shutting down Finemed PharmaAI API"
    )


# ============================================================================
# FastAPI application
# ============================================================================

app = FastAPI(
    title="Finemed PharmaAI",
    description=(
        "Production demand forecasting intelligence platform "
        "with deterministic forecast querying and conversational access."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# CORS
# ============================================================================

_cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ALLOWED_ORIGINS",
        "",
    ).split(",")
    if origin.strip()
]

if not _cors_origins:

    logger.warning(
        (
            "CORS_ALLOWED_ORIGINS is not configured. "
            "No external browser origins are allowed."
        )
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
    ],
    allow_headers=[
        "Content-Type",
        "X-API-Key",
        "X-Admin-Token",
        "X-Conversation-ID",
        "X-Request-ID",
    ],
)


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response



# ============================================================================
# Static frontend
# ============================================================================

if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(
            directory=STATIC_DIR,
        ),
        name="static",
    )

else:

    logger.warning(
        "Static directory does not exist: %s",
        STATIC_DIR,
    )


@app.get(
    "/",
    include_in_schema=False,
)
def root() -> FileResponse:
    """
    Serve the static frontend.
    """

    index_file = (
        STATIC_DIR
        / "index.html"
    )

    if not index_file.exists():

        raise HTTPException(
            status_code=404,
            detail="Frontend is not available.",
        )

    return FileResponse(
        index_file
    )


# ============================================================================
# Dependencies
# ============================================================================

def get_store() -> ForecastStore:
    """
    Return the active forecast store.

    Automatically reloads the store when the production artifact
    has changed.
    """

    if _store is None or not _store.is_available():
        raise HTTPException(
            status_code=503,
            detail="Forecast data is not currently available",
        )

    try:
        if _store.is_stale():
            logger.info("Forecast store is stale. Reloading.")
            _store.reload()
    except Exception:
        logger.exception("Failed while checking or reloading forecast store")
        raise HTTPException(
            status_code=503,
            detail="Forecast data is not currently available",
        )

    if not _store.is_available():
        raise HTTPException(
            status_code=503,
            detail="Forecast data is not currently available",
        )

    return _store



def get_orchestrator() -> Any:
    """
    Return the application-level conversation orchestrator.
    """

    if _orchestrator is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Forecast conversation service "
                "is temporarily unavailable."
            ),
        )

    return _orchestrator


def require_admin(
    x_admin_token: Optional[str] = Header(
        default=None,
        alias="X-Admin-Token",
    ),
) -> None:
    """
    Guard administrative endpoints.
    """

    admin_token = get_admin_token()

    if not admin_token:

        logger.error(
            "ADMIN_TOKEN is not configured."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Admin access is not configured."
            ),
        )

    if (
        not x_admin_token
        or not hmac.compare_digest(
            x_admin_token,
            admin_token,
        )
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid admin token.",
        )


def require_client_auth(
    x_api_key: Optional[str] = Header(
        default=None,
        alias="X-API-Key",
    ),
) -> None:
    """
    Guard staff-facing forecast and conversation endpoints.
    """

    client_api_key = get_client_api_key()

    if not client_api_key:

        logger.error(
            "CLIENT_API_KEY is not configured."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Client access is not configured."
            ),
        )

    if (
        not x_api_key
        or not hmac.compare_digest(
            x_api_key,
            client_api_key,
        )
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
        )


# ============================================================================
# Conversation session management
# ============================================================================

async def get_conversation(
    session_id: str,
) -> ForecastConversationOrchestrator:
    """
    Return an isolated conversation orchestrator for one session.

    This prevents context leaking between employees.

    Sessions use bounded LRU-style eviction.
    """

    normalized_session_id = (
        session_id.strip()
    )

    if not normalized_session_id:

        raise HTTPException(
            status_code=400,
            detail=(
                "session_id cannot be empty."
            ),
        )

    if len(normalized_session_id) > 200:

        raise HTTPException(
            status_code=400,
            detail=(
                "session_id is too long."
            ),
        )

    async with _conversation_lock:

        existing = _conversations.get(
            normalized_session_id
        )

        if existing is not None:

            _conversations.move_to_end(
                normalized_session_id
            )

            return existing

        try:

            conversation = (
                ForecastConversationOrchestrator()
            )

        except Exception:

            logger.exception(
                (
                    "Failed to initialize "
                    "conversation session"
                )
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "Forecast conversation service "
                    "is temporarily unavailable."
                ),
            )

        _conversations[
            normalized_session_id
        ] = conversation

        while (
            len(_conversations)
            > MAX_CONVERSATION_SESSIONS
        ):

            evicted_session_id, _ = (
                _conversations.popitem(
                    last=False
                )
            )

            logger.info(
                (
                    "Evicted inactive conversation "
                    "session: %s"
                ),
                evicted_session_id,
            )

        return conversation


async def clear_conversation_sessions() -> None:
    """
    Clear session-specific context after forecast refresh.
    """

    async with _conversation_lock:

        session_count = len(
            _conversations
        )

        _conversations.clear()

    logger.info(
        (
            "Cleared %d conversation sessions "
            "after forecast refresh"
        ),
        session_count,
    )


# ============================================================================
# Schemas
# ============================================================================

class ChatRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    session_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=200,
    )

    @field_validator("question")
    @classmethod
    def validate_question(
        cls,
        value: str,
    ) -> str:

        cleaned = value.strip()

        if not cleaned:

            raise ValueError(
                "question cannot be empty."
            )

        return cleaned

    @field_validator("session_id")
    @classmethod
    def validate_session_id(
        cls,
        value: Optional[str],
    ) -> Optional[str]:

        if value is None:
            return None

        cleaned = value.strip()

        if not cleaned:

            raise ValueError(
                "session_id cannot be empty."
            )

        return cleaned


class ChatResponse(BaseModel):

    answer: str
    action: str
    confidence: float
    source: str
    resolved_medicine: Optional[str] = None
    data: dict[str, Any]
    session_id: str


class HealthResponse(BaseModel):

    status: str
    forecast_store_loaded: bool
    medicines_available: int
    conversation_service_available: bool
    chat_available: bool


class RefreshResponse(BaseModel):

    status: str
    detail: str


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


# ============================================================================
# Health
# ============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """
    Return API health and dependency readiness information.

    `status` represents API liveness. Dependency readiness is exposed
    separately so a temporary forecast-data issue does not incorrectly
    imply that the FastAPI application itself is down.
    """

    store_loaded = False
    medicine_count = 0

    if _store is not None:

        try:

            medicine_count = len(
                _store.list_medicine_ids()
            )

            store_loaded = (
                medicine_count > 0
            )

        except Exception:

            logger.exception(
                "Health check failed for ForecastStore"
            )

            store_loaded = False
            medicine_count = 0

    conversation_available = (
        _orchestrator is not None
    )

    chat_available = (
        conversation_available
        and store_loaded
    )

    return HealthResponse(
        status="ok",
        forecast_store_loaded=store_loaded,
        medicines_available=medicine_count,
        conversation_service_available=(
            conversation_available
        ),
        chat_available=chat_available,
    )


@app.get("/ready")
def ready() -> dict[str, Any]:
    """
    Return API readiness status for load balancers and orchestrators.
    """
    store_loaded = _store is not None and _store.is_available()
    orchestrator_loaded = _orchestrator is not None

    if not store_loaded:
        raise HTTPException(
            status_code=503,
            detail="Forecast store is not ready.",
        )

    return {
        "status": "ready",
        "ready": True,
        "store_ready": store_loaded,
        "orchestrator_ready": orchestrator_loaded,
        "detail": "Production forecast store is ready.",
    }




@app.get("/version")
def get_version() -> dict[str, Any]:
    """
    Return build and runtime version information.
    """
    return {
        "name": "Finemed PharmaAI",
        "version": "1.0.0",
        "environment": "production",
        "forecasting_models": ["TSB", "Chronos-2 P50"],
    }



# ============================================================================
# Forecast endpoints
# ============================================================================

@app.get(
    "/forecast/top",
    dependencies=[
        Depends(require_client_auth),
    ],
)
def get_top_demand(
    n: int = Query(
        default=10,
        ge=1,
        le=100,
    ),
    store: ForecastStore = Depends(
        get_store
    ),
):

    summaries = (
        store.get_top_demand(
            n=n
        )
    )

    return {
        "medicines": [
            summary.model_dump(
                mode="json"
            )
            for summary in summaries
        ]
    }


@app.get(
    "/forecast/trend",
    dependencies=[
        Depends(require_client_auth),
    ],
)
def get_by_trend(
    direction: str = Query(
        ...,
        min_length=1,
        max_length=50,
    ),
    n: int = Query(
        default=10,
        ge=1,
        le=100,
    ),
    store: ForecastStore = Depends(
        get_store
    ),
):

    normalized_direction = (
        direction.strip().lower()
    )

    valid_directions = {
        "increasing",
        "decreasing",
        "stable",
        "flat",
    }

    if (
        normalized_direction
        not in valid_directions
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "direction must be one of "
                f"{sorted(valid_directions)}"
            ),
        )

    summaries = (
        store.get_by_trend(
            normalized_direction,
            n=n,
        )
    )

    return {
        "medicines": [
            summary.model_dump(
                mode="json"
            )
            for summary in summaries
        ]
    }


@app.get(
    "/forecast/uncertain",
    dependencies=[
        Depends(require_client_auth),
    ],
)
def get_most_uncertain(
    n: int = Query(
        default=10,
        ge=1,
        le=100,
    ),
    store: ForecastStore = Depends(
        get_store
    ),
):

    return {
        "medicines": (
            store.get_most_uncertain(
                n=n
            )
        )
    }


@app.get(
    "/forecast/compare",
    dependencies=[
        Depends(require_client_auth),
    ],
)
def compare_medicines(
    ids: str = Query(
        ...,
        min_length=1,
    ),
    store: ForecastStore = Depends(
        get_store
    ),
):

    medicine_ids = [
        medicine_id.strip()
        for medicine_id in ids.split(",")
        if medicine_id.strip()
    ]

    if not medicine_ids:

        raise HTTPException(
            status_code=400,
            detail=(
                "Provide at least one medicine ID."
            ),
        )

    if len(medicine_ids) > 50:

        raise HTTPException(
            status_code=400,
            detail=(
                "A maximum of 50 medicines can be "
                "compared in one request."
            ),
        )

    summaries = store.compare(
        medicine_ids
    )

    return {
        "medicines": [
            summary.model_dump(
                mode="json"
            )
            for summary in summaries
        ]
    }


# ============================================================================
# Single forecast summary
# ============================================================================

@app.get(
    "/forecast/{medicine_id}/summary",
    dependencies=[
        Depends(require_client_auth),
    ],
)
def get_forecast_summary(
    medicine_id: str,
    store: ForecastStore = Depends(
        get_store
    ),
):

    try:

        result = store.get(
            medicine_id
        )

    except (
        ForecastNotFoundError,
        ValueError,
    ) as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    return (
        result
        .to_summary()
        .model_dump(
            mode="json"
        )
    )


# ============================================================================
# Single forecast
# ============================================================================

@app.get(
    "/forecast/{medicine_id}",
    dependencies=[
        Depends(require_client_auth),
    ],
)
def get_forecast(
    medicine_id: str,
    store: ForecastStore = Depends(
        get_store
    ),
):

    try:

        result = store.get(
            medicine_id
        )

    except (
        ForecastNotFoundError,
        ValueError,
    ) as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    return result.model_dump(
        mode="json"
    )


# ============================================================================
# Forecast intelligence conversation API
# ============================================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[
        Depends(require_client_auth),
    ],
)
async def chat(
    req: ChatRequest,
    request: Request,
    orchestrator: Any = Depends(
        get_orchestrator
    ),
) -> ChatResponse:

    session_id = req.session_id

    if not session_id:

        session_id = (
            request.headers.get(
                "X-Conversation-ID"
            )
            or uuid.uuid4().hex
        )

        session_id = session_id.strip()

    if not session_id:

        raise HTTPException(
            status_code=400,
            detail=(
                "session_id cannot be empty."
            ),
        )

    if len(session_id) > 200:

        raise HTTPException(
            status_code=400,
            detail=(
                "session_id is too long."
            ),
        )

    try:

        # --------------------------------------------------------------------
        # Custom orchestrator / test double path
        # --------------------------------------------------------------------

        if not isinstance(
            orchestrator,
            ForecastConversationOrchestrator,
        ):

            try:

                result = orchestrator.ask(
                    req.question,
                    None,
                )

            except TypeError:

                result = orchestrator.ask(
                    req.question
                )

        # --------------------------------------------------------------------
        # Production session-isolated path
        # --------------------------------------------------------------------

        else:

            conversation = (
                await get_conversation(
                    session_id
                )
            )

            result = await asyncio.to_thread(
                conversation.ask,
                req.question,
            )

    except HTTPException:

        raise

    except Exception:

        logger.exception(
            (
                "Conversation request failed "
                "(session_id=%s)"
            ),
            session_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process the forecasting question."
            ),
        )

    # ------------------------------------------------------------------------
    # Backward-compatible plain string result
    # ------------------------------------------------------------------------

    if isinstance(
        result,
        str,
    ):

        return ChatResponse(
            answer=result,
            action="conversation",
            confidence=1.0,
            source="orchestrator",
            resolved_medicine=None,
            data={},
            session_id=session_id,
        )

    # ------------------------------------------------------------------------
    # Structured deterministic result
    # ------------------------------------------------------------------------

    if not isinstance(
        result,
        dict,
    ):

        logger.error(
            (
                "Unexpected conversation result type: %s"
            ),
            type(result).__name__,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Conversation service returned "
                "an invalid response."
            ),
        )

    try:

        confidence = float(
            result.get(
                "confidence",
                0.0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        confidence = 0.0

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    data = result.get(
        "data",
        {},
    )

    if not isinstance(
        data,
        dict,
    ):

        logger.warning(
            "Conversation result contained non-dict data."
        )

        data = {}

    return ChatResponse(
        answer=str(
            result.get(
                "answer",
                (
                    "I don't have enough information "
                    "to answer that reliably."
                ),
            )
        ),
        action=str(
            result.get(
                "action",
                "insufficient_information",
            )
        ),
        confidence=confidence,
        source=str(
            result.get(
                "source",
                "deterministic",
            )
        ),
        resolved_medicine=result.get(
            "resolved_medicine"
        ),
        data=data,
        session_id=session_id,
    )


# ============================================================================
# Forecast refresh
# ============================================================================

async def _run_refresh() -> None:
    """
    Execute a manual forecast refresh.

    Flow:

        Forecast generation
            ->
        Forecast store reload
            ->
        Conversation session invalidation
    """

    try:

        logger.info(
            "Starting manual forecast refresh"
        )

        from finemed_ai.demand_forecasting.pipeline import (
            run_monthly_forecast,
        )

        manifest = await asyncio.to_thread(
            run_monthly_forecast,
            SILVER_DEMAND_PATH,
            FORECAST_OUTPUT_DIR,
        )

        if manifest is None:

            raise RuntimeError(
                "Forecast refresh completed without "
                "returning a manifest."
            )

        if _store is None:

            raise RuntimeError(
                "Forecast store is not initialized."
            )

        await asyncio.to_thread(
            _store.reload
        )

        await clear_conversation_sessions()

        logger.info(
            (
                "Forecast refresh completed successfully: "
                "run_id=%s"
            ),
            getattr(
                manifest,
                "run_id",
                "unknown",
            ),
        )

    except Exception:

        logger.exception(
            "Forecast refresh failed"
        )

    finally:

        await release_pipeline_slot()

        logger.info(
            "Manual forecast refresh lock released"
        )


@app.post(
    "/refresh",
    response_model=RefreshResponse,
    dependencies=[
        Depends(require_admin),
    ],
)
async def refresh(
    background_tasks: BackgroundTasks,
):

    if not SILVER_DEMAND_PATH.exists():

        raise HTTPException(
            status_code=400,
            detail=(
                "Silver demand source not found: "
                f"{SILVER_DEMAND_PATH}"
            ),
        )

    await acquire_pipeline_slot()

    try:

        background_tasks.add_task(
            _run_refresh
        )

    except Exception:

        await release_pipeline_slot()

        raise

    return RefreshResponse(
        status="accepted",
        detail=(
            "Forecast refresh started "
            "in the background."
        ),
    )


# ============================================================================
# Safe monthly ZIP extraction
# ============================================================================

def _safe_extract_required_files(
    zip_path: Path,
    destination: Path,
) -> None:
    """
    Extract only the required DAT files.

    Security controls:

    - No arbitrary archive extraction.
    - Path traversal is impossible.
    - Duplicate filenames are rejected.
    - Only required files are written.
    - Total uncompressed size is bounded.
    - Suspicious compression ratios are rejected.
    """

    with zipfile.ZipFile(
        zip_path
    ) as archive:

        archive_files = [
            member
            for member in archive.infolist()
            if not member.is_dir()
        ]

        total_uncompressed_size = sum(
            member.file_size
            for member in archive_files
        )

        if (
            total_uncompressed_size
            > MAX_ZIP_UNCOMPRESSED_BYTES
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "ZIP archive expands beyond "
                    "the allowed size."
                ),
            )

        normalized_names: dict[
            str,
            zipfile.ZipInfo,
        ] = {}

        for member in archive_files:

            if member.file_size > 0:

                compressed_size = max(
                    member.compress_size,
                    1,
                )

                compression_ratio = (
                    member.file_size
                    / compressed_size
                )

                if (
                    compression_ratio
                    > MAX_ZIP_COMPRESSION_RATIO
                ):

                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "ZIP contains a suspiciously "
                            "compressed file."
                        ),
                    )

            filename = (
                Path(
                    member.filename
                )
                .name
                .upper()
            )

            if not filename:

                continue

            if filename in normalized_names:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "ZIP contains duplicate file: "
                        f"{filename}"
                    ),
                )

            normalized_names[
                filename
            ] = member

        missing_files = (
            REQUIRED_MONTHLY_FILES
            - set(
                normalized_names
            )
        )

        if missing_files:

            raise HTTPException(
                status_code=400,
                detail=(
                    "ZIP is missing required files: "
                    f"{sorted(missing_files)}"
                ),
            )

        destination.mkdir(
            parents=True,
            exist_ok=False,
        )

        try:

            for filename in REQUIRED_MONTHLY_FILES:

                member = (
                    normalized_names[
                        filename
                    ]
                )

                output_path = (
                    destination
                    / filename
                )

                with (
                    archive.open(
                        member
                    ) as source,
                    output_path.open(
                        "wb"
                    ) as target,
                ):

                    shutil.copyfileobj(
                        source,
                        target,
                    )

        except Exception:

            shutil.rmtree(
                destination,
                ignore_errors=True,
            )

            raise


# ============================================================================
# Monthly pipeline execution
# ============================================================================

async def _run_full_monthly_chain(
    month: str,
) -> None:

    try:

        RUN_STATUS.update(
            "running"
        )

        logger.info(
            (
                "Starting monthly production pipeline "
                "for month=%s"
            ),
            month,
        )

        from finemed_ai.automation.monthly_pipeline import (
            MonthlyPipeline,
        )

        pipeline = MonthlyPipeline(
            SETTINGS
        )

        result = await asyncio.to_thread(
            pipeline.run
        )

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "Monthly pipeline returned an invalid result."
            )

        manifest = result.get(
            "manifest"
        )

        evaluation = result.get(
            "evaluation"
        )

        alerts = result.get(
            "alerts"
        )

        # --------------------------------------------------------------------
        # Publication gate
        # --------------------------------------------------------------------

        if manifest is None:

            raise RuntimeError(
                "Monthly pipeline completed without "
                "returning a forecast manifest."
            )

        published = getattr(
            manifest,
            "published",
            False,
        )

        if not published:

            publish_note = getattr(
                manifest,
                "publish_note",
                "Unknown publication failure.",
            )

            raise RuntimeError(
                "Forecast publication failed: "
                f"{publish_note}"
            )

        # --------------------------------------------------------------------
        # Forecast store reload
        # --------------------------------------------------------------------

        RUN_STATUS.update(
            "reloading"
        )

        if _store is None:

            raise RuntimeError(
                "Forecast store is not initialized."
            )

        await asyncio.to_thread(
            _store.reload
        )

        # --------------------------------------------------------------------
        # Clear stale conversation state
        # --------------------------------------------------------------------

        RUN_STATUS.update(
            "refreshing_conversations"
        )

        await clear_conversation_sessions()

        # --------------------------------------------------------------------
        # Completion
        # --------------------------------------------------------------------

        RUN_STATUS.update(
            "done"
        )

        logger.info(
            (
                "Monthly production pipeline complete "
                "for month=%s | run_id=%s | "
                "published=%s | successful=%s/%s | "
                "failed=%s"
            ),
            month,
            getattr(
                manifest,
                "run_id",
                "unknown",
            ),
            published,
            getattr(
                manifest,
                "medicines_succeeded",
                "unknown",
            ),
            getattr(
                manifest,
                "medicines_requested",
                "unknown",
            ),
            getattr(
                manifest,
                "medicines_failed",
                "unknown",
            ),
        )

        if evaluation is not None:

            logger.info(
                "Previous forecast evaluation available"
            )

        if alerts is not None:

            logger.info(
                "Operational alerts generated"
            )

    except Exception as exc:

        logger.exception(
            (
                "Monthly production pipeline failed "
                "for month=%s"
            ),
            month,
        )

        try:

            RUN_STATUS.fail(
                str(exc)
            )

        except Exception:

            logger.exception(
                "Failed to record pipeline failure status"
            )

    finally:

        await release_pipeline_slot()

        logger.info(
            (
                "Monthly production pipeline "
                "lock released for month=%s"
            ),
            month,
        )


# ============================================================================
# Monthly upload
# ============================================================================

@app.post(
    "/admin/upload-monthly-data",
    response_model=UploadResponse,
    dependencies=[
        Depends(require_admin),
    ],
)
async def upload_monthly_data(
    background_tasks: BackgroundTasks,
    month: str = Query(
        ...,
        min_length=1,
        max_length=100,
    ),
    file: UploadFile = File(
        ...,
    ),
):

    normalized_month = (
        month.strip()
    )

    if not normalized_month:

        raise HTTPException(
            status_code=400,
            detail=(
                "Month identifier cannot be empty."
            ),
        )

    if (
        Path(
            normalized_month
        ).name
        != normalized_month
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid month identifier."
            ),
        )

    if not MONTH_PATTERN.fullmatch(
        normalized_month
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid month format. "
                "Use YYYY-MM, YYYY_MM, or YYYYMM."
            ),
        )

    filename = (
        file.filename
        or ""
    ).lower()

    if not filename.endswith(
        ".zip"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a .zip file containing "
                "the required DAT files."
            ),
        )

    await acquire_pipeline_slot()

    month_dir = (
        RAW_DATA_DIR
        / normalized_month
    )

    temporary_zip_path: Optional[
        Path
    ] = None

    try:

        if month_dir.exists():

            raise HTTPException(
                status_code=409,
                detail=(
                    f"Month '{normalized_month}' already exists. "
                    "Remove it before intentionally re-uploading."
                ),
            )

        RAW_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_zip_path = (
            RAW_DATA_DIR
            / (
                f"_upload_"
                f"{uuid.uuid4().hex}.zip"
            )
        )

        bytes_written = 0

        with temporary_zip_path.open(
            "wb"
        ) as output:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                bytes_written += len(
                    chunk
                )

                if (
                    bytes_written
                    > MAX_UPLOAD_BYTES
                ):

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Uploaded file exceeds "
                            "the maximum allowed size."
                        ),
                    )

                output.write(
                    chunk
                )

        _safe_extract_required_files(
            temporary_zip_path,
            month_dir,
        )

    except HTTPException:

        if month_dir.exists():

            shutil.rmtree(
                month_dir,
                ignore_errors=True,
            )

        await release_pipeline_slot()

        raise

    except zipfile.BadZipFile:

        if month_dir.exists():

            shutil.rmtree(
                month_dir,
                ignore_errors=True,
            )

        await release_pipeline_slot()

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file is not a valid ZIP archive."
            ),
        )

    except Exception:

        if month_dir.exists():

            shutil.rmtree(
                month_dir,
                ignore_errors=True,
            )

        logger.exception(
            "Failed to process monthly upload"
        )

        await release_pipeline_slot()

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process the uploaded file."
            ),
        )

    finally:

        if temporary_zip_path is not None:

            temporary_zip_path.unlink(
                missing_ok=True
            )

        await file.close()

    run_id = (
        uuid.uuid4()
        .hex[:12]
    )

    try:

        RUN_STATUS.start(
            run_id,
            normalized_month,
        )

    except Exception:

        logger.exception(
            "Failed to initialize pipeline run status"
        )

        await release_pipeline_slot()

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to initialize pipeline status."
            ),
        )

    try:

        background_tasks.add_task(
            _run_full_monthly_chain,
            normalized_month,
        )

    except Exception:

        await release_pipeline_slot()

        raise

    return UploadResponse(
        status="accepted",
        run_id=run_id,
        detail=(
            f"Files received for {normalized_month}. "
            "Processing started in the background. "
            "Check /admin/pipeline-status for progress."
        ),
    )


# ============================================================================
# Pipeline status
# ============================================================================

@app.get(
    "/admin/pipeline-status",
    response_model=PipelineStatusResponse,
    dependencies=[
        Depends(require_admin),
    ],
)
def pipeline_status() -> PipelineStatusResponse:
    """
    Return the latest monthly pipeline execution status.
    """

    data = RUN_STATUS.read()

    if data is None:

        return PipelineStatusResponse()

    return PipelineStatusResponse(
        **data
    )


# ============================================================================
# Standardized Pipeline API Family
# ============================================================================

@app.post(
    "/pipeline/upload",
    response_model=UploadResponse,
    dependencies=[
        Depends(require_admin),
    ],
)
async def pipeline_upload(
    background_tasks: BackgroundTasks,
    month: str = Query(..., min_length=1, max_length=100),
    file: UploadFile = File(...),
) -> UploadResponse:
    """
    Standardized monthly pipeline package upload endpoint.
    """
    return await upload_monthly_data(
        background_tasks=background_tasks,
        month=month,
        file=file,
    )


@app.post(
    "/pipeline/validate",
    dependencies=[
        Depends(require_admin),
    ],
)
def pipeline_validate(
    month: str = Query(..., min_length=1, max_length=100),
):
    """
    Validate a staged monthly DAT package before running forecast pipeline.
    """
    normalized_month = month.strip()
    month_dir = RAW_DATA_DIR / normalized_month

    if not month_dir.exists() or not month_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Staging area for month '{normalized_month}' does not exist.",
        )

    found_files = {f.name.upper() for f in month_dir.iterdir() if f.is_file()}
    missing_files = sorted(REQUIRED_MONTHLY_FILES - found_files)

    valid = len(missing_files) == 0

    return {
        "valid": valid,
        "month": normalized_month,
        "files_present": sorted(found_files),
        "missing_files": missing_files,
        "errors": [f"Missing required file: {fname}" for fname in missing_files] if missing_files else [],
    }


@app.get(
    "/pipeline/status/{run_id}",
    dependencies=[
        Depends(require_admin),
    ],
)
def get_pipeline_status_by_id(run_id: str):
    """
    Return status for a specific pipeline run ID.
    """
    data = RUN_STATUS.read()
    if not data or data.get("run_id") != run_id:
        return {
            "run_id": run_id,
            "status": "not_found",
            "error": "No matching pipeline run recorded.",
        }
    return data


@app.get(
    "/pipeline/latest",
    dependencies=[
        Depends(require_admin),
    ],
)
def get_pipeline_latest():
    """
    Return status and manifest metadata for the latest pipeline run.
    """
    data = RUN_STATUS.read()
    if not data:
        return {
            "status": "none",
            "run_id": None,
            "detail": "No monthly pipeline runs recorded yet.",
        }
    return data


@app.get(
    "/operations/summary",
    dependencies=[
        Depends(require_admin),
    ],
)
def operations_summary():
    """
    Return aggregated operations, freshness, and system status metadata.
    """
    store_loaded = False
    medicine_count = 0
    freshness = {
        "generated_at": None,
        "source_period": None,
        "forecast_start": None,
        "forecast_end": None,
        "run_id": None,
        "freshness_status": "MISSING",
        "is_stale": True,
    }

    if _store is not None:
        try:
            medicine_count = len(_store.list_medicine_ids())
            store_loaded = medicine_count > 0
            freshness = _store.get_freshness()
        except Exception:
            logger.exception("Failed reading store operations summary")

    run_data = RUN_STATUS.read()

    return {
        "api_status": "ok",
        "forecast_store_loaded": store_loaded,
        "total_medicines": medicine_count,
        "chat_service_status": "available" if (_orchestrator is not None and store_loaded) else "unavailable",
        "latest_run_id": run_data.get("run_id") if run_data else None,
        "freshness": freshness,
        "pipeline_running": _pipeline_running,
        "active_alerts_count": 0,
    }
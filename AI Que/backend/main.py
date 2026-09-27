"""
main.py — FastAPI application entry point.

Start with:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend import database, model, scheduler
from backend.schemas import (
    HealthResponse,
    PredictionResponse,
    QueueUpdateRequest,
    QueueLogEntry,
    ServiceEventRequest,
    TokenCreateRequest,
    TokenResponse,
    TokenStatusUpdateRequest,
    TokenSummaryResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀  Starting AI Queue Prediction System …")
    database.init_db()
    model.load_model()
    if not model.is_model_ready():
        logger.warning("⚠️  No trained model found — will use formula fallback. Run /api/retrain to train.")
    scheduler.start_scheduler()
    yield
    scheduler.stop_scheduler()
    logger.info("👋  Shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Queue Prediction API",
    description="Predicts waiting time at bank/government office queues and manages smart tokens.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_prediction_response(
    queue_depth: int,
    active_counters: int,
    avg_service_time_mins: float,
    predicted_wait: float,
    confidence: str,
) -> PredictionResponse:
    if predicted_wait < 10:
        label, color = "short", "#22c55e"
    elif predicted_wait < 25:
        label, color = "moderate", "#f59e0b"
    else:
        label, color = "long", "#ef4444"

    message = (
        f"~{int(predicted_wait)} min wait · {queue_depth} "
        f"{'person' if queue_depth == 1 else 'people'} ahead"
    )

    return PredictionResponse(
        queue_depth=queue_depth,
        active_counters=active_counters,
        avg_service_time_mins=avg_service_time_mins,
        predicted_wait_mins=predicted_wait,
        confidence=confidence,
        status_label=label,
        status_color=color,
        message=message,
    )


def _compute_current_prediction():
    """Helper to evaluate wait time for queue or token issuance."""
    now = datetime.now()
    logs = database.get_recent_logs(limit=1)
    
    # Check currently waiting tokens in DB to enrich or use as queue count
    tokens_summary = database.get_token_summary()
    waiting_tokens = tokens_summary["waiting"]

    if logs:
        last = logs[0]
        # Use maximum of manual queue depth or active waiting tokens
        queue_depth = max(last["queue_depth"], waiting_tokens)
        active_counters = max(1, last["active_counters"])
    else:
        queue_depth = waiting_tokens
        active_counters = 3

    avg_svc = database.get_rolling_avg_service_time()

    predicted_wait, confidence = model.predict(
        hour=now.hour,
        minute=now.minute,
        day_of_week=now.weekday(),
        week_of_year=now.isocalendar()[1],
        month=now.month,
        quarter=(now.month - 1) // 3 + 1,
        is_month_start=1 if now.day <= 3 else 0,
        is_month_end=1 if now.day >= 28 else 0,
        is_holiday_adjacent=0,
        queue_depth=queue_depth,
        active_counters=active_counters,
        avg_service_time_mins=avg_svc,
    )

    return queue_depth, active_counters, avg_svc, predicted_wait, confidence


# ---------------------------------------------------------------------------
# UI Page Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to the public display page."""
    return FileResponse(str(FRONTEND_DIR / "display.html"))


@app.get("/staff", include_in_schema=False)
async def staff_page():
    return FileResponse(str(FRONTEND_DIR / "staff.html"))


@app.get("/tokens", include_in_schema=False)
async def token_kiosk_page():
    """Token dispenser & customer check-in page."""
    return FileResponse(str(FRONTEND_DIR / "tokens.html"))


# ---------------------------------------------------------------------------
# Queue Routes
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check — returns model readiness and DB path."""
    return HealthResponse(
        status="ok",
        model_ready=model.is_model_ready(),
        db_path=str(database.DB_PATH),
    )


@app.get("/api/status", response_model=PredictionResponse, tags=["Queue"])
async def get_status():
    """
    Get current queue status using recent observations and tokens.
    Uses the ML model (or formula fallback) to predict wait time.
    """
    queue_depth, active_counters, avg_svc, predicted_wait, confidence = _compute_current_prediction()
    return _build_prediction_response(
        queue_depth=queue_depth,
        active_counters=active_counters,
        avg_service_time_mins=avg_svc,
        predicted_wait=predicted_wait,
        confidence=confidence,
    )


@app.post("/api/queue/update", response_model=PredictionResponse, tags=["Queue"])
async def update_queue(req: QueueUpdateRequest):
    """
    Staff endpoint: log current queue depth + counter count,
    get back an immediate prediction.
    """
    now = datetime.now()
    avg_svc = req.avg_service_time_mins or database.get_rolling_avg_service_time()

    predicted_wait, confidence = model.predict(
        hour=now.hour,
        minute=now.minute,
        day_of_week=now.weekday(),
        week_of_year=now.isocalendar()[1],
        month=now.month,
        quarter=(now.month - 1) // 3 + 1,
        is_month_start=1 if now.day <= 3 else 0,
        is_month_end=1 if now.day >= 28 else 0,
        is_holiday_adjacent=0,
        queue_depth=req.queue_depth,
        active_counters=req.active_counters,
        avg_service_time_mins=avg_svc,
    )

    database.insert_queue_log(
        queue_depth=req.queue_depth,
        active_counters=req.active_counters,
        avg_service_time_mins=avg_svc,
        predicted_wait_mins=predicted_wait,
    )

    return _build_prediction_response(
        queue_depth=req.queue_depth,
        active_counters=req.active_counters,
        avg_service_time_mins=avg_svc,
        predicted_wait=predicted_wait,
        confidence=confidence,
    )


@app.post("/api/service/log", tags=["Queue"])
async def log_service_event(req: ServiceEventRequest):
    """
    Log a completed service event to improve the rolling average.
    Call this each time a customer finishes being served.
    """
    database.insert_service_event(req.service_time_mins)
    new_avg = database.get_rolling_avg_service_time()
    return {"message": "Service event logged", "new_avg_service_time_mins": new_avg}


@app.get("/api/history", response_model=list[QueueLogEntry], tags=["Queue"])
async def get_history(limit: int = 50):
    """Return recent queue log entries (most recent first)."""
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    return database.get_recent_logs(limit=limit)


# ---------------------------------------------------------------------------
# Token & Slot Assignment Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/tokens/issue", response_model=TokenResponse, tags=["Tokens"])
async def create_token(req: TokenCreateRequest):
    """
    Generate a customer token assigned according to AI estimated queue wait times.
    """
    _, _, _, predicted_wait, _ = _compute_current_prediction()
    token = database.issue_token(
        service_type=req.service_type,
        customer_name=req.customer_name,
        predicted_wait_mins=predicted_wait
    )
    return token


@app.get("/api/tokens/active", response_model=list[TokenResponse], tags=["Tokens"])
async def get_active_tokens():
    """List tokens waiting or serving."""
    return database.get_waiting_tokens()


@app.get("/api/tokens/summary", response_model=TokenSummaryResponse, tags=["Tokens"])
async def get_token_summary():
    """Summary of token queue stats today."""
    return database.get_token_summary()


@app.get("/api/tokens/{token_number}", response_model=TokenResponse, tags=["Tokens"])
async def lookup_token(token_number: str):
    """Check estimated time, status, and queue position for a given token."""
    token = database.get_token_by_number(token_number)
    if not token:
        raise HTTPException(status_code=404, detail="Token not found")
    return token


@app.patch("/api/tokens/{token_number}/status", response_model=TokenResponse, tags=["Tokens"])
async def change_token_status(token_number: str, req: TokenStatusUpdateRequest):
    """
    Staff call: Mark token as 'serving' at counter X, or 'completed' / 'cancelled'.
    """
    updated = database.update_token_status(
        token_number=token_number,
        status=req.status,
        counter_id=req.counter_id
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Token not found")
    
    # If completed and called_at is available, calculate actual service time & log it
    if req.status == "completed" and updated.get("called_at") and updated.get("completed_at"):
        try:
            t1 = datetime.fromisoformat(updated["called_at"])
            t2 = datetime.fromisoformat(updated["completed_at"])
            mins = max(0.5, round((t2 - t1).total_seconds() / 60.0, 1))
            database.insert_service_event(mins)
        except Exception as e:
            logger.warning("Could not auto-log service event from token completion: %s", e)

    return updated


# ---------------------------------------------------------------------------
# Training Trigger
# ---------------------------------------------------------------------------

@app.post("/api/retrain", tags=["System"])
async def retrain_model():
    """
    Manually trigger model retraining using historical CSV + live DB data.
    This is run automatically every night at 01:00.
    """
    try:
        extra_rows = database.get_logs_for_training()
        stats = model.train(extra_rows=extra_rows if extra_rows else None)
        return {"message": "Model retrained successfully", **stats}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

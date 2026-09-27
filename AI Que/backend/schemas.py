"""
schemas.py — Pydantic v2 request/response models for the Queue Prediction API.
Includes Token generation, verification, and transition schemas.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any


class QueueUpdateRequest(BaseModel):
    """Payload sent by staff when they log the current queue state."""
    queue_depth: int = Field(..., ge=0, le=500, description="Number of people in queue right now")
    active_counters: int = Field(..., ge=1, le=20, description="Number of open service windows")
    avg_service_time_mins: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=60.0,
        description="Override average service time in minutes (uses rolling DB avg if omitted)",
    )


class ServiceEventRequest(BaseModel):
    """Log a completed service event to improve rolling average."""
    service_time_mins: float = Field(..., ge=0.5, le=60.0)


class PredictionResponse(BaseModel):
    """Response from the prediction engine."""
    queue_depth: int
    active_counters: int
    avg_service_time_mins: float
    predicted_wait_mins: float
    confidence: str  # "model" | "formula"
    status_label: str  # "short" | "moderate" | "long"
    status_color: str  # hex colour for the display UI
    message: str


class QueueLogEntry(BaseModel):
    id: int
    timestamp: str
    hour: int
    day_of_week: int
    is_month_start: int
    is_month_end: int
    queue_depth: int
    active_counters: int
    avg_service_time_mins: float
    actual_wait_mins: Optional[float]
    predicted_wait_mins: Optional[float]


class HealthResponse(BaseModel):
    # 'model_ready' trips pydantic's protected 'model_' namespace otherwise
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_ready: bool
    db_path: str


# ---------------------------------------------------------------------------
# Token schemas
# ---------------------------------------------------------------------------

class TokenCreateRequest(BaseModel):
    service_type: str = Field(default="General Banking", description="Type of counter service requested")
    customer_name: Optional[str] = Field(default=None, max_length=60, description="Optional name of the visitor")


class TokenResponse(BaseModel):
    id: int
    token_number: str
    service_type: str
    customer_name: Optional[str]
    created_at: str
    estimated_time: str
    estimated_wait_mins: float
    status: str
    people_ahead: int
    counter_id: Optional[int] = None
    called_at: Optional[str] = None
    completed_at: Optional[str] = None


class TokenStatusUpdateRequest(BaseModel):
    status: str = Field(..., pattern="^(serving|completed|cancelled|waiting)$")
    counter_id: Optional[int] = Field(default=None, ge=1, le=20)


class TokenSummaryResponse(BaseModel):
    waiting: int
    serving: int
    completed: int
    active_serving: List[Dict[str, Any]]

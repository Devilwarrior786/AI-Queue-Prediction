"""
database.py — SQLite setup and CRUD operations using the standard library.
Includes token assignment, queue management, and service events.
"""

import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "data" / "queue.db"

# Columns added to queue_logs after the first release (for migrating older DBs)
NEW_QUEUE_LOG_COLUMNS = (
    ("minute", "INTEGER NOT NULL DEFAULT 0"),
    ("week_of_year", "INTEGER NOT NULL DEFAULT 1"),
    ("month", "INTEGER NOT NULL DEFAULT 1"),
    ("quarter", "INTEGER NOT NULL DEFAULT 1"),
    ("is_holiday_adjacent", "INTEGER NOT NULL DEFAULT 0"),
)


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory set for dict-like rows."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the first release to pre-existing databases."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(queue_logs)").fetchall()}
    for col, decl in NEW_QUEUE_LOG_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE queue_logs ADD COLUMN {col} {decl}")
            logger.info("Migrated queue_logs: added column %s", col)


def init_db() -> None:
    """Create tables if they don't exist yet, and migrate older schemas."""
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS queue_logs (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp             TEXT    NOT NULL,
                hour                  INTEGER NOT NULL,
                minute                INTEGER NOT NULL DEFAULT 0,
                day_of_week           INTEGER NOT NULL,
                week_of_year          INTEGER NOT NULL DEFAULT 1,
                month                 INTEGER NOT NULL DEFAULT 1,
                quarter               INTEGER NOT NULL DEFAULT 1,
                is_month_start        INTEGER NOT NULL DEFAULT 0,
                is_month_end          INTEGER NOT NULL DEFAULT 0,
                is_holiday_adjacent   INTEGER NOT NULL DEFAULT 0,
                queue_depth           INTEGER NOT NULL,
                active_counters       INTEGER NOT NULL,
                avg_service_time_mins REAL    NOT NULL DEFAULT 4.5,
                actual_wait_mins      REAL,
                predicted_wait_mins   REAL
            );

            CREATE TABLE IF NOT EXISTS service_events (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT    NOT NULL,
                service_time_mins REAL   NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tokens (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                token_number     TEXT    NOT NULL UNIQUE,
                service_type     TEXT    NOT NULL DEFAULT 'General Banking',
                customer_name    TEXT,
                created_at       TEXT    NOT NULL,
                estimated_time   TEXT    NOT NULL,
                estimated_wait_mins REAL NOT NULL,
                status           TEXT    NOT NULL DEFAULT 'waiting', -- waiting, serving, completed, cancelled
                counter_id       INTEGER,
                called_at        TEXT,
                completed_at     TEXT
            );
            """
        )
        _migrate_schema(conn)


# ---------------------------------------------------------------------------
# Queue log helpers
# ---------------------------------------------------------------------------

def insert_queue_log(
    queue_depth: int,
    active_counters: int,
    avg_service_time_mins: float,
    predicted_wait_mins: float | None = None,
    actual_wait_mins: float | None = None,
) -> int:
    """Insert a new queue observation (with all model features) and return its id."""
    now = datetime.now()
    ts = now.isoformat(timespec="seconds")
    month = now.month
    row = (
        ts,
        now.hour,
        now.minute,
        now.weekday(),
        now.isocalendar()[1],
        month,
        (month - 1) // 3 + 1,
        1 if now.day <= 3 else 0,
        1 if now.day >= 28 else 0,
        0,  # is_holiday_adjacent — not tracked live
        queue_depth,
        active_counters,
        avg_service_time_mins,
        actual_wait_mins,
        predicted_wait_mins,
    )
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO queue_logs
                (timestamp, hour, minute, day_of_week, week_of_year, month, quarter,
                 is_month_start, is_month_end, is_holiday_adjacent,
                 queue_depth, active_counters, avg_service_time_mins,
                 actual_wait_mins, predicted_wait_mins)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        return cur.lastrowid


def get_recent_logs(limit: int = 50) -> list[dict]:
    """Return the most recent queue log entries."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM queue_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_logs_for_training() -> list[dict]:
    """Return all logs that have an actual_wait_mins recorded (used for ML training)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM queue_logs WHERE actual_wait_mins IS NOT NULL"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Service event helpers (for rolling average service time)
# ---------------------------------------------------------------------------

def insert_service_event(service_time_mins: float) -> None:
    """Log a completed service event."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO service_events (timestamp, service_time_mins) VALUES (?, ?)",
            (datetime.now().isoformat(timespec="seconds"), service_time_mins),
        )


def get_rolling_avg_service_time(last_n: int = 50) -> float:
    """Return the average service time over the last N events, or default 4.5 min."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT AVG(service_time_mins) as avg
            FROM (
                SELECT service_time_mins FROM service_events
                ORDER BY id DESC LIMIT ?
            )
            """,
            (last_n,),
        ).fetchone()
    avg = row["avg"] if row and row["avg"] is not None else None
    return round(avg, 2) if avg else 4.5


# ---------------------------------------------------------------------------
# Token management helpers
# ---------------------------------------------------------------------------

def generate_next_token_number(service_type: str = "General") -> str:
    """
    Create token numbers like C-101, C-102 depending on service type.

    The sequence is derived from the highest number that already exists for the
    prefix (not from a daily count), so numbers are never reused across days —
    daily resets used to violate the UNIQUE constraint and crash issuance.
    """
    prefix_map = {
        "Cash Deposit / Withdrawal": "C",
        "Account Services": "A",
        "Loans & Mortgages": "L",
        "Govt Documentation / Passbook": "G",
        "General Banking": "T"
    }
    prefix = prefix_map.get(service_type, "T")

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT MAX(CAST(SUBSTR(token_number, ?) AS INTEGER)) as max_seq
            FROM tokens
            WHERE token_number LIKE ?
            """,
            (len(prefix) + 2, f"{prefix}-%"),
        ).fetchone()

    max_seq = 100
    if row and row["max_seq"] is not None:
        max_seq = int(row["max_seq"])
    return f"{prefix}-{max_seq + 1}"


def issue_token(
    service_type: str,
    customer_name: str | None,
    predicted_wait_mins: float
) -> dict:
    """Issue a new token, calculating the estimated service slot timestamp."""
    now = datetime.now()
    created_at = now.isoformat(timespec="seconds")
    est_service_dt = now + timedelta(minutes=predicted_wait_mins)
    estimated_time = est_service_dt.strftime("%I:%M %p")

    with get_connection() as conn:
        # Retry on the (rare) race where two kiosks allocate the same number
        for attempt in range(5):
            token_num = generate_next_token_number(service_type)
            try:
                cur = conn.execute(
                    """
                    INSERT INTO tokens (token_number, service_type, customer_name, created_at,
                                        estimated_time, estimated_wait_mins, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'waiting')
                    """,
                    (token_num, service_type, customer_name, created_at, estimated_time, predicted_wait_mins),
                )
                break
            except sqlite3.IntegrityError:
                if attempt == 4:
                    raise
        token_id = cur.lastrowid
        row = conn.execute("SELECT * FROM tokens WHERE id = ?", (token_id,)).fetchone()
        
        # Also return how many people are ahead of this token
        ahead_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM tokens WHERE status = 'waiting' AND id < ?",
            (token_id,)
        ).fetchone()["cnt"]

    res = dict(row)
    res["people_ahead"] = ahead_count
    return res


def get_waiting_tokens() -> list[dict]:
    """Retrieve currently waiting/serving tokens with their queue position."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM tokens
            WHERE status IN ('waiting', 'serving')
            ORDER BY id ASC
            """
        ).fetchall()
        results = []
        for r in rows:
            res = dict(r)
            if res["status"] == "waiting":
                res["people_ahead"] = conn.execute(
                    "SELECT COUNT(*) as cnt FROM tokens WHERE status = 'waiting' AND id < ?",
                    (res["id"],)
                ).fetchone()["cnt"]
            else:
                res["people_ahead"] = 0
            results.append(res)
    return results


def get_token_by_number(token_number: str) -> dict | None:
    """Lookup a token status and position."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM tokens WHERE token_number = ?", (token_number.strip().upper(),)
        ).fetchone()
        if not row:
            return None
        res = dict(row)
        if res["status"] == "waiting":
            ahead = conn.execute(
                "SELECT COUNT(*) as cnt FROM tokens WHERE status = 'waiting' AND id < ?",
                (res["id"],)
            ).fetchone()["cnt"]
            res["people_ahead"] = ahead
        else:
            res["people_ahead"] = 0
        return res


def update_token_status(token_number: str, status: str, counter_id: int | None = None) -> dict | None:
    """
    Update a token state (serving, completed, cancelled).

    When a token is called ('serving'), the customer's real wait
    (issue → call) is written to queue_logs with the observed system
    state, so nightly retraining learns from live operations.
    """
    now_dt = datetime.now()
    now = now_dt.isoformat(timespec="seconds")

    # Compute outside the transaction to avoid nested connections
    avg_svc = get_rolling_avg_service_time()

    with get_connection() as conn:
        token = conn.execute("SELECT * FROM tokens WHERE token_number = ?", (token_number,)).fetchone()
        if not token:
            return None

        if status == "serving":
            conn.execute(
                "UPDATE tokens SET status = ?, counter_id = ?, called_at = ? WHERE token_number = ?",
                (status, counter_id, now, token_number),
            )
        elif status == "completed":
            conn.execute(
                "UPDATE tokens SET status = ?, completed_at = ? WHERE token_number = ?",
                (status, now, token_number),
            )
        else:
            conn.execute(
                "UPDATE tokens SET status = ? WHERE token_number = ?",
                (status, token_number),
            )

        updated = conn.execute("SELECT * FROM tokens WHERE token_number = ?", (token_number,)).fetchone()
        res = dict(updated)

        # people_ahead is required by the TokenResponse schema
        if status == "waiting":
            res["people_ahead"] = conn.execute(
                "SELECT COUNT(*) as cnt FROM tokens WHERE status = 'waiting' AND id < ?",
                (res["id"],)
            ).fetchone()["cnt"]
        else:
            res["people_ahead"] = 0

        # --- Continuous feedback loop: log the realised wait for training ---
        if status == "serving" and token["created_at"]:
            try:
                actual_wait = max(
                    0.0,
                    (now_dt - datetime.fromisoformat(token["created_at"])).total_seconds() / 60.0,
                )
                waiting_now = conn.execute(
                    "SELECT COUNT(*) as c FROM tokens WHERE status = 'waiting'"
                ).fetchone()["c"]
                counters_busy = max(1, conn.execute(
                    "SELECT COUNT(DISTINCT counter_id) as c FROM tokens "
                    "WHERE status = 'serving' AND counter_id IS NOT NULL"
                ).fetchone()["c"])
                month = now_dt.month
                conn.execute(
                    """
                    INSERT INTO queue_logs
                        (timestamp, hour, minute, day_of_week, week_of_year, month, quarter,
                         is_month_start, is_month_end, is_holiday_adjacent,
                         queue_depth, active_counters, avg_service_time_mins,
                         actual_wait_mins, predicted_wait_mins)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now, now_dt.hour, now_dt.minute, now_dt.weekday(),
                        now_dt.isocalendar()[1], month, (month - 1) // 3 + 1,
                        1 if now_dt.day <= 3 else 0,
                        1 if now_dt.day >= 28 else 0,
                        0,
                        waiting_now + 1,
                        counters_busy,
                        avg_svc,
                        round(actual_wait, 1),
                        token["estimated_wait_mins"],
                    ),
                )
            except Exception:
                logger.exception("Failed to log realised wait for token %s", token_number)

        return res


def get_token_summary() -> dict:
    """Get active count stats for tokens today."""
    today = datetime.now().strftime("%Y-%m-%d")
    with get_connection() as conn:
        waiting = conn.execute("SELECT COUNT(*) as c FROM tokens WHERE status = 'waiting' AND created_at LIKE ?", (f"{today}%",)).fetchone()["c"]
        serving = conn.execute("SELECT COUNT(*) as c FROM tokens WHERE status = 'serving' AND created_at LIKE ?", (f"{today}%",)).fetchone()["c"]
        completed = conn.execute("SELECT COUNT(*) as c FROM tokens WHERE status = 'completed' AND created_at LIKE ?", (f"{today}%",)).fetchone()["c"]
        currently_serving = conn.execute("SELECT token_number, counter_id FROM tokens WHERE status = 'serving' ORDER BY called_at DESC LIMIT 3").fetchall()

    return {
        "waiting": waiting,
        "serving": serving,
        "completed": completed,
        "active_serving": [dict(r) for r in currently_serving]
    }

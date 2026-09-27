"""
scheduler.py — APScheduler job that retrains the model nightly at 01:00.
Imported and started from main.py.
"""

import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _retrain_job():
    """Nightly retraining job — pulls latest DB logs and retrains the model."""
    from backend import model, database  # lazy import to avoid circular deps
    logger.info("🔄  Nightly model retraining started …")
    try:
        extra_rows = database.get_logs_for_training()
        stats = model.train(extra_rows=extra_rows if extra_rows else None)
        logger.info("✅  Retraining done: %s", stats)
    except Exception as exc:
        logger.error("❌  Retraining failed: %s", exc)


def start_scheduler():
    """Start the background scheduler with the nightly retraining job."""
    global _scheduler
    _scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    _scheduler.add_job(
        _retrain_job,
        trigger=CronTrigger(hour=1, minute=0),  # every night at 01:00 IST
        id="nightly_retrain",
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("⏰  Scheduler started — nightly retrain at 01:00 IST")


def stop_scheduler():
    """Gracefully stop the scheduler on app shutdown."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("⏹️  Scheduler stopped")

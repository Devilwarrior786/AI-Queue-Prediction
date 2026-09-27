"""
model.py — ML training and prediction engine for queue wait-time estimation.

Strategy:
  1. Try to load a pre-trained model from disk.
  2. If not available, use the formula-based fallback.
  3. train() reads historical.csv + DB logs, tunes hyperparameters via
     RandomizedSearchCV, trains a GradientBoostingRegressor pipeline,
     and saves the best model to models/queue_model.pkl.
"""

import joblib
import logging
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform

logger = logging.getLogger(__name__)

MODEL_PATH    = Path(__file__).parent.parent / "models" / "queue_model.pkl"
METADATA_PATH = Path(__file__).parent.parent / "models" / "model_metadata.json"
HIST_CSV      = Path(__file__).parent / "data" / "historical.csv"

# All features the model accepts — must match the CSV columns
FEATURES = [
    "hour",
    "minute",
    "day_of_week",
    "week_of_year",
    "month",
    "quarter",
    "is_month_start",
    "is_month_end",
    "is_holiday_adjacent",
    "queue_depth",
    "active_counters",
    "avg_service_time_mins",
]

# Legacy feature set (for older CSV / DB rows that don't have the new columns)
LEGACY_FEATURES = [
    "hour",
    "day_of_week",
    "is_month_start",
    "is_month_end",
    "queue_depth",
    "active_counters",
    "avg_service_time_mins",
]

TARGET = "actual_wait_mins"

# Global model reference (loaded once at startup)
_pipeline: Optional[Pipeline] = None
_feature_set: list[str] = FEATURES  # tracks which feature set the loaded model uses


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model() -> bool:
    """Load the trained model from disk. Returns True if successful."""
    global _pipeline, _feature_set
    if MODEL_PATH.exists():
        try:
            _pipeline = joblib.load(MODEL_PATH)
            # Read feature set from metadata if available
            if METADATA_PATH.exists():
                meta = json.loads(METADATA_PATH.read_text())
                _feature_set = meta.get("features", FEATURES)
            logger.info("Model loaded from %s", MODEL_PATH)
            return True
        except Exception as exc:
            logger.warning("Failed to load model: %s", exc)
    _pipeline = None
    return False


def is_model_ready() -> bool:
    return _pipeline is not None


def predict(
    hour: int,
    day_of_week: int,
    is_month_start: int,
    is_month_end: int,
    queue_depth: int,
    active_counters: int,
    avg_service_time_mins: float,
    minute: int = 0,
    week_of_year: int = 1,
    month: int = 1,
    quarter: int = 1,
    is_holiday_adjacent: int = 0,
) -> tuple[float, str]:
    """
    Predict wait time in minutes.
    Returns (predicted_wait_mins, confidence) where confidence is
    "model" if ML was used, "formula" for fallback.
    """
    if _pipeline is not None:
        # Build feature vector matching the feature set the model was trained on
        full_vals = {
            "hour": hour,
            "minute": minute,
            "day_of_week": day_of_week,
            "week_of_year": week_of_year,
            "month": month,
            "quarter": quarter,
            "is_month_start": is_month_start,
            "is_month_end": is_month_end,
            "is_holiday_adjacent": is_holiday_adjacent,
            "queue_depth": queue_depth,
            "active_counters": active_counters,
            "avg_service_time_mins": avg_service_time_mins,
        }
        X = np.array([[full_vals[f] for f in _feature_set]])
        wait = float(_pipeline.predict(X)[0])
        wait = max(0.5, round(wait, 1))
        return wait, "model"
    else:
        # Formula-based fallback: M/c/∞
        utilisation = queue_depth / max(active_counters, 1)
        wait = utilisation * avg_service_time_mins
        return max(0.5, round(wait, 1)), "formula"


def train(extra_rows: Optional[list[dict]] = None) -> dict:
    """
    Train (or retrain) the model.

    Steps:
      1. Load seed CSV + DB rows, merge and clean.
      2. Determine available features (full or legacy).
      3. RandomizedSearchCV over GBR hyperparameters (5-fold CV).
      4. Evaluate best estimator on held-out test set.
      5. Save pipeline + metadata to disk.

    Returns:
        dict with training stats.
    """
    global _pipeline, _feature_set

    dfs = []

    # --- Load seed CSV ---
    if HIST_CSV.exists():
        df_hist = pd.read_csv(HIST_CSV)
        dfs.append(df_hist)
        logger.info("Loaded %d rows from historical CSV", len(df_hist))

    # --- Append live DB data ---
    if extra_rows:
        df_db = pd.DataFrame(extra_rows)
        dfs.append(df_db)
        logger.info("Appended %d rows from DB logs", len(df_db))

    if not dfs:
        raise ValueError("No training data available. Run scripts/seed_data.py first.")

    df = pd.concat(dfs, ignore_index=True)

    # --- Determine feature set ---
    # Use the full feature set if all columns are present, otherwise fall back
    if all(f in df.columns for f in FEATURES):
        feat_cols = FEATURES
    else:
        feat_cols = [f for f in LEGACY_FEATURES if f in df.columns]
        logger.warning("Some new feature columns missing — using legacy feature set: %s", feat_cols)

    required = feat_cols + [TARGET]
    df = df.dropna(subset=required)

    if len(df) < 50:
        raise ValueError(f"Too few samples ({len(df)}). Need at least 50.")

    logger.info("Training on %d samples with %d features: %s", len(df), len(feat_cols), feat_cols)

    X = df[feat_cols].values
    y = df[TARGET].values

    # --- Train / test split (15 % held out) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # --- Hyperparameter search space ---
    param_dist = {
        "gbr__n_estimators":      [150, 250, 350],
        "gbr__max_depth":         [3, 4, 5],
        "gbr__learning_rate":     [0.05, 0.08, 0.12],
        "gbr__subsample":         [0.75, 0.85, 0.95],
        "gbr__min_samples_split": [5, 10],
    }

    base_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr",    GradientBoostingRegressor(random_state=42)),
    ])

    # Tune hyperparameters efficiently on a representative subsample (up to 15,000 rows)
    tune_size = min(15000, len(X_train))
    sample_indices = np.random.choice(len(X_train), size=tune_size, replace=False)
    X_tune = X_train[sample_indices]
    y_tune = y_train[sample_indices]

    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    logger.info("Tuning hyperparameters on %d rows (8 iterations x 3-fold CV)...", tune_size)
    search = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_dist,
        n_iter=8,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_tune, y_tune)
    best_params = search.best_params_
    logger.info("Best CV MAE : %.3f min", -search.best_score_)
    logger.info("Best parameters found: %s", best_params)

    # Train final model on the ENTIRE 123,744 dataset using best hyperparameters
    logger.info("Training final production pipeline on all %d samples...", len(X_train))
    best_pipeline = search.best_estimator_
    best_pipeline.fit(X_train, y_train)

    # --- Evaluate on held-out test set ---
    y_pred = best_pipeline.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = math.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    logger.info("Test MAE=%.2f min  RMSE=%.2f min  R2=%.4f", mae, rmse, r2)

    # --- Feature importances ---
    gbr_step    = best_pipeline.named_steps["gbr"]
    importances = dict(zip(feat_cols, gbr_step.feature_importances_.tolist()))
    top_features = sorted(importances.items(), key=lambda x: -x[1])
    logger.info("Feature importances: %s", top_features)

    # --- Save model + metadata ---
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    _pipeline  = best_pipeline
    _feature_set = feat_cols

    metadata = {
        "trained_at":        datetime.now().isoformat(timespec="seconds"),
        "n_samples":         len(df),
        "n_train":           len(X_train),
        "n_test":            len(X_test),
        "features":          feat_cols,
        "n_features":        len(feat_cols),
        "best_params":       {k: (int(v) if hasattr(v, 'item') else v)
                              for k, v in search.best_params_.items()},
        "cv_mae_minutes":    round(-search.best_score_, 3),
        "test_mae_minutes":  round(mae, 3),
        "test_rmse_minutes": round(rmse, 3),
        "test_r2":           round(r2, 4),
        "feature_importances": {k: round(v, 4) for k, v in top_features},
        "model_path":        str(MODEL_PATH),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    logger.info("Model + metadata saved.")
    return metadata

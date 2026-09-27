# AI-Queue-Prediction
# AI Queue Prediction & Smart Token System 🏦

> Predicts waiting time at bank/government office queues using machine learning and assigns dynamic time slots with tokens.  
> Runs fully **on-device** on edge systems (e.g. Raspberry Pi) or any standard server — zero cloud dependency.

---

## 🚀 Live Interfaces

When running locally (`uvicorn backend.main:app --host 0.0.0.0 --port 8000`), access:

- 🎫 **Smart Token Kiosk** → [http://localhost:8000/tokens](http://localhost:8000/tokens) (Issue tokens, check estimated slot timing, search ticket progress)
- 🖥️ **Public Live Display** → [http://localhost:8000/](http://localhost:8000/) (Wall-mounted kiosk with active token announcements & particle animation)
- 👷 **Staff Management Desk** → [http://localhost:8000/staff](http://localhost:8000/staff) (Manage windows, call next tokens, log service durations)
- 📖 **Interactive API Docs** → [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate 5-year dataset (~123k records at 5-minute intervals)
python scripts/seed_data.py

# 3. Train ML model with hyperparameter search & cross-validation
python -c "from backend import model, database; database.init_db(); print(model.train())"

# 4. Start the backend server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🍓 Raspberry Pi Deployment

```bash
bash scripts/install.sh
sudo reboot
```

The script will:
1. Install Python, Chromium, and system packages
2. Generate seed data and train the model
3. Register a `systemd` service (`ai-queue`) for auto-start on boot
4. Configure Chromium in kiosk mode for the wall display

---

## 🤖 ML Model Performance (Trained on 123,744 Rows)

- **Algorithm:** `GradientBoostingRegressor` (scikit-learn pipeline with `StandardScaler`)
- **Dataset Size:** 5 years (2019–2024), 123,744 samples, 5-minute granularity
- **Test Set Accuracy ($R^2$):** **0.955** (95.5% variance explained)
- **Mean Absolute Error (MAE):** **1.91 minutes** (~1 min 54s error margin)
- **Hyperparameter Search:** RandomizedSearchCV (evaluating tree depth, subsample ratios, learning rates, estimators)

### Input Features:
| Feature | Type | Description |
|---------|------|-------------|
| `queue_depth` | int | People physically/digitally queued |
| `active_counters` | int | Open service windows |
| `avg_service_time_mins` | float | Rolling average service time |
| `hour` | int | Hour of day (0–23) |
| `minute` | int | Minute in hour (0–59) |
| `day_of_week` | int | Monday=0 ... Sunday=6 |
| `week_of_year` | int | 1–52 |
| `month` | int | 1–12 |
| `quarter` | int | 1–4 |
| `is_month_start` | int | Salary/pension rush (days 1–3) |
| `is_month_end` | int | Bill payment rush (days 28+) |
| `is_holiday_adjacent` | int | Surrounding weekend or major public holiday |

---

## 🎫 Smart Token Kiosk & Dynamic Slot Assignment

- **Token Formats:** Prefix-coded tokens (`C-101` for Cash, `A-102` for Account Desk, `L-101` for Loans, `G-101` for Documentation, `T-101` for General).
- **Assigned Timing:** Computes predicted wait and calculates estimated clock time (e.g., `07:35 PM`).
- **Ticket Tracking:** Self-service lookup shows how many visitors are ahead in line and which counter is calling.
- **Continuous Feedback Loop:** Completing a token automatically logs the actual elapsed service time into SQLite, constantly refining future predictions.

---

## 🔌 API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | Current queue status & predicted wait time |
| `POST` | `/api/queue/update` | Staff update of queue depth & open windows |
| `POST` | `/api/service/log` | Manual logging of service completion duration |
| `POST` | `/api/tokens/issue` | Generates token with AI-estimated slot time |
| `GET` | `/api/tokens/active` | List of active/waiting tokens |
| `GET` | `/api/tokens/summary` | Today's token counts & currently served tokens |
| `GET` | `/api/tokens/{number}` | Check status and queue position for a ticket |
| `PATCH` | `/api/tokens/{number}/status` | Mark token as `serving`, `completed`, or `cancelled` |
| `POST` | `/api/retrain` | Trigger manual retraining |
| `GET` | `/api/health` | Service and model readiness check |

---

## 📂 Project Structure

```
ai-queue/
├── backend/
│   ├── __init__.py
│   ├── main.py          ← FastAPI routes (Queue, Tokens, Pages)
│   ├── model.py         ← ML engine, RandomizedSearchCV & inference
│   ├── database.py      ← SQLite schema, logs & token management
│   ├── schemas.py       ← Pydantic schemas
│   ├── scheduler.py     ← Nightly model retrain scheduler
│   └── data/
│       ├── historical.csv (123k+ observations)
│       └── queue.db       (SQLite store)
├── frontend/
│   ├── tokens.html      ← Smart token dispenser & ticket tracker
│   ├── display.html     ← Animated wall kiosk with token announcements
│   └── staff.html       ← Staff management panel & token desk
├── models/
│   ├── queue_model.pkl       ← Trained scikit-learn pipeline
│   └── model_metadata.json   ← Training metrics, feature importance, params
├── scripts/
│   ├── seed_data.py     ← 5-year synthetic dataset generator
│   └── install.sh       ← Raspberry Pi setup script
├── tests/
│   └── test_api.py      ← End-to-end API & token lifecycle tests
├── requirements.txt
└── README.md
```

---

## ✅ Running the Tests

The API is covered by an end-to-end test suite (health, predictions, validation,
full token lifecycle, daily token-number collision regression, feedback-loop
checks):

```bash
pytest tests/ -v
```

---

## 🔄 How the Feedback Loop Works

1. A token is issued with an AI-estimated wait.
2. When staff **call** the token, the *realised* wait (issue → call) plus the
   live queue state is written to `queue_logs` with `actual_wait_mins`.
3. When the service is **completed**, the service duration feeds the rolling
   average used by future predictions.
4. Nightly retraining (01:00 IST) merges the seed CSV with all realised waits
   from `queue_logs`, so the model keeps adapting to the actual branch.

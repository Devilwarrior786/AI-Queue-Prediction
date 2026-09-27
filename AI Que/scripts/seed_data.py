"""
seed_data.py — Generate realistic synthetic historical data for the queue model.

Usage:
    python scripts/seed_data.py

Outputs:
    backend/data/historical.csv   (5 years of realistic queue observations)
"""

import csv
import random
import math
from datetime import datetime, timedelta, date
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent.parent / "backend" / "data" / "historical.csv"

# Opening hours: 09:00 - 17:00, Monday-Friday (0=Mon ... 4=Fri)
OPEN_HOUR  = 9
CLOSE_HOUR = 17
OPEN_DAYS  = {0, 1, 2, 3, 4}

# 5-year window: Jan 2019 - Jan 2024
START_DATE = datetime(2019, 1, 1, OPEN_HOUR, 0)
END_DATE   = datetime(2024, 1, 1)

# Observation every 5 minutes
INTERVAL_MINUTES = 5

FIELDNAMES = [
    "timestamp",
    "hour",
    "minute",
    "day_of_week",
    "week_of_year",
    "month",
    "quarter",
    "is_month_start",
    "is_month_end",
    "is_holiday_adjacent",   # day before/after a long weekend
    "queue_depth",
    "active_counters",
    "avg_service_time_mins",
    "actual_wait_mins",
]

# ---- Indian public holidays (approximate fixed dates, not regional) ----
HOLIDAYS = set()
for year in range(2019, 2025):
    HOLIDAYS.update([
        date(year,  1, 26),   # Republic Day
        date(year,  8, 15),   # Independence Day
        date(year, 10,  2),   # Gandhi Jayanti
        date(year, 11, 14),   # Children's Day
        date(year, 12, 25),   # Christmas
    ])


def is_holiday_adjacent(d: date) -> bool:
    """True if the working day is next to a holiday / weekend."""
    prev_day = d - timedelta(days=1)
    next_day = d + timedelta(days=1)
    return (prev_day in HOLIDAYS or next_day in HOLIDAYS or
            prev_day.weekday() >= 5 or next_day.weekday() >= 5)


def seasonal_factor(month: int) -> float:
    """Banks are busier in certain months (fiscal year end, tax season)."""
    busy = {3: 1.35, 4: 1.20, 12: 1.15, 1: 1.10}
    return busy.get(month, 1.0)


def base_queue(hour: int, minute: int, day_of_week: int,
               is_month_start: bool, is_month_end: bool,
               is_holiday_adj: bool, month: int) -> float:
    """
    Model expected queue depth with multiple overlapping Gaussian peaks.
    """
    t = hour + minute / 60.0  # continuous time (e.g. 10.5 = 10:30)

    # Intra-day pattern
    intra = (
        9.0 * math.exp(-0.5 * ((t - 9.6)  / 1.2) ** 2)   # opening rush
        + 6.0 * math.exp(-0.5 * ((t - 11.5) / 1.4) ** 2)  # pre-lunch
        + 4.0 * math.exp(-0.5 * ((t - 14.5) / 1.8) ** 2)  # post-lunch
        + 3.0 * math.exp(-0.5 * ((t - 16.3) / 0.8) ** 2)  # end-of-day rush
    )

    # Weekly: Mondays busier (pending weekend transactions), Fridays busier
    weekly = {0: 1.35, 1: 1.05, 2: 1.00, 3: 1.05, 4: 1.25}.get(day_of_week, 1.0)

    # Monthly cycles
    rush = 1.0
    if is_month_start:  rush = 1.65   # salary, pension, government grants
    elif is_month_end:  rush = 1.40   # bill payments, loan EMIs
    if is_holiday_adj:  rush *= 1.20  # people rush before/after long weekends

    season = seasonal_factor(month)

    return max(0.0, intra * weekly * rush * season)


def simulate_wait(queue_depth: int, active_counters: int, avg_service: float) -> float:
    """M/c/infinity queuing formula with realistic noise."""
    if active_counters == 0:
        return 99.0
    utilisation = queue_depth / (active_counters * (60 / avg_service))
    # Add a slight queueing delay when utilisation is high
    delay_factor = 1.0 + max(0, utilisation - 0.7) * 0.8
    base = (queue_depth / active_counters) * avg_service * delay_factor
    # Noise proportional to load (busier = more variance)
    noise_std = base * 0.12 + 0.4
    return max(0.5, round(base + random.gauss(0, noise_std), 1))


def yearly_drift(year: int) -> float:
    """Simulate mild year-on-year growth in footfall (~5% per year)."""
    return 1.0 + (year - 2019) * 0.05


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    rows = []
    current = START_DATE
    total_days = 0

    print("Generating dataset...")

    while current < END_DATE:
        dow = current.weekday()
        d   = current.date()

        # Skip weekends and public holidays
        if dow not in OPEN_DAYS or d in HOLIDAYS:
            current += timedelta(minutes=INTERVAL_MINUTES)
            continue

        if OPEN_HOUR <= current.hour < CLOSE_HOUR:
            is_ms       = current.day <= 3
            is_me       = current.day >= 28
            is_hol_adj  = is_holiday_adjacent(d)
            month       = current.month
            quarter     = (month - 1) // 3 + 1
            week        = current.isocalendar()[1]
            drift       = yearly_drift(current.year)

            # Max counters varies by time of day and day
            if current.hour < 10 or current.hour >= 16:
                max_c = 3
            elif dow in {0, 4}:     # Mon / Fri
                max_c = 5
            else:
                max_c = 4
            active_counters = random.randint(max(1, max_c - 1), max_c)

            avg_service = max(1.0, round(random.gauss(4.5, 0.9), 1))

            expected = base_queue(
                current.hour, current.minute, dow,
                is_ms, is_me, is_hol_adj, month
            ) * drift

            queue = max(0, int(random.gauss(expected, expected * 0.22 + 1)))
            wait  = simulate_wait(queue, active_counters, avg_service)

            rows.append({
                "timestamp":            current.strftime("%Y-%m-%d %H:%M:%S"),
                "hour":                 current.hour,
                "minute":               current.minute,
                "day_of_week":          dow,
                "week_of_year":         week,
                "month":                month,
                "quarter":              quarter,
                "is_month_start":       int(is_ms),
                "is_month_end":         int(is_me),
                "is_holiday_adjacent":  int(is_hol_adj),
                "queue_depth":          queue,
                "active_counters":      active_counters,
                "avg_service_time_mins":avg_service,
                "actual_wait_mins":     wait,
            })

        # Advance interval
        current += timedelta(minutes=INTERVAL_MINUTES)

        # Progress every 6 months
        if current.day == 1 and current.month in {1, 7}:
            total_days += 1
            print(f"  ... {current.strftime('%Y-%m')}  ({len(rows):,} rows so far)")

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Generated {len(rows):,} rows -> {OUTPUT_PATH}")
    print(f"     Date range : {START_DATE.date()} to {END_DATE.date()}")
    print(f"     Interval   : every {INTERVAL_MINUTES} minutes")
    print(f"     Features   : {len(FIELDNAMES)} columns")


if __name__ == "__main__":
    main()

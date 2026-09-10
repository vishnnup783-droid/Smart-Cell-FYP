"""
SmartCell — terminal demo for the classical SOH/RUL estimator.

Edit the values in the INPUT section below to whatever sensor readings you
have (by hand today, from the ESP32 in a later stage), then run:

    python scripts/predict_demo.py

Prints today's predicted SOH and the predicted RUL to the terminal. Uses the
same estimator functions in classical_soh_rul.py that would run on the real
device — no GUI, no hardware needed.
"""

from datetime import datetime, timedelta

from classical_soh_rul import (
    SohReading,
    classify_soh,
    estimate_rul,
    internal_resistance_ohms,
    soh_from_resistance,
)

# ============================== INPUT ======================================
# Today's cranking-test reading (replace with real sensor values later).
V_REST = 12.0
V_MIN = 10.4
I_CRANK = 35
TEMP_C = 25

HISTORY = [
    (60, 75.0),
    (45, 68.0),
    (30, 62.0),
    (15, 58.0),
    (5, 56.0),
]
# ============================================================================


def main():
    r_ohms = internal_resistance_ohms(V_REST, V_MIN, I_CRANK)
    soh_today = soh_from_resistance(r_ohms, TEMP_C)
    soh_class = classify_soh(soh_today)

    print("=== Today's reading ===")
    print(f"  Resting voltage:   {V_REST} V")
    print(f"  Min crank voltage: {V_MIN} V")
    print(f"  Cranking current:  {I_CRANK} A")
    print(f"  Temperature:       {TEMP_C} C")
    print(f"  -> Internal resistance: {r_ohms * 1000:.1f} mOhm")
    print(f"  -> Predicted SOH:       {soh_today:.1f}%  ({soh_class})")

    now = datetime.now()
    all_readings = HISTORY + [(0, soh_today)]
    history = sorted(
        [SohReading(now - timedelta(days=d), s) for d, s in all_readings],
        key=lambda reading: reading.timestamp,
    )

    days_remaining, rul_class = estimate_rul(history)

    print("\n=== RUL prediction (trend across history + today) ===")
    for d, s in sorted(all_readings, reverse=True):
        print(f"  {d:>3} days ago: {s:.1f}% SOH")
    if days_remaining is None:
        print(f"  -> RUL: not forecastable ({rul_class})")
    else:
        print(f"  -> Predicted RUL: {days_remaining:.0f} days  ({rul_class})")


if __name__ == "__main__":
    main()

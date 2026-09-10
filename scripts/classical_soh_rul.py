"""
Classical (non-ML) SOH/RUL estimator for the 12V lead-acid battery on the
SmartCell two-wheeler hardware. Per the Revised Directions doc, this replaces
the ML/LSTM layer with a transparent method built on the same ESP32-sensed
signals: voltage, current, temperature.

Two independent measurements feed the estimate:
  1. Resting open-circuit voltage (OCV) -> State of Charge (SOC), via a
     temperature-compensated lookup table.
  2. Cranking-event voltage sag -> internal resistance -> State of Health
     (SOH), compared against a calibrated new-battery baseline.

RUL is a linear trend fit through the SOH history, extrapolated to the EOL
threshold.

CALIBRATE BEFORE DEPLOYMENT: OCV_SOC_TABLE, R_NEW_OHMS and the temperature
coefficients below are standard textbook approximations for a small 12V
lead-acid battery, not measurements of your actual battery. Replace them with
values from your battery's datasheet or bench calibration against a
known-good unit (see the calibration steps at the bottom of this file).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

# ---- Calibration constants (replace with datasheet / bench-calibrated values) ----

# Resting OCV (V) -> SOC (%) at 25 C, standard 12V lead-acid discharge curve.
OCV_SOC_TABLE = [
    (10.50, 0), (11.31, 10), (11.58, 20), (11.75, 30), (11.90, 40),
    (12.06, 50), (12.20, 60), (12.32, 70), (12.42, 80), (12.50, 90), (12.70, 100),
]

TEMP_COEFF_V_PER_C = 0.024   # OCV drops ~24 mV per degree C below 25 C for a given SOC (approx)
REFERENCE_TEMP_C = 25.0

R_NEW_OHMS = 0.025           # internal resistance of a new battery of this spec, at 25 C (approx)
R_TEMP_COEFF_PER_C = 0.02    # internal resistance rises ~2%/degree C below 25 C (approx)

SOH_BINS = [(0, 65, "Critical"), (65, 80, "Degrading"), (80, 101, "Healthy")]
EOL_SOH_PCT = 70.0            # matches the NASA-baseline convention (Saha & Goebel EOL threshold)

RUL_BINS_DAYS = [(0, 30, "Short"), (30, 120, "Medium"), (120, float("inf"), "Long")]


def soc_from_ocv(ocv_v: float, temp_c: float) -> float:
    """Resting open-circuit voltage -> state of charge (%), temperature-compensated."""
    compensated = ocv_v + TEMP_COEFF_V_PER_C * (REFERENCE_TEMP_C - temp_c)
    xs = [v for v, _ in OCV_SOC_TABLE]
    if compensated <= xs[0]:
        return 0.0
    if compensated >= xs[-1]:
        return 100.0
    for (v0, s0), (v1, s1) in zip(OCV_SOC_TABLE, OCV_SOC_TABLE[1:]):
        if v0 <= compensated <= v1:
            frac = (compensated - v0) / (v1 - v0)
            return s0 + frac * (s1 - s0)
    return 100.0  # unreachable


def internal_resistance_ohms(v_rest: float, v_min_during_crank: float, i_crank_a: float) -> float:
    """Internal resistance from a cranking-event voltage sag. i_crank_a must be > 0."""
    if i_crank_a <= 0:
        raise ValueError("i_crank_a must be a positive cranking current")
    return (v_rest - v_min_during_crank) / i_crank_a


def soh_from_resistance(r_ohms: float, temp_c: float) -> float:
    """Internal resistance -> SOH (%), temperature-compensated against a new-battery baseline."""
    r_new_at_temp = R_NEW_OHMS * (1 + R_TEMP_COEFF_PER_C * (REFERENCE_TEMP_C - temp_c))
    soh = 100.0 * (r_new_at_temp / r_ohms)
    return max(0.0, min(100.0, soh))


def classify_soh(soh_pct: float) -> str:
    for lo, hi, label in SOH_BINS:
        if lo <= soh_pct < hi:
            return label
    return SOH_BINS[-1][2]


@dataclass
class SohReading:
    timestamp: datetime
    soh_pct: float


def estimate_rul(history: list[SohReading]) -> tuple[float | None, str]:
    """Fit a linear trend to SOH-vs-time and extrapolate to the EOL threshold.

    Returns (days_remaining, rul_class). days_remaining is None if there are
    fewer than 2 points or the trend is flat/improving (no forecastable EOL).
    """
    if len(history) < 2:
        return None, "Unknown"

    t0 = history[0].timestamp
    xs = [(r.timestamp - t0).total_seconds() / 86400.0 for r in history]  # days since first reading
    ys = [r.soh_pct for r in history]

    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var = sum((x - mean_x) ** 2 for x in xs)
    if var == 0:
        return None, "Unknown"
    slope = cov / var           # % SOH per day
    intercept = mean_y - slope * mean_x

    if slope >= 0:
        return None, "Long"     # flat or improving trend, no forecastable EOL

    days_to_eol_from_t0 = (EOL_SOH_PCT - intercept) / slope
    days_remaining = max(0.0, days_to_eol_from_t0 - xs[-1])

    for lo, hi, label in RUL_BINS_DAYS:
        if lo <= days_remaining < hi:
            return days_remaining, label
    return days_remaining, RUL_BINS_DAYS[-1][2]


def demo():
    """Runnable self-check: sanity-checks each function against known reference points."""
    assert soc_from_ocv(12.70, 25.0) == 100.0
    assert soc_from_ocv(10.00, 25.0) == 0.0
    assert abs(soc_from_ocv(12.06, 25.0) - 50.0) < 1e-6
    # a colder battery reads a lower raw OCV for the same real SOC; compensation should cancel that out
    assert abs(soc_from_ocv(12.06 - TEMP_COEFF_V_PER_C * 10, 15.0) - soc_from_ocv(12.06, 25.0)) < 1e-6

    r_healthy = internal_resistance_ohms(v_rest=12.6, v_min_during_crank=12.2, i_crank_a=100)
    assert abs(r_healthy - 0.004) < 1e-9
    assert abs(soh_from_resistance(R_NEW_OHMS, 25.0) - 100.0) < 1e-6
    assert abs(soh_from_resistance(R_NEW_OHMS * 2, 25.0) - 50.0) < 1e-6
    assert classify_soh(100.0) == "Healthy"
    assert classify_soh(70.0) == "Degrading"
    assert classify_soh(40.0) == "Critical"

    # synthetic linearly-degrading history: 90% at day 0, -1%/day, 10 points (day 0..9, last reading 81%)
    start = datetime(2026, 1, 1)
    history = [SohReading(start + timedelta(days=d), 90.0 - d) for d in range(10)]
    days, rul_class = estimate_rul(history)
    # crosses the 70% EOL threshold at day 20 from t0; last reading is at day 9 -> 11 days remaining
    assert days is not None and abs(days - 11.0) < 1e-6
    assert rul_class == "Short"

    print("All classical SOH/RUL estimator checks passed.")


if __name__ == "__main__":
    demo()


# ---- Calibration steps (do these before trusting the numbers above) ----
#
# 1. OCV_SOC_TABLE: either take it from your battery's datasheet, or measure it
#    yourself: fully charge the battery, let it rest 2+ hours (no load), record
#    OCV, then discharge in known SOC steps (e.g. via a controlled load and
#    Coulomb counting) and record resting OCV at each step.
#
# 2. R_NEW_OHMS: measure on a brand-new unit of the same battery spec, using
#    the same cranking-event method the ESP32 will use in the field (apply a
#    known load current, e.g. the starter motor, and record the voltage sag).
#    If you have the battery's Cold Cranking Amps (CCA) rating instead, CCA
#    tests are already a standardized internal-resistance proxy -- a higher
#    CCA rating means lower resistance.
#
# 3. Cranking-event detector (ESP32 firmware): define what counts as a
#    cranking event (e.g. current > 20 A sustained for > 200 ms), and capture
#    v_rest (voltage just before the event), v_min (minimum voltage during
#    it), and i_crank (average or peak current during it).
#
# 4. Rest-state detector: OCV is only meaningful after the battery has been
#    unloaded for a while (e.g. ignition off for 30+ minutes) -- track time
#    since last significant load before trusting a voltage reading as OCV.
#
# 5. Once real cranking-event and rest-OCV data is being logged from the
#    hardware, re-derive R_NEW_OHMS, the temperature coefficients, and the
#    SOH/RUL bin edges from that data instead of the approximations here.

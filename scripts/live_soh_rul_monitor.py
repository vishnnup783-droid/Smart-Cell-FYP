"""
SmartCell - live SOH/RUL estimator for the real ESP32 + INA219 hardware.

This is the classical estimator recommended in the Revised Directions
document (no ML): Coulomb counting for state of charge, periodic OCV
recalibration to correct integration drift, internal-resistance trend for
SOH, and a linear trend projection for RUL. It reads the CSV stream the
ESP32 firmware (firmware/esp32_battery_monitor/) sends over serial.

Usage:
    python3 live_soh_rul_monitor.py --port /dev/ttyUSB0 --capacity-ah 5.0
    python3 live_soh_rul_monitor.py --simulate                # no hardware needed, sanity-check the math
    python3 live_soh_rul_monitor.py --test                    # runs the self-check and exits

Every sample is appended to outputs/live_battery_log.csv so you have raw
data for your report even before enough history exists for SOH/RUL.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

from classical_soh_rul import (
    REFERENCE_TEMP_C,
    SohReading,
    classify_soh,
    estimate_rul,
    soc_from_ocv,
    soh_from_resistance,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

# ---------------------------------------------------------------------------
# Config - the numbers you'll actually need to change for your battery
# ---------------------------------------------------------------------------
CURRENT_SIGN = 1          # flip to -1 if your INA219 wiring reports discharge as negative
REST_CURRENT_A = 0.05     # |current| below this counts as "at rest" for OCV reading
REST_SETTLE_S = 120       # must stay at rest this long before OCV is trusted (surface charge decay)
STEP_CURRENT_A = 0.5      # a jump bigger than this between samples triggers a resistance measurement

# This board has no temperature sensor wired up yet, so OCV/resistance readings
# are treated as if they were taken at the reference temperature (no compensation).
# Once a temperature sensor is added, pass its reading through instead.
ASSUMED_TEMP_C = REFERENCE_TEMP_C


class CoulombCounter:
    """Tracks remaining charge (Ah) by integrating current, with OCV recalibration."""

    def __init__(self, rated_capacity_ah: float, initial_soc_pct: float = 100.0):
        self.rated_capacity_ah = rated_capacity_ah
        self.charge_ah = rated_capacity_ah * initial_soc_pct / 100.0
        self.rest_start_t = None

    def update(self, t: float, dt_hours: float, current_a: float):
        # positive current = discharging (see CURRENT_SIGN), so it subtracts charge
        self.charge_ah -= current_a * dt_hours
        self.charge_ah = max(0.0, min(self.rated_capacity_ah, self.charge_ah))

    def maybe_recalibrate(self, t: float, current_a: float, voltage: float) -> bool:
        """If the battery has been at rest long enough, trust OCV over the drifted integral."""
        if abs(current_a) > REST_CURRENT_A:
            self.rest_start_t = None
            return False
        if self.rest_start_t is None:
            self.rest_start_t = t
        if t - self.rest_start_t < REST_SETTLE_S:
            return False
        soc_ocv = soc_from_ocv(voltage, ASSUMED_TEMP_C)
        self.charge_ah = self.rated_capacity_ah * soc_ocv / 100.0
        return True

    @property
    def soc_pct(self) -> float:
        return 100.0 * self.charge_ah / self.rated_capacity_ah


class ResistanceTracker:
    """Detects load steps and estimates internal resistance from the resulting V/I jump."""

    def __init__(self):
        self.prev_v = None
        self.prev_i = None
        self.history = []  # list of (t_seconds, r_ohms)
        self.baseline_r = None

    def update(self, t: float, voltage: float, current: float):
        r = None
        if self.prev_v is not None:
            d_i = current - self.prev_i
            if abs(d_i) >= STEP_CURRENT_A:
                d_v = voltage - self.prev_v
                r = abs(d_v / d_i)
                self.history.append((t, r))
                if self.baseline_r is None:
                    self.baseline_r = r  # first measured healthy-battery resistance becomes the reference
        self.prev_v, self.prev_i = voltage, current
        return r

    @property
    def latest_r(self):
        return self.history[-1][1] if self.history else None

    def soh_pct(self):
        if self.baseline_r is None or self.latest_r is None:
            return None
        # this device has no factory-calibrated baseline, so the first measured
        # resistance stands in for it -- see soh_from_resistance's r_new_ohms param.
        return soh_from_resistance(self.latest_r, ASSUMED_TEMP_C, r_new_ohms=self.baseline_r)


def rul_days_from_trend(soh_history: list):
    """Linear-fit SOH(%) vs time(days) and project forward to EOL_SOH_PCT. None if not enough/no decline."""
    if len(soh_history) < 5:
        return None
    readings = [SohReading(datetime.fromtimestamp(t), s) for t, s in soh_history]
    days_remaining, _rul_class = estimate_rul(readings)
    return days_remaining


# ---------------------------------------------------------------------------
# Main loop - shared by real serial input and --simulate
# ---------------------------------------------------------------------------
def run(sample_source, rated_capacity_ah: float, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    coulomb = CoulombCounter(rated_capacity_ah)
    resistance = ResistanceTracker()
    soh_history = []

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["wall_time", "device_ms", "voltage_v", "current_a", "soc_pct", "soh_pct", "soh_class", "rul_days"])

        prev_t = None
        for wall_t, device_ms, voltage, current in sample_source:
            current *= CURRENT_SIGN
            dt_hours = 0.0 if prev_t is None else max(0.0, (wall_t - prev_t) / 3600.0)
            prev_t = wall_t

            coulomb.update(wall_t, dt_hours, current)
            recalibrated = coulomb.maybe_recalibrate(wall_t, current, voltage)
            resistance.update(wall_t, voltage, current)

            soh = resistance.soh_pct()
            if soh is not None:
                soh_history.append((wall_t, soh))
            rul = rul_days_from_trend(soh_history)

            row = [
                f"{wall_t:.2f}", device_ms, f"{voltage:.3f}", f"{current:.4f}",
                f"{coulomb.soc_pct:.1f}",
                f"{soh:.1f}" if soh is not None else "",
                classify_soh(soh) if soh is not None else "",
                f"{rul:.1f}" if rul is not None else "",
            ]
            writer.writerow(row)
            f.flush()

            status = f"V={voltage:6.2f}V  I={current:6.2f}A  SOC={coulomb.soc_pct:5.1f}%"
            if soh is not None:
                status += f"  SOH={soh:5.1f}% ({classify_soh(soh)})"
            if rul is not None:
                status += f"  RUL~{rul:.0f}d"
            if recalibrated:
                status += "  [OCV recalibrated]"
            print(status)


def serial_source(port: str, baud: int):
    import serial  # local import: not needed for --simulate/--test

    ser = serial.Serial(port, baud, timeout=5)
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line or line.startswith("millis") or line.startswith("ERROR"):
            continue
        try:
            device_ms, v, i = line.split(",")
            yield time.time(), int(device_ms), float(v), float(i)
        except ValueError:
            continue  # malformed line, skip


def simulate_source():
    """Synthetic stream: a battery discharging with a few load steps and slowly rising
    internal resistance, compressed into seconds so you can see the estimator work without
    hardware. Not a substitute for real data - just exercises the same code path."""
    import random

    t = time.time()
    ocv = 12.6
    r_true = 0.05  # ohms, rises over the run to simulate degradation
    for step in range(200):
        # every ~15 samples, flip the load between "resting" and "under load"
        under_load = (step // 15) % 2 == 1
        current = (2.0 + random.uniform(-0.05, 0.05)) if under_load else 0.0
        r_true += 0.0003  # slow degradation
        voltage = ocv - current * r_true + random.uniform(-0.01, 0.01)
        ocv -= 0.0005  # slow discharge of open-circuit voltage
        yield t, step * 1000, voltage, current
        t += 5  # 5 simulated seconds per sample, so 200 samples ~ 16.7 min of "battery time"


def self_test():
    """One runnable check: feeds simulate_source() through the real estimator and asserts
    the outputs behave sanely. Run with --test."""
    coulomb = CoulombCounter(rated_capacity_ah=5.0)
    resistance = ResistanceTracker()
    soh_history = []
    prev_t = None
    soc_values = []

    for wall_t, device_ms, voltage, current in simulate_source():
        dt_hours = 0.0 if prev_t is None else (wall_t - prev_t) / 3600.0
        prev_t = wall_t
        coulomb.update(wall_t, dt_hours, current)
        coulomb.maybe_recalibrate(wall_t, current, voltage)
        resistance.update(wall_t, voltage, current)
        soc_values.append(coulomb.soc_pct)
        soh = resistance.soh_pct()
        if soh is not None:
            soh_history.append((wall_t, soh))

    assert soc_values[-1] < soc_values[0], "SOC should drop over a net-discharging run"
    assert len(resistance.history) > 5, "should have detected multiple load-step resistance measurements"
    assert resistance.latest_r > resistance.baseline_r, "resistance should have risen (simulated degradation)"
    assert soh_history[-1][1] < soh_history[0][1], "SOH should decline as resistance rises"
    rul = rul_days_from_trend(soh_history)
    assert rul is not None and rul >= 0, "declining SOH trend should produce a finite RUL projection"

    assert soc_from_ocv(12.7, ASSUMED_TEMP_C) == 100.0 and soc_from_ocv(10.5, ASSUMED_TEMP_C) == 0.0
    assert 0 < soc_from_ocv(12.3, ASSUMED_TEMP_C) < 100

    print("self_test: OK")
    print(f"  SOC {soc_values[0]:.1f}% -> {soc_values[-1]:.1f}%")
    print(f"  R {resistance.baseline_r:.4f} ohm -> {resistance.latest_r:.4f} ohm")
    print(f"  SOH {soh_history[0][1]:.1f}% -> {soh_history[-1][1]:.1f}%")
    print(f"  projected RUL (compressed sim time): {rul:.2f} days")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyUSB0", help="serial port the ESP32 is on")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--capacity-ah", type=float, default=5.0, help="rated capacity of YOUR battery in Ah")
    ap.add_argument("--simulate", action="store_true", help="use synthetic data instead of a real serial port")
    ap.add_argument("--test", action="store_true", help="run the self-check and exit")
    args = ap.parse_args()

    if args.test:
        self_test()
        return

    log_path = OUT_DIR / "live_battery_log.csv"
    source = simulate_source() if args.simulate else serial_source(args.port, args.baud)
    try:
        run(source, args.capacity_ah, log_path)
    except KeyboardInterrupt:
        print("\nStopped. Log saved to", log_path)
        sys.exit(0)


if __name__ == "__main__":
    main()

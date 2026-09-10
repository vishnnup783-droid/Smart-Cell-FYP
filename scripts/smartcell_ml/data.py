"""Load NASA .mat files, extract per-discharge-cycle features, and build SOH/RUL labels."""

import numpy as np
import pandas as pd
import scipy.io as sio

from . import config


def extract_battery_frame(battery_id: str) -> pd.DataFrame:
    """Parse one NASA .mat file into a per-discharge-cycle feature table."""
    raw = sio.loadmat(config.DATA_DIR / f"{battery_id}.mat", simplify_cells=True)
    cycles = raw[battery_id]["cycle"]

    rows = []
    discharge_index = 0
    for cyc in cycles:
        if cyc["type"] != "discharge":
            continue
        d = cyc["data"]
        t = np.asarray(d["Time"], dtype=float)
        v = np.asarray(d["Voltage_measured"], dtype=float)
        i = np.asarray(d["Current_measured"], dtype=float)
        temp = np.asarray(d["Temperature_measured"], dtype=float)
        capacity = float(d["Capacity"])

        if len(t) < 3:
            continue  # malformed/truncated cycle, skip

        duration = t[-1] - t[0]
        trapz_fn = getattr(np, "trapezoid", None) or np.trapz
        energy_wh = trapz_fn(v * np.abs(i), t) / 3600.0  # current is negative during discharge

        # "time to reach 3.0V" - classic health-indicator feature: how long the
        # cell holds voltage above a fixed cutoff before it starts to sag hard.
        below_3v = np.where(v <= 3.0)[0]
        time_to_3v = t[below_3v[0]] - t[0] if len(below_3v) > 0 else duration

        voltage_drop_rate = (v[0] - v[-1]) / duration if duration > 0 else np.nan

        rows.append(
            {
                "battery_id": battery_id,
                "discharge_cycle": discharge_index,  # 0-indexed order among discharge cycles
                "ambient_temperature": cyc.get("ambient_temperature", np.nan),
                "duration_s": duration,
                "mean_voltage": v.mean(),
                "std_voltage": v.std(),
                "min_voltage": v.min(),
                "max_voltage": v.max(),
                "voltage_drop_rate": voltage_drop_rate,
                "mean_current": i.mean(),
                "std_current": i.std(),
                "mean_temperature": temp.mean(),
                "max_temperature": temp.max(),
                "std_temperature": temp.std(),
                "energy_wh": energy_wh,
                "time_to_3v_s": time_to_3v,
                "capacity_ah": capacity,
            }
        )
        discharge_index += 1

    return pd.DataFrame(rows)


def _add_soh_labels(df: pd.DataFrame) -> None:
    df["soh_pct"] = 100.0 * df["capacity_ah"] / config.RATED_CAPACITY_AH
    df["soh_class"] = pd.cut(df["soh_pct"], bins=config.SOH_BINS, labels=config.SOH_LABELS, right=False)


def _add_rul_labels(df: pd.DataFrame) -> None:
    """RUL = cycles remaining until capacity crosses the EOL threshold, computed per battery."""
    eol_capacity = config.EOL_FRACTION * config.RATED_CAPACITY_AH
    rul_values = np.zeros(len(df))
    for b in config.BATTERIES:
        sub = df.loc[df["battery_id"] == b].sort_values("discharge_cycle")
        below_eol = sub.index[sub["capacity_ah"] <= eol_capacity]
        eol_idx = (
            sub.loc[below_eol[0], "discharge_cycle"]
            if len(below_eol) > 0
            else sub["discharge_cycle"].max()  # never reached EOL in the log
        )
        rul = (eol_idx - sub["discharge_cycle"]).clip(lower=0)
        rul_values[sub.index] = rul.values
    df["rul_cycles"] = rul_values

    # 3-class label from pooled tertiles - documented, reproducible thresholds
    q1, q2 = df["rul_cycles"].quantile([1 / 3, 2 / 3])
    rul_bins = [-1, q1, q2, df["rul_cycles"].max() + 1]
    df["rul_class"] = pd.cut(df["rul_cycles"], bins=rul_bins, labels=config.RUL_LABELS)
    df.attrs["rul_bin_edges"] = [float(q1), float(q2)]


def build_dataset() -> pd.DataFrame:
    """Load all batteries and attach SOH/RUL class labels. Entry point for this module."""
    df = pd.concat([extract_battery_frame(b) for b in config.BATTERIES], ignore_index=True)
    _add_soh_labels(df)
    _add_rul_labels(df)
    return df

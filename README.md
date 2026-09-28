# SmartCell — IoT-Based Predictive Battery Health Monitoring

Final Year Project. SmartCell continuously monitors the 12V lead-acid battery on a
petrol two-wheeler (motorcycle/scooter), estimates its **State of Health (SOH)**
and **Remaining Useful Life (RUL)**, and reports both to a rider-facing app over
BLE. The goal: catch degradation in the ~85% of a battery's life *before* it
becomes noticeable to the rider as dim headlights or a weak horn.

## Why this project looks the way it does

After Review 0, faculty feedback was: (1) an ML/LSTM prediction layer is
unnecessary complexity for this problem, (2) the project needed more technical
depth, (3) reframe around sustainability/SDGs. This repo reflects that pivot
(full reasoning in `SmartCell_Revised_Directions.pdf`):

- The **deployed** SOH/RUL algorithm is classical and explainable — internal
  resistance trend, Coulomb counting, OCV lookup, threshold classification.
  No neural network runs on the device.
- A **separate, one-time proof-of-concept** (`scripts/smartcell_ml/`) trains ML
  classifiers on a public Li-ion dataset purely to justify that decision: it
  shows the signal that actually drives good predictions (discharge-curve
  shape, time-to-cutoff-voltage) is exactly what the classical estimator
  already uses — so dropping ML costs no accuracy for this problem.

## Repository structure

```
FYP/
├── SmartCell_Revised_Directions.pdf         Faculty-feedback pivot rationale
├── Final_Year_Project_Literature_Survey_Guide.docx   Lit-survey methodology & template
├── data/nasa_mat/                           Raw NASA PCoE Li-ion dataset (B0005/6/7/18.mat)
├── firmware/esp32_battery_monitor/          ESP32 + INA219 voltage/current logger (Arduino sketch)
├── scripts/
│   ├── classical_soh_rul.py                 THE deployed estimator: OCV->SOC, R->SOH, trend->RUL
│   ├── live_soh_rul_monitor.py              Runs the estimator against real (or simulated) serial data
│   ├── predict_demo.py                      Terminal demo: edit input values, see SOH/RUL printed
│   ├── soh_rul_pipeline.py                  Entry point for the NASA ML proof-of-concept
│   └── smartcell_ml/                        ML pipeline: config, data loading, training, plots
├── outputs/                                 Generated: engineered_features.csv, results.json,
│                                             results_summary.md, plots/
└── requirements.txt
```

## Part 1 — Dataset and features (ML proof-of-concept)

Since no public dataset exists for 12V lead-acid batteries in two-wheeler use,
the ML proof-of-concept uses the **NASA PCoE Li-ion Battery Data Set** (Saha &
Goebel, 2007) — room-temperature cycling-to-failure data for 4 cells (B0005,
B0006, B0007, B0018), rated at 2.0 Ah. Only **discharge** cycles are used:
**636 discharge cycles** total (168+168+168+132).

Each discharge cycle is collapsed from its raw voltage/current/temperature/time
trace into a **14-feature vector** (`scripts/smartcell_ml/config.py`):

| Feature | Why it's included |
|---|---|
| `mean/std/min/max_voltage`, `voltage_drop_rate` | Discharge-curve shape degrades characteristically with age |
| `energy_wh` = ∫ V·\|I\| dt / 3600 | Directly tracks capacity fade |
| `time_to_3v_s` | Classic health indicator — healthy cells hold voltage longer before sagging |
| `mean/max/std_temperature`, `ambient_temperature` | Aged cells run hotter under load |
| `mean/std_current`, `duration_s` | Discharge consistency/energy delivered |

These are deliberately signals an ESP32 can already sense (voltage, current,
temperature over time) — nothing exotic or dataset-specific.

**Labels:**
- **SOH%** = 100 × measured discharge capacity ÷ 2.0 Ah rated capacity, binned
  into Critical (<65%) / Degrading (65–80%) / Healthy (≥80%).
- **RUL** = discharge cycles remaining until capacity crosses **70% of rated**
  (standard EOL threshold from Saha & Goebel), binned into Short/Medium/Long by
  pooled tertiles.

## Part 2 — Mathematics

### 2a. ML proof-of-concept (validates the *methodology*, not deployed)

Three classifiers (`scripts/smartcell_ml/config.py`) trained under
**Leave-One-Battery-Out cross-validation** — each battery held out as the test
set in turn, so every prediction is on an unseen battery:

- Logistic Regression (scaled features)
- Decision Tree (max depth 5)
- Random Forest (300 trees, max depth 8)

| Target | Best model | Accuracy | F1 (weighted) |
|---|---|---|---|
| SOH (3-class) | Random Forest | 0.96 | 0.96 |
| RUL (3-class) | Random Forest | 0.85 | 0.85 |

Feature importance ranks `time_to_3v_s` and `energy_wh` highest for both
targets — the same signal the classical estimator below is built on.

### 2b. Classical estimator (the deployed algorithm) — `scripts/classical_soh_rul.py`

**State of Charge**, from resting open-circuit voltage, temperature-compensated:

```
V_compensated = V_ocv + k_T · (25 − T)
SOC(%) = piecewise-linear interpolation of V_compensated
         over the standard 12V lead-acid OCV table
```

**Internal resistance**, from a cranking-event voltage sag (Ohm's law applied
to the *change* caused by the load):

```
R = (V_rest − V_min) / I_crank
```

**State of Health**, comparing measured resistance to a temperature-compensated
new-battery baseline:

```
R_new(T) = R_new · (1 + k_R · (25 − T))
SOH(%) = 100 × R_new(T) / R,  clamped to [0, 100]
```

**Remaining Useful Life**, ordinary-least-squares linear regression of the SOH
history against time, extrapolated to the end-of-life threshold:

```
fit:  SOH(t) = slope · t + intercept        (t = days since first reading)
RUL (days) = (SOH_EOL − intercept) / slope − t_latest
```
where `SOH_EOL = 70%` (same convention as the ML baseline). A non-negative
slope (flat/improving trend) means no forecastable EOL yet.

Both SOH and RUL are then bucketed by simple threshold comparison — Critical
/Degrading/Healthy and Short/Medium/Long — no model involved, fully
explainable from raw sensor readings.

### 2c. Live monitor additions — `scripts/live_soh_rul_monitor.py`

For continuous on-vehicle monitoring (rather than one-off readings), two
mechanisms build on the same formulas above:

**Coulomb counting**, integrating current over time to track charge directly:

```
charge_Ah(t) = charge_Ah(t − dt) − I(t) · dt_hours
```

Because integration drifts over time, it's periodically **recalibrated**
against the OCV-based SOC whenever the battery has been at rest (|I| below a
threshold) for long enough for surface-charge effects to settle.

**Self-calibrating resistance baseline** — a real device has no
factory-measured "new battery" resistance, so the first detected load-step
resistance measurement becomes the baseline that subsequent readings are
compared against (using the same `SOH(%) = 100 × R_new(T)/R` formula, with
`R_new` = that self-measured baseline instead of a fixed constant).

All the "why" for the individual formulas, and calibration constants that
**must** be replaced with real datasheet/bench values before deployment, are
documented inline in `classical_soh_rul.py`.

## Part 3 — Running the demo

No hardware needed — edit sensor values and see predictions instantly:

```bash
python scripts/predict_demo.py
```

Edit the `INPUT` section at the top (`V_REST`, `V_MIN`, `I_CRANK`, `TEMP_C`,
and `HISTORY`) to any scenario, then re-run. Output prints the computed
internal resistance, predicted SOH% and class, the SOH history, and the
predicted RUL in days with its class — all in the terminal, so you can point
at the input numbers next to the output during a live demo.

For a continuous/streaming version (simulated or real serial data from the
ESP32 + INA219 firmware in `firmware/esp32_battery_monitor/`):

```bash
python scripts/live_soh_rul_monitor.py --simulate       # no hardware needed
python scripts/live_soh_rul_monitor.py --test            # runs the self-check
python scripts/live_soh_rul_monitor.py --port COM3 --capacity-ah 5.0   # real hardware
```

## Running the ML proof-of-concept yourself

```bash
pip install -r requirements.txt
python scripts/soh_rul_pipeline.py
```

Regenerates everything under `outputs/` (feature table, metrics, plots) in
under a minute. Requires the raw `.mat` files in `data/nasa_mat/` — see
**Data source** below.

## Data source

The `.mat` files are the original NASA Prognostics Center of Excellence
Battery Data Set (B. Saha and K. Goebel, 2007), room-temperature aging
experiments on four Li-ion cells cycled to failure. Official repository:
NASA's PCoE Data Set Repository (mirrored by the PHM Society at
data.phmsociety.org/nasa). If those are unreachable, the same files are also
available via the `Maruthi-prasanth/Battery-Health-Prophecy` GitHub
repository.

## Literature survey

See `Final_Year_Project_Literature_Survey_Guide.docx` for the classification
categories, paper-documentation template, comparison matrix, and current
research-gap hypothesis: no existing published work combines low-cost,
continuously-installed, IoT-enabled predictive health monitoring specifically
for conventional 12V lead-acid batteries on petrol two-wheelers.

"""All tunable settings for the SOH/RUL pipeline. Change values here, not in the other files."""

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "nasa_mat"
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots"

BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
RATED_CAPACITY_AH = 2.0   # nameplate capacity for these NASA cells
EOL_FRACTION = 0.70       # standard EOL threshold (Saha & Goebel, 2007)

SOH_BINS = [0, 65, 80, 200]                       # % thresholds
SOH_LABELS = ["Critical", "Degrading", "Healthy"]
RUL_LABELS = ["Short", "Medium", "Long"]          # tertile-split at runtime, see data.py

RANDOM_STATE = 42

FEATURE_COLS = [
    "ambient_temperature",
    "duration_s",
    "mean_voltage",
    "std_voltage",
    "min_voltage",
    "max_voltage",
    "voltage_drop_rate",
    "mean_current",
    "std_current",
    "mean_temperature",
    "max_temperature",
    "std_temperature",
    "energy_wh",
    "time_to_3v_s",
]

MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "DecisionTree": lambda: DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE
    ),
}

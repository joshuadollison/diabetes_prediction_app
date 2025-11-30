"""
Temporary local prediction runner.

Loads the toy horse model, filters a provided input CSV to a specific race_id,
computes a simple probability per horse, and writes a compact prediction CSV.
"""

import math
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd


class WeightedAverageModel:
    """
    Minimal stand-in for the dict-backed model artifact.
    Uses two features and a logistic transform to produce a pseudo-probability.
    """

    def __init__(self, weight: float = 0.5):
        self.weight = weight

    def __call__(self, row: pd.Series) -> float:
        f1 = _safe_float(row.get("pace_pressure_score", 0.0))
        f2 = _safe_float(row.get("race_strength_index", 0.0))
        score = self.weight * f1 + (1 - self.weight) * f2
        return 1 / (1 + math.exp(-score))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
        if math.isnan(val):
            return default
        return val
    except (TypeError, ValueError):
        return default


def load_model(model_path: Path) -> WeightedAverageModel:
    raw = pickle.load(model_path.open("rb"))
    if isinstance(raw, dict) and raw.get("type") == "pair_weighted_average":
        return WeightedAverageModel(weight=_safe_float(raw.get("w", 0.5), 0.5))
    raise ValueError(f"Unsupported model format in {model_path}")


def run_predictions(
    model_path: Path,
    input_csv: Path,
    race_id: str,
    output_csv: Path,
) -> None:
    model = load_model(model_path)
    df = pd.read_csv(input_csv)
    df = df[df["race_id"] == race_id].copy()

    if df.empty:
        raise ValueError(f"No rows found for race_id={race_id} in {input_csv}")

    df["prediction"] = df.apply(model, axis=1)

    out = df[["race_id", "horse_id", "horse_name", "prediction"]]
    out.to_csv(output_csv, index=False)
    print(f"Wrote {len(out)} predictions to {output_csv}")


if __name__ == "__main__":
    base_dir = Path(__file__).parent
    run_predictions(
        model_path=base_dir / "horsey_model.pkl",
        input_csv=base_dir / "model_input_20231201.csv",
        race_id="AQU_2023-12-01_1",
        output_csv=base_dir / "prediction.csv",
    )

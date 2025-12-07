"""
Temporary local prediction runner.

Loads the local horse model, filters a provided input CSV to a specific race_id,
runs real predict_proba-based inference, and writes a compact prediction CSV.
"""

from pathlib import Path

import pandas as pd

from diabetes_prediction_app import load_local_model, _predict_with_model


def run_predictions(
    model_path: Path,
    input_csv: Path,
    race_id: str,
    output_csv: Path,
) -> None:
    model, resolved_path = load_local_model(model_path)
    df = pd.read_csv(input_csv)
    df = df[df["race_id"] == race_id].copy()

    if df.empty:
        raise ValueError(f"No rows found for race_id={race_id} in {input_csv}")

    preds = _predict_with_model(model, df, resolved_path)
    if len(preds) != len(df):
        raise RuntimeError(f"Model returned {len(preds)} predictions for {len(df)} rows")
    df["prediction"] = preds

    total = df["prediction"].sum()
    if total <= 0:
        raise RuntimeError(f"Local model returned non-positive probabilities for race_id={race_id}")
    df["prediction"] = df["prediction"] / total

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

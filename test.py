"""
Model execution helper:

Loads a scikit-learn model (pickled) and applies it to the processed daily
dataset (model_features_duck) for the specified date.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable
from datetime import datetime

import pandas as pd
from sklearn.isotonic import IsotonicRegression

try:
    import lightgbm as lgb  # type: ignore
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore

try:
    import xgboost as xgb  # type: ignore
except ImportError:  # pragma: no cover
    xgb = None  # type: ignore

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stored model against daily DuckDB features.")
    parser.add_argument("--date", required=True, help="Race date (YYYY-MM-DD).")
    parser.add_argument("--model-name", required=True, help="Model name (used to locate pickle).")
    parser.add_argument(
        "--model-dir",
        default="horsey/model",
        help="Directory containing pickled models (default: horsey/model).",
    )
    parser.add_argument(
        "--daily-root",
        default="horsey/data/daily",
        help="Base directory for daily outputs (default: horsey/data/daily).",
    )
    parser.add_argument(
        "--target-column",
        default="target_win",
        help="Column name for the target (default: target_win). Will be dropped before prediction.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional prediction CSV path. "
            "Defaults to model/predictions_{model}_{run_date}.csv (run_date = today)."
        ),
    )
    parser.add_argument(
        "--ensemble-members",
        help="Comma-separated list of member model names (without extension) when using an ensemble config.",
    )
    parser.add_argument(
        "--score-column",
        help="When the model expects 1-D input (e.g., IsotonicRegression), use this numeric column name from features.",
    )
    parser.add_argument(
        "--calibration-base-model",
        help=(
            "When using a calibrator-only model (e.g., IsotonicRegression), predict with this base model "
            "and feed its scores into the calibrator. If omitted, the runner will try a sensible default."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model pickle not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_daily_features(daily_dir: Path) -> pd.DataFrame:
    parquet_path = daily_dir / "duckdb_cat5" / "model_features_duck.parquet"
    csv_path = daily_dir / "duckdb_cat5" / "model_features_duck.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} exists.")


def run_predictions(args: argparse.Namespace) -> Path:
    run_started = datetime.now()
    run_date_str = run_started.date().isoformat()
    run_timestamp = run_started.isoformat()
    date_str = args.date
    model_dir = Path(args.model_dir)
    model_path = model_dir / f"{args.model_name}.pkl"
    model = load_model(model_path)

    daily_dir = Path(args.daily_root) / date_str
    df = load_daily_features(daily_dir)
    feature_df = df.drop(columns=[args.target_column], errors="ignore")
    numeric_df = feature_df.select_dtypes(include=["number", "bool"]).copy()
    numeric_df = numeric_df.fillna(0)
    non_numeric = sorted(set(feature_df.columns) - set(numeric_df.columns))
    if non_numeric:
        print(f"[INFO] Dropping non-numeric columns for prediction: {non_numeric}")

    def align_to_model_columns(model_obj, df_in: pd.DataFrame):
        booster = None
        feature_names: list[str] | None = None

        # XGBoost booster
        if xgb is not None:
            if isinstance(model_obj, xgb.Booster):
                booster = model_obj
                feature_names = booster.feature_names
            elif hasattr(model_obj, "get_booster"):
                try:
                    booster = model_obj.get_booster()
                    feature_names = booster.feature_names
                except Exception:
                    booster = None

        # LightGBM booster
        if booster is None and lgb is not None:
            if isinstance(model_obj, lgb.Booster):
                booster = model_obj
                feature_names = booster.feature_name()
            elif hasattr(model_obj, "booster_"):
                try:
                    booster = model_obj.booster_
                    feature_names = booster.feature_name()
                except Exception:
                    booster = None

        if booster is None:
            # Non-booster models: try to align by feature names or feature count.
            if hasattr(model_obj, "feature_names_in_"):
                names = list(model_obj.feature_names_in_)
                aligned = pd.DataFrame(index=df_in.index)
                for col in names:
                    aligned[col] = df_in[col] if col in df_in.columns else 0.0
                extra_cols = [c for c in df_in.columns if c not in names]
                if extra_cols:
                    print(f"[INFO] Dropping {len(extra_cols)} extra columns not in model: {extra_cols}")
                missing_cols = [c for c in names if c not in df_in.columns]
                if missing_cols:
                    print(f"[INFO] Filling missing model columns with 0: {missing_cols}")
                return aligned
            if hasattr(model_obj, "n_features_in_"):
                expected = int(getattr(model_obj, "n_features_in_"))
                cols_sorted = sorted(df_in.columns)
                aligned = df_in[cols_sorted]
                if aligned.shape[1] > expected:
                    drop_cols = cols_sorted[expected:]
                    print(f"[INFO] Trimming {len(drop_cols)} columns to match model feature count {expected}.")
                    aligned = aligned.iloc[:, :expected]
                elif aligned.shape[1] < expected:
                    missing = expected - aligned.shape[1]
                    print(f"[INFO] Padding {missing} missing columns with 0 to match model feature count {expected}.")
                    for i in range(missing):
                        aligned[f"pad_{i}"] = 0.0
                return aligned
            return df_in

        names = feature_names
        if names:
            # If the model was trained without column names, LightGBM stores names as Column_0..N.
            # In that case, align by position instead of name so we keep real feature values.
            if all(isinstance(col, str) and col.startswith("Column_") for col in names):
                indexed = {name: df_in.iloc[:, idx] if idx < df_in.shape[1] else 0.0 for idx, name in enumerate(names)}
                if df_in.shape[1] > len(names):
                    drop_cols = df_in.columns[len(names):]
                    print(f"[INFO] Trimming {len(drop_cols)} columns to match unnamed model feature count ({len(names)}).")
                aligned = pd.DataFrame(indexed, index=df_in.index)
                return aligned

            aligned = pd.DataFrame(index=df_in.index)
            for col in names:
                aligned[col] = df_in[col] if col in df_in.columns else 0.0
            extra_cols = [c for c in df_in.columns if c not in names]
            if extra_cols:
                print(f"[INFO] Dropping {len(extra_cols)} extra columns not in model: {extra_cols}")
            missing_cols = [c for c in names if c not in df_in.columns]
            if missing_cols:
                print(f"[INFO] Filling missing model columns with 0: {missing_cols}")
            return aligned

        # Fallback when booster has no feature names: enforce column count if available.
        expected = None
        if hasattr(booster, "num_features"):
            try:
                expected = booster.num_features()
            except Exception:
                expected = None
        if not expected:
            return df_in
        cols_sorted = sorted(df_in.columns)
        trimmed = df_in[cols_sorted]
        if trimmed.shape[1] > expected:
            drop_cols = cols_sorted[expected:]
            print(f"[INFO] Dropping {len(drop_cols)} extra columns to match model feature count: {drop_cols}")
            trimmed = trimmed.iloc[:, :expected]
        elif trimmed.shape[1] < expected:
            missing = expected - trimmed.shape[1]
            print(f"[INFO] Padding {missing} missing columns with 0 to match model feature count ({expected}).")
            for i in range(missing):
                trimmed[f"pad_{i}"] = 0.0
        return trimmed

    def predict_from_model(model_obj, df_in: pd.DataFrame, *, model_label: str | None = None):
        if isinstance(model_obj, dict) and model_obj.get("type") == "pair_weighted_average":
            member_count = int(model_obj.get("members", 2))
            weight = float(model_obj.get("w", 0.5))
            member_names: list[str] = []
            if args.ensemble_members:
                member_names = [m.strip() for m in args.ensemble_members.split(",") if m.strip()]
            if not member_names:
                candidates: list[str] = []
                for p in sorted(Path(args.model_dir).glob("*.pkl")):
                    if model_label is not None and p.stem == model_label:
                        continue
                    try:
                        obj = load_model(p)
                    except Exception:
                        continue
                    if isinstance(obj, IsotonicRegression):
                        continue
                    candidates.append(p.stem)
                member_names = candidates[:member_count]
            if len(member_names) < member_count:
                raise SystemExit(
                    f"Ensemble model expects {member_count} members but only found {len(member_names)} "
                    f"(candidates: {member_names}). Provide --ensemble-members to specify models."
                )
            member_models = []
            for name in member_names[:member_count]:
                path = model_dir / f"{name}.pkl"
                if not path.exists():
                    raise SystemExit(f"Ensemble member model not found: {path}")
                member_models.append(load_model(path))
            weights = []
            if member_count == 2:
                weights = [weight, 1.0 - weight]
            else:
                weights = [1.0 / member_count] * member_count
            preds = []
            for sub_model, w in zip(member_models, weights):
                preds.append(w * predict_from_model(sub_model, df_in, model_label=name))
            return sum(preds)

        if isinstance(model_obj, IsotonicRegression):
            if df_in.empty:
                raise SystemExit("No numeric features available for isotonic regression input.")
            score_col = None
            if args.score_column and args.score_column in df_in.columns:
                score_col = args.score_column
            else:
                base_name = args.calibration_base_model
                # Choose a default base if none provided.
                if base_name is None:
                    default_path = Path(args.model_dir) / "more_real.pkl"
                    if default_path.exists():
                        base_name = "more_real"
                    else:
                        for p in sorted(Path(args.model_dir).glob("*.pkl")):
                            if model_label is not None and p.stem == model_label:
                                continue
                            try:
                                obj = load_model(p)
                            except Exception:
                                continue
                            if isinstance(obj, IsotonicRegression) or isinstance(obj, dict):
                                continue
                            base_name = p.stem
                            break
                if base_name:
                    if base_name == (model_label or args.model_name):
                        raise SystemExit("Calibration base model cannot match the calibrator model.")
                    base_path = Path(args.model_dir) / f"{base_name}.pkl"
                    if not base_path.exists():
                        raise SystemExit(f"Calibration base model not found: {base_path}")
                    print(f"[INFO] Using calibration base model '{base_name}'.")
                    base_model = load_model(base_path)
                    base_scores = predict_from_model(base_model, df_in, model_label=base_name)
                    series = pd.Series(base_scores, index=df_in.index)
                elif df_in.shape[1] == 1:
                    score_col = df_in.columns[0]
                    series = df_in[score_col].astype(float)
                else:
                    raise SystemExit(
                        "IsotonicRegression requires a 1-D input. "
                        "Provide --score-column <col> or --calibration-base-model <model> to supply the scores. "
                        "If using an ensemble base, also set --ensemble-members to specify its components."
                    )
            if score_col:
                series = df_in[score_col].astype(float)
            calibrated = model_obj.predict(series.values)
            unique_vals = pd.unique(calibrated)
            dominant_ratio = pd.Series(calibrated).value_counts(normalize=True, dropna=False).iloc[0]
            if (
                (len(unique_vals) <= 3)
                or (pd.Series(calibrated).std(ddof=0) < 1e-8)
                or (dominant_ratio > 0.9 and getattr(series, "nunique", lambda: 0)() > 1)
            ):
                print("[WARN] Calibrated outputs are near-constant; falling back to raw scores.")
                return series.values
            return calibrated

        aligned = align_to_model_columns(model_obj, df_in)

        # LightGBM booster support
        if lgb is not None and isinstance(model_obj, lgb.Booster):
            num_iter = getattr(model_obj, "best_iteration", None)
            return model_obj.predict(aligned, num_iteration=num_iter)
        if lgb is not None and hasattr(model_obj, "predict") and model_obj.__class__.__module__.startswith("lightgbm"):
            return model_obj.predict(aligned)

        if xgb is not None and isinstance(model_obj, xgb.Booster):
            dmatrix = xgb.DMatrix(aligned)
            return model_obj.predict(dmatrix)
        if hasattr(model_obj, "predict_proba"):
            proba_input = aligned
            if xgb is not None and hasattr(model_obj, "get_booster") and isinstance(model_obj.get_booster(), xgb.Booster):
                proba_input = xgb.DMatrix(aligned)
            proba = model_obj.predict_proba(proba_input)
            return proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
        predict_input = aligned
        if xgb is not None and hasattr(model_obj, "get_booster") and isinstance(model_obj.get_booster(), xgb.Booster):
            predict_input = xgb.DMatrix(aligned)
        return model_obj.predict(predict_input)

    predictions = predict_from_model(model, numeric_df, model_label=args.model_name)

    output_path = (
        Path(args.output)
        if args.output
        else Path("model") / f"predictions_{args.model_name}_{run_date_str}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_cols = [
        col
        for col in [
            "race_id",
            "horse_id",
            "horse_name",
            "entry_name",
            "program_num",
            "track_code",
        ]
        if col in df.columns
    ]
    out_df = df[base_cols].copy() if base_cols else df.copy()
    out_df["race_date"] = date_str
    out_df["run_at"] = run_timestamp
    out_df["model"] = args.model_name
    out_df["prediction"] = predictions

    holdout_path = Path("horsey/data/holdout_dec23_firsts.csv")
    if not holdout_path.exists():
        holdout_path = Path("horsey/data/canonical/holdout_dec23_firsts.csv")
    holdout_map: dict[str, dict[str, str | None]] = {}
    if holdout_path.exists():
        try:
            holdout_df = pd.read_csv(holdout_path)
            if "race_id" in holdout_df.columns:
                name_col = "entry_name" if "entry_name" in holdout_df.columns else "NAME" if "NAME" in holdout_df.columns else None
                prog_col = "program_num" if "program_num" in holdout_df.columns else "PROGRAM_NUM_1" if "PROGRAM_NUM_1" in holdout_df.columns else None
                if name_col:
                    subset = holdout_df[["race_id", name_col] + ([prog_col] if prog_col else [])].drop_duplicates("race_id")
                    for _, row in subset.iterrows():
                        holdout_map[str(row["race_id"])] = {
                            "winner_name": str(row[name_col]) if pd.notna(row[name_col]) else "",
                            "program_num": str(row[prog_col]) if prog_col and pd.notna(row[prog_col]) else None,
                        }
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Failed to load holdout file {holdout_path}: {exc}")

    out_df["holdout_winner_name"] = ""
    out_df["holdout_winner_horse_id"] = ""
    out_df["is_holdout_winner"] = False
    if "race_id" in out_df.columns:
        out_df = out_df.sort_values(["race_id", "prediction"], ascending=[True, False]).reset_index(drop=True)
        if holdout_map:
            def annotate(group: pd.DataFrame) -> pd.DataFrame:
                race = str(group["race_id"].iloc[0]) if "race_id" in group.columns else str(group.name)
                info = holdout_map.get(race)
                if info is None or group.empty:
                    return group
                winner_name = info.get("winner_name", "") or ""
                program_num = info.get("program_num")

                def _norm(val: str | float | None) -> str:
                    return str(val).strip().lower() if val is not None and pd.notna(val) else ""

                winner_name_norm = _norm(winner_name)
                winner_horse_id = ""

                candidates = group.copy()
                name_match = candidates["horse_name"].apply(_norm) == winner_name_norm
                entry_match = candidates["entry_name"].apply(_norm) == winner_name_norm if "entry_name" in candidates.columns else pd.Series([False] * len(candidates), index=candidates.index)
                prog_match = (
                    candidates["program_num"].apply(_norm) == _norm(program_num)
                    if program_num is not None and "program_num" in candidates.columns
                    else pd.Series([False] * len(candidates), index=candidates.index)
                )
                combined = name_match | entry_match | prog_match
                if combined.any():
                    winner_horse_id = str(candidates.loc[combined.idxmax(), "horse_id"])

                group = group.copy()
                if "race_id" not in group.columns:
                    group["race_id"] = race
                group["holdout_winner_name"] = winner_name
                group["holdout_winner_horse_id"] = winner_horse_id
                if winner_horse_id:
                    group["is_holdout_winner"] = group["horse_id"].astype(str) == winner_horse_id
                else:
                    group["is_holdout_winner"] = name_match | entry_match
                return group

            out_df = out_df.groupby("race_id", group_keys=False).apply(annotate, include_groups=False)
    else:
        out_df = out_df.sort_values("prediction", ascending=False).reset_index(drop=True)

    header = not output_path.exists()
    out_df.to_csv(output_path, index=False, mode="a", header=header)
    return output_path


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output = run_predictions(args)
    print(f"Predictions written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

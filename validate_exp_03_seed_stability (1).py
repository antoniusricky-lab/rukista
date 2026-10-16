"""Paired multi-seed validation for EXP-03; never creates a submission."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
ANALYSIS_DIR = SCRIPT_PATH.parents[1]
LOCAL_DEPS = ANALYSIS_DIR / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ANALYSIS_DIR))
sys.path.insert(0, str(SCRIPT_PATH.parent))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score

from audio_features import numeric_feature_columns
from baseline import LABEL_ORDER, align_probabilities, balanced_group_splits, make_model
from exp_03_balanced_bootstrap import fit_balanced_forest, predict_forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=ANALYSIS_DIR / "cache/audio_features.csv",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=ANALYSIS_DIR / "experiments/results/exp_03_balanced_bootstrap",
    )
    parser.add_argument("--seeds", default="13,137,2026")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def run_baseline(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros((len(y), len(LABEL_ORDER)), dtype=float)
    fold_scores: list[float] = []
    for fold, (train_index, valid_index) in enumerate(
        balanced_group_splits(y, groups, n_splits=4), start=1
    ):
        model = make_model(n_estimators, n_jobs, seed + fold)
        model.fit(X[train_index], y[train_index])
        probability = align_probabilities(
            model.predict_proba(X[valid_index]), model.classes_, LABEL_ORDER
        )
        oof[valid_index] = probability
        prediction = np.asarray(LABEL_ORDER)[probability.argmax(axis=1)]
        fold_scores.append(
            f1_score(
                y[valid_index],
                prediction,
                labels=LABEL_ORDER,
                average="macro",
                zero_division=0,
            )
        )
    return oof, np.asarray(fold_scores, dtype=float)


def run_challenger(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_estimators: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.zeros((len(y), len(LABEL_ORDER)), dtype=float)
    fold_scores: list[float] = []
    for fold, (train_index, valid_index) in enumerate(
        balanced_group_splits(y, groups, n_splits=4), start=1
    ):
        trees = fit_balanced_forest(
            X[train_index],
            y[train_index],
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            seed=seed + fold,
        )
        probability = predict_forest(trees, X[valid_index])
        oof[valid_index] = probability
        prediction = np.asarray(LABEL_ORDER)[probability.argmax(axis=1)]
        fold_scores.append(
            f1_score(
                y[valid_index],
                prediction,
                labels=LABEL_ORDER,
                average="macro",
                zero_division=0,
            )
        )
    return oof, np.asarray(fold_scores, dtype=float)


def metrics_for_oof(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(LABEL_ORDER)[probability.argmax(axis=1)]
    report = classification_report(
        y,
        prediction,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "cendrawasih_f1": float(report["Cendrawasih"]["f1-score"]),
        "cendrawasih_recall": float(report["Cendrawasih"]["recall"]),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")

    features = pd.read_csv(args.feature_cache, low_memory=False)
    columns = numeric_feature_columns(features)
    train = features.loc[features["split"] == "train"].reset_index(drop=True)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(train[columns].replace([np.inf, -np.inf], np.nan))
    y = train["label"].astype(str).to_numpy()
    groups = train["source_group"].astype(str).to_numpy()

    rows: list[dict] = []
    for seed in seeds:
        baseline_probability, baseline_fold = run_baseline(
            X, y, groups, seed, args.n_estimators, args.n_jobs
        )
        challenger_probability, challenger_fold = run_challenger(
            X, y, groups, seed, args.n_estimators, args.n_jobs
        )
        baseline_metrics = metrics_for_oof(y, baseline_probability)
        challenger_metrics = metrics_for_oof(y, challenger_probability)
        deltas = challenger_fold - baseline_fold
        seed_passed = bool(
            np.all(deltas > 1e-12)
            and challenger_metrics["macro_f1"] > baseline_metrics["macro_f1"]
            and challenger_metrics["cendrawasih_f1"]
            > baseline_metrics["cendrawasih_f1"]
            and challenger_metrics["cendrawasih_recall"]
            > baseline_metrics["cendrawasih_recall"]
        )
        row = {
            "seed": seed,
            **{f"baseline_fold_{i + 1}": score for i, score in enumerate(baseline_fold)},
            **{
                f"challenger_fold_{i + 1}": score
                for i, score in enumerate(challenger_fold)
            },
            **{f"delta_fold_{i + 1}": score for i, score in enumerate(deltas)},
            "baseline_fold_mean": float(np.mean(baseline_fold)),
            "challenger_fold_mean": float(np.mean(challenger_fold)),
            "mean_delta": float(np.mean(deltas)),
            "baseline_fold_std": float(np.std(baseline_fold, ddof=1)),
            "challenger_fold_std": float(np.std(challenger_fold, ddof=1)),
            "baseline_oof_macro_f1": baseline_metrics["macro_f1"],
            "challenger_oof_macro_f1": challenger_metrics["macro_f1"],
            "baseline_cendrawasih_f1": baseline_metrics["cendrawasih_f1"],
            "challenger_cendrawasih_f1": challenger_metrics["cendrawasih_f1"],
            "baseline_cendrawasih_recall": baseline_metrics["cendrawasih_recall"],
            "challenger_cendrawasih_recall": challenger_metrics["cendrawasih_recall"],
            "strict_seed_gate_passed": seed_passed,
        }
        rows.append(row)
        print(
            f"seed={seed} mean_delta={row['mean_delta']:+.6f} "
            f"all_fold_up={bool(np.all(deltas > 0))} "
            f"cend_f1={baseline_metrics['cendrawasih_f1']:.6f}"
            f"->{challenger_metrics['cendrawasih_f1']:.6f} "
            f"passed={seed_passed}",
            flush=True,
        )

    result = pd.DataFrame(rows)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.result_dir / "seed_stability.csv", index=False)
    summary = {
        "seeds": seeds,
        "all_seeds_passed": bool(result["strict_seed_gate_passed"].all()),
        "mean_of_paired_mean_deltas": float(result["mean_delta"].mean()),
        "minimum_fold_delta_all_seeds": float(
            result[[f"delta_fold_{i}" for i in range(1, 5)]].to_numpy().min()
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.result_dir / "seed_stability_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

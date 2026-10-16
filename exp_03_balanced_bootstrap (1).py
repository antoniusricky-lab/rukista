"""EXP-03: replace class-weight balancing with balanced bootstrap per tree.

Each ExtraTree receives an equal-size bootstrap sample from every class.  The
base tree parameters, cached features, imputation, group folds, seeds, and
baseline Unknown threshold remain fixed.  A submission is emitted only if the
Macro F1 score increases in every fold and aggregate Cendrawasih F1 improves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
ANALYSIS_DIR = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
LOCAL_DEPS = ANALYSIS_DIR / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ANALYSIS_DIR))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.tree import ExtraTreeClassifier

from audio_features import SEED, numeric_feature_columns
from baseline import LABEL_ORDER, align_probabilities, balanced_group_splits


BASELINE_THRESHOLD = 0.5738699004323102


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=ANALYSIS_DIR / "cache/audio_features.csv",
    )
    parser.add_argument(
        "--baseline-folds",
        type=Path,
        default=ANALYSIS_DIR / "diagnostics/results/fold_metrics.csv",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=ANALYSIS_DIR / "results/classification_report_group_cv.csv",
    )
    parser.add_argument(
        "--baseline-fold-class",
        type=Path,
        default=ANALYSIS_DIR / "diagnostics/results/fold_class_metrics.csv",
    )
    parser.add_argument(
        "--baseline-oof",
        type=Path,
        default=ANALYSIS_DIR / "diagnostics/results/oof_diagnostics.csv",
    )
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=PROJECT_ROOT / "Data Schematics/sample_submission.csv",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=ANALYSIS_DIR / "experiments/results/exp_03_balanced_bootstrap",
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=ANALYSIS_DIR / "submissions",
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(payload: dict, path: Path) -> None:
    def convert(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value)}")

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=convert),
        encoding="utf-8",
    )


def balanced_bootstrap_indices(y: np.ndarray, seed: int) -> np.ndarray:
    """Draw an equal number from every class, using replacement per class."""
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y == label) for label in LABEL_ORDER]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("Every training fold must contain all known classes.")
    rows_per_class = int(np.ceil(len(y) / len(LABEL_ORDER)))
    selected = np.concatenate(
        [rng.choice(indices, size=rows_per_class, replace=True) for indices in class_indices]
    )
    rng.shuffle(selected)
    return selected


def fit_balanced_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int,
    n_jobs: int,
    seed: int,
) -> list[ExtraTreeClassifier]:
    def fit_one(tree_index: int) -> ExtraTreeClassifier:
        tree_seed = seed + tree_index
        selected = balanced_bootstrap_indices(y, tree_seed)
        tree = ExtraTreeClassifier(
            criterion="gini",
            splitter="random",
            max_features="sqrt",
            min_samples_split=2,
            min_samples_leaf=2,
            class_weight=None,
            random_state=tree_seed,
        )
        tree.fit(X[selected], y[selected])
        return tree

    return joblib.Parallel(n_jobs=n_jobs, prefer="threads")(
        joblib.delayed(fit_one)(tree_index) for tree_index in range(n_estimators)
    )


def predict_forest(trees: list[ExtraTreeClassifier], X: np.ndarray) -> np.ndarray:
    probability = np.zeros((len(X), len(LABEL_ORDER)), dtype=float)
    for tree in trees:
        probability += align_probabilities(tree.predict_proba(X), tree.classes_, LABEL_ORDER)
    probability /= len(trees)
    return probability


def run_group_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_estimators: int,
    n_jobs: int,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    oof_probability = np.zeros((len(y), len(LABEL_ORDER)), dtype=float)
    oof_fold = np.zeros(len(y), dtype=int)
    rows: list[dict] = []
    importances: list[np.ndarray] = []
    for fold, (train_index, valid_index) in enumerate(
        balanced_group_splits(y, groups, n_splits=4), start=1
    ):
        trees = fit_balanced_forest(
            X[train_index],
            y[train_index],
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            seed=SEED + fold,
        )
        probability = predict_forest(trees, X[valid_index])
        oof_probability[valid_index] = probability
        oof_fold[valid_index] = fold
        prediction = np.asarray(LABEL_ORDER)[probability.argmax(axis=1)]
        score = f1_score(
            y[valid_index],
            prediction,
            labels=LABEL_ORDER,
            average="macro",
            zero_division=0,
        )
        fold_report = classification_report(
            y[valid_index],
            prediction,
            labels=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        )
        rows.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "valid_rows": len(valid_index),
                "balanced_rows_per_tree": int(
                    len(LABEL_ORDER) * np.ceil(len(train_index) / len(LABEL_ORDER))
                ),
                "macro_f1": float(score),
                "cendrawasih_f1": float(fold_report["Cendrawasih"]["f1-score"]),
                "cendrawasih_recall": float(fold_report["Cendrawasih"]["recall"]),
            }
        )
        importances.append(np.mean([tree.feature_importances_ for tree in trees], axis=0))
        print(f"Group CV fold {fold}/4: macro F1={score:.6f}", flush=True)
    return oof_probability, pd.DataFrame(rows), np.mean(importances, axis=0), oof_fold


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    allowed = set(LABEL_ORDER + ["Unknown"])
    if list(submission.columns) != ["file_id", "label"]:
        raise ValueError("Submission must have exactly file_id,label columns.")
    if len(submission) != len(sample):
        raise ValueError("Submission row count differs from sample_submission.")
    if submission.isna().any().any() or submission["file_id"].duplicated().any():
        raise ValueError("Submission contains missing values or duplicate IDs.")
    if not submission["file_id"].astype(str).equals(sample["file_id"].astype(str)):
        raise ValueError("Submission ID order differs from sample_submission.")
    if not set(submission["label"].astype(str)).issubset(allowed):
        raise ValueError("Submission contains invalid labels.")


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    np.random.seed(SEED)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.feature_cache, low_memory=False)
    columns = numeric_feature_columns(features)
    train = features.loc[features["split"] == "train"].reset_index(drop=True)
    test = features.loc[features["split"] == "test"].reset_index(drop=True)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train[columns].replace([np.inf, -np.inf], np.nan))
    X_test = imputer.transform(test[columns].replace([np.inf, -np.inf], np.nan))
    y = train["label"].astype(str).to_numpy()
    groups = train["source_group"].astype(str).to_numpy()

    probability, experiment_fold, mean_importance, oof_fold = run_group_cv(
        X_train, y, groups, args.n_estimators, args.n_jobs
    )
    prediction = np.asarray(LABEL_ORDER)[probability.argmax(axis=1)]
    baseline_fold = pd.read_csv(args.baseline_folds).sort_values("fold").reset_index(drop=True)
    experiment_fold = experiment_fold.sort_values("fold").reset_index(drop=True)
    if not experiment_fold["fold"].equals(baseline_fold["fold"]):
        raise ValueError("Experiment folds do not match baseline folds.")

    baseline_scores = baseline_fold["macro_f1"].to_numpy(dtype=float)
    experiment_scores = experiment_fold["macro_f1"].to_numpy(dtype=float)
    deltas = experiment_scores - baseline_scores
    report = classification_report(
        y,
        prediction,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    baseline_report = pd.read_csv(args.baseline_report, index_col=0)
    baseline_oof_full = pd.read_csv(args.baseline_oof)
    baseline_fold_class = pd.read_csv(args.baseline_fold_class)
    baseline_cendrawasih_fold = (
        baseline_fold_class.loc[baseline_fold_class["label"] == "Cendrawasih"]
        .sort_values("fold")
        .reset_index(drop=True)
    )
    if not experiment_fold["fold"].equals(baseline_cendrawasih_fold["fold"]):
        raise ValueError("Cendrawasih fold rows do not match the baseline.")
    baseline_cendrawasih_f1 = float(baseline_report.loc["Cendrawasih", "f1-score"])
    baseline_cendrawasih_recall = float(baseline_report.loc["Cendrawasih", "recall"])
    experiment_cendrawasih_f1 = float(report["Cendrawasih"]["f1-score"])
    experiment_cendrawasih_recall = float(report["Cendrawasih"]["recall"])
    baseline_oof_macro_f1 = float(baseline_report.loc["macro avg", "f1-score"])
    experiment_oof_macro_f1 = float(
        f1_score(y, prediction, labels=LABEL_ORDER, average="macro", zero_division=0)
    )
    baseline_std = float(np.std(baseline_scores, ddof=1))
    experiment_std = float(np.std(experiment_scores, ddof=1))
    cendrawasih_fold_delta = (
        experiment_fold["cendrawasih_f1"].to_numpy(dtype=float)
        - baseline_cendrawasih_fold["f1"].to_numpy(dtype=float)
    )
    experiment_source = pd.DataFrame(
        {
            "source_group": groups[y == "Cendrawasih"],
            "correct": prediction[y == "Cendrawasih"] == "Cendrawasih",
        }
    ).groupby("source_group", as_index=False).agg(
        rows=("correct", "size"), experiment_recall=("correct", "mean")
    )
    baseline_source = (
        baseline_oof_full.loc[baseline_oof_full["label"] == "Cendrawasih"]
        .groupby("source_group", as_index=False)
        .agg(rows=("correct", "size"), baseline_recall=("correct", "mean"))
    )
    source_comparison = baseline_source.merge(
        experiment_source,
        on=["source_group", "rows"],
        how="outer",
        validate="one_to_one",
    )
    if source_comparison.isna().any().any():
        raise ValueError("Cendrawasih source groups differ from the baseline.")
    source_comparison["recall_delta"] = (
        source_comparison["experiment_recall"] - source_comparison["baseline_recall"]
    )
    baseline_source_mean = float(source_comparison["baseline_recall"].mean())
    experiment_source_mean = float(source_comparison["experiment_recall"].mean())
    baseline_source_median = float(source_comparison["baseline_recall"].median())
    experiment_source_median = float(source_comparison["experiment_recall"].median())
    strict_gate_passed = bool(
        np.all(deltas > 1e-12)
        and experiment_oof_macro_f1 > baseline_oof_macro_f1 + 1e-12
        and experiment_cendrawasih_f1 > baseline_cendrawasih_f1 + 1e-12
        and experiment_cendrawasih_recall > baseline_cendrawasih_recall + 1e-12
        and np.all(cendrawasih_fold_delta >= -1e-12)
        and np.any(cendrawasih_fold_delta > 1e-12)
        and experiment_source_mean > baseline_source_mean + 1e-12
        and experiment_source_median > baseline_source_median + 1e-12
        and experiment_std <= baseline_std + 0.01
    )

    comparison = pd.DataFrame(
        {
            "fold": experiment_fold["fold"],
            "baseline_macro_f1": baseline_scores,
            "experiment_macro_f1": experiment_scores,
            "delta": deltas,
            "balanced_rows_per_tree": experiment_fold["balanced_rows_per_tree"],
            "baseline_cendrawasih_f1": baseline_cendrawasih_fold["f1"],
            "experiment_cendrawasih_f1": experiment_fold["cendrawasih_f1"],
            "cendrawasih_f1_delta": cendrawasih_fold_delta,
        }
    )
    comparison.to_csv(args.result_dir / "fold_comparison.csv", index=False)
    source_comparison.to_csv(
        args.result_dir / "cendrawasih_source_recall.csv", index=False
    )
    pd.DataFrame(report).T.to_csv(args.result_dir / "classification_report.csv")
    pd.DataFrame(
        {"feature": columns, "importance": mean_importance}
    ).sort_values("importance", ascending=False).to_csv(
        args.result_dir / "feature_importance.csv", index=False
    )

    oof = train[["file_id", "label", "source_group", "duration_sec"]].copy()
    for class_index, label in enumerate(LABEL_ORDER):
        oof[f"prob_{label}"] = probability[:, class_index]
    oof["prediction"] = prediction
    oof["confidence"] = probability.max(axis=1)
    oof["correct"] = oof["label"] == oof["prediction"]
    oof["fold"] = oof_fold
    baseline_oof = baseline_oof_full[["file_id", "fold"]]
    fold_check = oof[["file_id", "fold"]].merge(
        baseline_oof,
        on="file_id",
        how="outer",
        suffixes=("_experiment", "_baseline"),
        validate="one_to_one",
    )
    if (
        fold_check.isna().any().any()
        or not (
            fold_check["fold_experiment"].to_numpy()
            == fold_check["fold_baseline"].to_numpy()
        ).all()
    ):
        raise ValueError("file_id-to-fold mapping differs from the baseline manifest.")
    oof.to_csv(args.result_dir / "oof_predictions.csv", index=False)

    submission_path: Path | None = None
    test_counts: dict[str, int] | None = None
    if strict_gate_passed:
        trees = fit_balanced_forest(
            X_train,
            y,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
            seed=SEED + 999,
        )
        test_probability = predict_forest(trees, X_test)
        test_prediction = np.asarray(LABEL_ORDER, dtype=object)[test_probability.argmax(axis=1)]
        test_prediction[test_probability.max(axis=1) < BASELINE_THRESHOLD] = "Unknown"
        sample = pd.read_csv(args.sample_submission, dtype={"file_id": str})
        submission = pd.DataFrame(
            {"file_id": test["file_id"].astype(str), "label": test_prediction}
        )
        validate_submission(submission, sample)
        args.submission_dir.mkdir(parents=True, exist_ok=True)
        submission_path = args.submission_dir / "submission_v03_balanced_bootstrap.csv"
        if submission_path.exists():
            existing = pd.read_csv(submission_path, dtype={"file_id": str})
            if not existing.equals(submission):
                raise FileExistsError(
                    f"{submission_path} exists with different content; refusing to overwrite."
                )
        else:
            submission.to_csv(submission_path, index=False)
        test_counts = submission["label"].value_counts().to_dict()

    metrics = {
        "experiment_id": "EXP-03",
        "single_change": "replace_class_weight_with_balanced_bootstrap_per_tree",
        "seed": SEED,
        "base_estimator": "ExtraTreeClassifier",
        "n_estimators": args.n_estimators,
        "feature_count": len(columns),
        "baseline_fold_macro_f1": baseline_scores,
        "experiment_fold_macro_f1": experiment_scores,
        "fold_delta": deltas,
        "baseline_fold_mean": float(np.mean(baseline_scores)),
        "baseline_fold_std_sample": baseline_std,
        "experiment_fold_mean": float(np.mean(experiment_scores)),
        "experiment_fold_std_sample": experiment_std,
        "mean_delta": float(np.mean(deltas)),
        "baseline_oof_aggregate_macro_f1": baseline_oof_macro_f1,
        "oof_aggregate_macro_f1": experiment_oof_macro_f1,
        "baseline_oof_accuracy": float(
            accuracy_score(
                baseline_oof_full["label"].astype(str),
                baseline_oof_full["prediction"].astype(str),
            )
        ),
        "experiment_oof_accuracy": float(accuracy_score(y, prediction)),
        "baseline_oof_balanced_accuracy": float(
            balanced_accuracy_score(
                baseline_oof_full["label"].astype(str),
                baseline_oof_full["prediction"].astype(str),
            )
        ),
        "experiment_oof_balanced_accuracy": float(
            balanced_accuracy_score(y, prediction)
        ),
        "baseline_cendrawasih_f1": baseline_cendrawasih_f1,
        "experiment_cendrawasih_f1": experiment_cendrawasih_f1,
        "baseline_cendrawasih_recall": baseline_cendrawasih_recall,
        "experiment_cendrawasih_recall": experiment_cendrawasih_recall,
        "cendrawasih_fold_f1_delta": cendrawasih_fold_delta,
        "baseline_cendrawasih_source_recall_mean": baseline_source_mean,
        "experiment_cendrawasih_source_recall_mean": experiment_source_mean,
        "baseline_cendrawasih_source_recall_median": baseline_source_median,
        "experiment_cendrawasih_source_recall_median": experiment_source_median,
        "strict_gate_passed": strict_gate_passed,
        "submission_created": submission_path is not None,
        "submission_path": str(submission_path) if submission_path else None,
        "submission_sha256": sha256(submission_path) if submission_path else None,
        "test_prediction_counts": test_counts,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "runtime_seconds": time.perf_counter() - started,
    }
    save_json(metrics, args.result_dir / "metrics.json")

    print(comparison.to_string(index=False), flush=True)
    print(
        f"EXP-03 mean+/-sd={metrics['experiment_fold_mean']:.6f}"
        f"+/-{metrics['experiment_fold_std_sample']:.6f} "
        f"delta={metrics['mean_delta']:+.6f}",
        flush=True,
    )
    print(
        f"Cendrawasih F1 baseline={baseline_cendrawasih_f1:.6f} "
        f"experiment={experiment_cendrawasih_f1:.6f}",
        flush=True,
    )
    print(f"strict_gate_passed={strict_gate_passed}", flush=True)
    print(f"submission_created={submission_path is not None}", flush=True)
    print(f"results={args.result_dir}", flush=True)


if __name__ == "__main__":
    main()

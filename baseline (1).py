"""Group-aware Extra Trees baseline with heuristic open-set rejection.

Outputs include CV metrics, OOF predictions, feature importance, error analysis,
and an unscored baseline submission. The Unknown threshold is explicitly treated
as a heuristic because no labeled Unknown examples are supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DEPS = SCRIPT_DIR / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

from audio_features import SEED, build_feature_cache, numeric_feature_columns

LABEL_ORDER = ["Bondol Jawa", "Cendrawasih", "Kutilang"]
ALL_SUBMISSION_LABELS = LABEL_ORDER + ["Unknown"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR.parent / "Data Schematics")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def save_json(payload: dict, path: Path) -> None:
    def convert(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value)}")

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=convert), encoding="utf-8")


def make_model(n_estimators: int, n_jobs: int, seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
    )


def balanced_group_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 4,
):
    """Yield deterministic group folds with every class represented per fold.

    Each source_group is label-pure in this dataset. Groups are allocated within
    each label from largest to smallest to the currently lightest fold. This
    preserves groups, guarantees representation when each label has at least
    ``n_splits`` groups, and balances row counts better than a naive round-robin.
    """
    group_frame = pd.DataFrame({"label": y, "group": groups}).groupby("group", as_index=False).agg(
        label=("label", "first"), rows=("label", "size"), label_count=("label", "nunique")
    )
    if int(group_frame["label_count"].max()) != 1:
        raise ValueError("balanced_group_splits requires label-pure groups")
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    for label, label_groups in group_frame.groupby("label"):
        if len(label_groups) < n_splits:
            raise ValueError(f"Label {label!r} has only {len(label_groups)} groups for {n_splits} folds")
        loads = np.zeros(n_splits, dtype=int)
        ordered = label_groups.sort_values(["rows", "group"], ascending=[False, True])
        for row in ordered.itertuples(index=False):
            fold = int(np.argmin(loads))
            folds[fold].append(str(row.group))
            loads[fold] += int(row.rows)
    for valid_groups in folds:
        valid_mask = np.isin(groups, valid_groups)
        yield np.flatnonzero(~valid_mask), np.flatnonzero(valid_mask)


def align_probabilities(probability: np.ndarray, classes: np.ndarray, target_classes: list[str]) -> np.ndarray:
    aligned = np.zeros((len(probability), len(target_classes)), dtype=float)
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    for target_idx, label in enumerate(target_classes):
        if label in class_to_index:
            aligned[:, target_idx] = probability[:, class_to_index[label]]
    return aligned


def run_group_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_estimators: int,
    n_jobs: int,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    oof_probability = np.zeros((len(y), len(LABEL_ORDER)), dtype=float)
    fold_rows: list[dict] = []
    importances: list[np.ndarray] = []
    for fold, (train_index, valid_index) in enumerate(balanced_group_splits(y, groups, n_splits=4), start=1):
        model = make_model(n_estimators, n_jobs, SEED + fold)
        model.fit(X[train_index], y[train_index])
        probability = model.predict_proba(X[valid_index])
        oof_probability[valid_index] = align_probabilities(probability, model.classes_, LABEL_ORDER)
        prediction = np.asarray(LABEL_ORDER)[np.argmax(oof_probability[valid_index], axis=1)]
        fold_rows.append(
            {
                "scheme": "BalancedGroupKFold",
                "fold": fold,
                "train_rows": len(train_index),
                "valid_rows": len(valid_index),
                "train_groups": len(np.unique(groups[train_index])),
                "valid_groups": len(np.unique(groups[valid_index])),
                "valid_group_ids": ",".join(sorted(np.unique(groups[valid_index]))),
                "valid_label_counts": json.dumps(pd.Series(y[valid_index]).value_counts().to_dict(), ensure_ascii=False),
                "macro_f1": f1_score(y[valid_index], prediction, labels=LABEL_ORDER, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y[valid_index], prediction),
            }
        )
        importances.append(model.feature_importances_)
        print(f"Group CV fold {fold}/4: macro F1={fold_rows[-1]['macro_f1']:.4f}", flush=True)
    return oof_probability, fold_rows, np.mean(importances, axis=0)


def run_random_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int,
    n_jobs: int,
) -> tuple[np.ndarray, list[dict]]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probability = np.zeros((len(y), len(LABEL_ORDER)), dtype=float)
    fold_rows: list[dict] = []
    for fold, (train_index, valid_index) in enumerate(splitter.split(X, y), start=1):
        model = make_model(n_estimators, n_jobs, SEED + 100 + fold)
        model.fit(X[train_index], y[train_index])
        probability = model.predict_proba(X[valid_index])
        oof_probability[valid_index] = align_probabilities(probability, model.classes_, LABEL_ORDER)
        prediction = np.asarray(LABEL_ORDER)[np.argmax(oof_probability[valid_index], axis=1)]
        fold_rows.append(
            {
                "scheme": "StratifiedKFold (leakage-prone comparator)",
                "fold": fold,
                "train_rows": len(train_index),
                "valid_rows": len(valid_index),
                "macro_f1": f1_score(y[valid_index], prediction, labels=LABEL_ORDER, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y[valid_index], prediction),
            }
        )
    return oof_probability, fold_rows


def pseudo_unknown_thresholds(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_estimators: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, float]:
    """Hold out each species as pseudo-Unknown and tune a confidence cutoff.

    This diagnostic is deliberately separate from the honest known-class CV.
    """
    rows: list[dict] = []
    threshold_grid = np.linspace(0.50, 0.99, 50)
    for held_label in LABEL_ORDER:
        known_classes = [label for label in LABEL_ORDER if label != held_label]
        known_mask = y != held_label
        X_known, y_known, groups_known = X[known_mask], y[known_mask], groups[known_mask]
        known_probability = np.zeros((len(y_known), len(known_classes)), dtype=float)

        for fold, (train_index, valid_index) in enumerate(
            balanced_group_splits(y_known, groups_known, n_splits=4), start=1
        ):
            model = make_model(max(250, n_estimators // 2), n_jobs, SEED + 200 + fold)
            model.fit(X_known[train_index], y_known[train_index])
            known_probability[valid_index] = align_probabilities(
                model.predict_proba(X_known[valid_index]), model.classes_, known_classes
            )

        held_model = make_model(max(250, n_estimators // 2), n_jobs, SEED + 300)
        held_model.fit(X_known, y_known)
        held_probability = align_probabilities(
            held_model.predict_proba(X[~known_mask]), held_model.classes_, known_classes
        )
        combined_probability = np.vstack([known_probability, held_probability])
        combined_true = np.concatenate([y_known, np.full(np.sum(~known_mask), "Unknown", dtype=object)])
        base_prediction = np.asarray(known_classes)[np.argmax(combined_probability, axis=1)]
        confidence = np.max(combined_probability, axis=1)

        best = None
        for threshold in threshold_grid:
            prediction = base_prediction.copy().astype(object)
            prediction[confidence < threshold] = "Unknown"
            score = f1_score(
                combined_true,
                prediction,
                labels=known_classes + ["Unknown"],
                average="macro",
                zero_division=0,
            )
            if best is None or score > best["pseudo_open_macro_f1"]:
                best = {
                    "held_species_as_unknown": held_label,
                    "threshold": float(threshold),
                    "pseudo_open_macro_f1": float(score),
                    "unknown_recall": float(np.mean(prediction[~known_mask] == "Unknown")),
                    "known_rejection_rate": float(np.mean(prediction[: len(y_known)] == "Unknown")),
                }
        assert best is not None
        rows.append(best)
    frame = pd.DataFrame(rows)
    return frame, float(frame["threshold"].median())


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    cache_dir = output_dir / "cache"
    for directory in (results_dir, figures_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    sns.set_theme(style="whitegrid", context="notebook")
    features = build_feature_cache(
        data_dir=data_dir,
        cache_path=cache_dir / "audio_features.csv",
        n_jobs=args.n_jobs,
        force=args.force_features,
    )
    columns = numeric_feature_columns(features)
    train = features.loc[features["split"] == "train"].reset_index(drop=True)
    test = features.loc[features["split"] == "test"].reset_index(drop=True)
    X_train_raw = train[columns].replace([np.inf, -np.inf], np.nan)
    X_test_raw = test[columns].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)
    X_test = imputer.transform(X_test_raw)
    y = train["label"].astype(str).to_numpy()
    groups = train["source_group"].astype(str).to_numpy()

    group_probability, group_fold_rows, mean_importance = run_group_cv(
        X_train, y, groups, args.n_estimators, args.n_jobs
    )
    random_probability, random_fold_rows = run_random_cv(
        X_train, y, max(300, args.n_estimators // 2), args.n_jobs
    )
    group_prediction = np.asarray(LABEL_ORDER)[np.argmax(group_probability, axis=1)]
    random_prediction = np.asarray(LABEL_ORDER)[np.argmax(random_probability, axis=1)]

    group_macro_f1 = float(f1_score(y, group_prediction, labels=LABEL_ORDER, average="macro", zero_division=0))
    random_macro_f1 = float(f1_score(y, random_prediction, labels=LABEL_ORDER, average="macro", zero_division=0))
    group_balanced_accuracy = float(balanced_accuracy_score(y, group_prediction))
    random_balanced_accuracy = float(balanced_accuracy_score(y, random_prediction))

    fold_scores = pd.DataFrame(group_fold_rows + random_fold_rows)
    fold_scores.to_csv(results_dir / "cv_fold_scores.csv", index=False)
    report = pd.DataFrame(
        classification_report(
            y,
            group_prediction,
            labels=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        )
    ).T
    report.to_csv(results_dir / "classification_report_group_cv.csv")

    oof = train[["file_id", "label", "source_group", "duration_sec"]].copy()
    for index, label in enumerate(LABEL_ORDER):
        oof[f"prob_{label}"] = group_probability[:, index]
    oof["prediction"] = group_prediction
    oof["confidence"] = np.max(group_probability, axis=1)
    ordered_probability = np.sort(group_probability, axis=1)
    oof["margin"] = ordered_probability[:, -1] - ordered_probability[:, -2]
    oof["correct"] = oof["label"] == oof["prediction"]
    oof.to_csv(results_dir / "oof_predictions.csv", index=False)

    group_error = (
        oof.groupby(["label", "source_group"], as_index=False)
        .agg(
            recordings=("file_id", "size"),
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
        )
        .sort_values("accuracy")
    )
    group_error.to_csv(results_dir / "error_by_source_group.csv", index=False)
    oof["duration_quartile"] = pd.qcut(oof["duration_sec"], 4, duplicates="drop")
    duration_error = (
        oof.groupby("duration_quartile", observed=True, as_index=False)
        .agg(recordings=("file_id", "size"), accuracy=("correct", "mean"), mean_confidence=("confidence", "mean"))
    )
    duration_error["duration_quartile"] = duration_error["duration_quartile"].astype(str)
    duration_error.to_csv(results_dir / "error_by_duration.csv", index=False)

    importance = pd.DataFrame({"feature": columns, "importance": mean_importance}).sort_values(
        "importance", ascending=False
    )
    importance.to_csv(results_dir / "feature_importance.csv", index=False)

    pseudo_unknown, pseudo_threshold = pseudo_unknown_thresholds(
        X_train, y, groups, args.n_estimators, args.n_jobs
    )
    pseudo_unknown.to_csv(results_dir / "pseudo_unknown_validation.csv", index=False)
    known_rejection_cap_threshold = float(np.quantile(np.max(group_probability, axis=1), 0.10))
    selected_threshold = float(min(pseudo_threshold, known_rejection_cap_threshold))
    selected_known_rejection_rate = float(np.mean(np.max(group_probability, axis=1) < selected_threshold))

    final_model = make_model(args.n_estimators, args.n_jobs, SEED + 999)
    final_model.fit(X_train, y)
    test_probability = align_probabilities(final_model.predict_proba(X_test), final_model.classes_, LABEL_ORDER)
    test_prediction = np.asarray(LABEL_ORDER)[np.argmax(test_probability, axis=1)].astype(object)
    test_confidence = np.max(test_probability, axis=1)
    test_prediction[test_confidence < selected_threshold] = "Unknown"

    submission = pd.DataFrame({"file_id": test["file_id"], "label": test_prediction})
    if submission["file_id"].duplicated().any() or set(submission["file_id"]) != set(test["file_id"]):
        raise RuntimeError("Submission file_id integrity check failed")
    if not set(submission["label"]).issubset(ALL_SUBMISSION_LABELS):
        raise RuntimeError("Submission contains an invalid label")
    submission.to_csv(results_dir / "baseline_submission.csv", index=False)

    test_predictions = submission.copy()
    for index, label in enumerate(LABEL_ORDER):
        test_predictions[f"prob_{label}"] = test_probability[:, index]
    test_predictions["confidence"] = test_confidence
    test_predictions["rejected_as_unknown"] = test_predictions["label"] == "Unknown"
    test_predictions.to_csv(results_dir / "test_predictions_with_probabilities.csv", index=False)

    matrix = confusion_matrix(y, group_prediction, labels=LABEL_ORDER)
    plt.figure(figsize=(7, 5.6))
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER)
    ax.set(title=f"Confusion matrix 4-fold group CV (Macro F1={group_macro_f1:.3f})", xlabel="Prediksi", ylabel="Aktual")
    plt.tight_layout()
    plt.savefig(figures_dir / "cv_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 6.5))
    top_importance = importance.head(20).sort_values("importance")
    ax = sns.barplot(data=top_importance, x="importance", y="feature", color="#2a9d8f")
    ax.set(title="20 feature importance teratas (rata-rata group CV)", xlabel="Importance", ylabel="Fitur")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    plot_folds = fold_scores.copy()
    plot_folds["fold"] = plot_folds["fold"].astype(str)
    ax = sns.barplot(data=plot_folds, x="fold", y="macro_f1", hue="scheme")
    ax.set(title="Macro F1 per fold: group CV vs random CV", xlabel="Fold", ylabel="Macro F1", ylim=(0, 1))
    plt.tight_layout()
    plt.savefig(figures_dir / "cv_fold_scores.png", dpi=180, bbox_inches="tight")
    plt.close()

    metrics = {
        "seed": SEED,
        "model": "ExtraTreesClassifier",
        "model_parameters": final_model.get_params(),
        "feature_count": len(columns),
        "train_rows": len(train),
        "test_rows": len(test),
        "group_cv_folds": 4,
        "group_cv_macro_f1_known_classes": group_macro_f1,
        "group_cv_balanced_accuracy_known_classes": group_balanced_accuracy,
        "group_cv_fold_macro_f1_mean": float(pd.DataFrame(group_fold_rows)["macro_f1"].mean()),
        "group_cv_fold_macro_f1_std": float(pd.DataFrame(group_fold_rows)["macro_f1"].std(ddof=1)),
        "random_cv_macro_f1_known_classes": random_macro_f1,
        "random_cv_balanced_accuracy_known_classes": random_balanced_accuracy,
        "random_minus_group_macro_f1_gap": random_macro_f1 - group_macro_f1,
        "pseudo_unknown_threshold_median": pseudo_threshold,
        "known_rejection_cap_threshold_p10": known_rejection_cap_threshold,
        "selected_unknown_threshold": selected_threshold,
        "selected_threshold_known_oof_rejection_rate": selected_known_rejection_rate,
        "test_unknown_count": int(np.sum(test_prediction == "Unknown")),
        "test_unknown_fraction": float(np.mean(test_prediction == "Unknown")),
        "test_prediction_counts": pd.Series(test_prediction).value_counts().to_dict(),
        "important_caveat": (
            "No labeled Unknown examples exist. The Unknown threshold is a heuristic based on leave-one-species-out "
            "pseudo-unknown experiments and a 10% known-class OOF rejection cap; it is not an honest 4-class CV score."
        ),
    }
    save_json(metrics, results_dir / "baseline_metrics.json")
    print(f"Group CV known-class Macro F1: {group_macro_f1:.4f}", flush=True)
    print(f"Random CV known-class Macro F1: {random_macro_f1:.4f}", flush=True)
    print(f"Selected Unknown threshold: {selected_threshold:.4f}", flush=True)
    print(f"Predicted Unknown in test: {metrics['test_unknown_count']}/{len(test)}", flush=True)
    print(f"Submission: {results_dir / 'baseline_submission.csv'}", flush=True)


if __name__ == "__main__":
    main()

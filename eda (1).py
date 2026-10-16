"""Reproducible EDA for the Schematics NPC open-set bird audio competition."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DEPS = SCRIPT_DIR / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
import soundfile
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audio_features import SEED, build_feature_cache, numeric_feature_columns

LABEL_ORDER = ["Bondol Jawa", "Cendrawasih", "Kutilang"]
PALETTE = {"Bondol Jawa": "#2a9d8f", "Cendrawasih": "#e9c46a", "Kutilang": "#e76f51"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR.parent / "Data Schematics")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def save_json(payload: dict, path: Path) -> None:
    def convert(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value)}")

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=convert), encoding="utf-8")


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def file_inventory(root: Path) -> pd.DataFrame:
    rows = []
    for path in root.rglob("*"):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative.startswith("analysis_output/"):
                continue
            rows.append(
                {
                    "top_level": relative.split("/")[0],
                    "extension": path.suffix.lower() or "<none>",
                    "file_count": 1,
                    "bytes": path.stat().st_size,
                }
            )
    inventory = pd.DataFrame(rows)
    return (
        inventory.groupby(["top_level", "extension"], as_index=False)
        .agg(file_count=("file_count", "sum"), bytes=("bytes", "sum"))
        .sort_values(["top_level", "extension"])
    )


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash_audit(data_dir: Path, metadata: pd.DataFrame, sample: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    expected_rows = len(metadata) + len(sample)
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if len(cached) == expected_rows:
            print(f"Using content-hash cache: {cache_path}", flush=True)
            return cached
    records = []
    train_paths = zip(metadata["file_id"], metadata["audio_path"])
    test_paths = ((file_id, f"test/audio/{file_id}.flac") for file_id in sample["file_id"])
    for split, pairs in (("train", train_paths), ("test", test_paths)):
        for file_id, relative_path in pairs:
            records.append((split, str(file_id), str(relative_path)))
    rows = []
    print(f"Hashing {len(records)} files for exact-duplicate audit...", flush=True)
    for index, (split, file_id, relative_path) in enumerate(records, start=1):
        rows.append(
            {
                "split": split,
                "file_id": file_id,
                "relative_path": relative_path,
                "content_sha256": sha256_file(data_dir / relative_path),
            }
        )
        if index % 200 == 0 or index == len(records):
            print(f"  hashed {index}/{len(records)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(cache_path, index=False)
    return frame


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figures"
    results_dir = output_dir / "results"
    cache_dir = output_dir / "cache"
    for directory in (figures_dir, results_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    sns.set_theme(style="whitegrid", context="notebook")

    metadata = pd.read_csv(data_dir / "train" / "metadata.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    train_audio_ids = {path.stem for path in (data_dir / "train" / "audio").glob("*.flac")}
    test_audio_ids = {path.stem for path in (data_dir / "test" / "audio").glob("*.flac")}

    inventory = file_inventory(output_dir.parent)
    inventory.to_csv(results_dir / "file_inventory.csv", index=False)

    audit = {
        "train_rows": len(metadata),
        "test_rows": len(sample),
        "train_audio_files": len(train_audio_ids),
        "test_audio_files": len(test_audio_ids),
        "train_columns": metadata.columns.tolist(),
        "submission_columns": sample.columns.tolist(),
        "train_missing_by_column": metadata.isna().sum().to_dict(),
        "submission_missing_by_column": sample.isna().sum().to_dict(),
        "duplicate_train_rows": int(metadata.duplicated().sum()),
        "duplicate_train_file_ids": int(metadata["file_id"].duplicated().sum()),
        "duplicate_train_paths": int(metadata["audio_path"].duplicated().sum()),
        "duplicate_test_file_ids": int(sample["file_id"].duplicated().sum()),
        "train_test_file_id_overlap": len(set(metadata["file_id"]) & set(sample["file_id"])),
        "missing_referenced_train_audio": len(set(metadata["file_id"]) - train_audio_ids),
        "orphan_train_audio": len(train_audio_ids - set(metadata["file_id"])),
        "missing_referenced_test_audio": len(set(sample["file_id"]) - test_audio_ids),
        "orphan_test_audio": len(test_audio_ids - set(sample["file_id"])),
        "label_counts": metadata["label"].value_counts().to_dict(),
        "source_groups": int(metadata["source_group"].nunique()),
        "groups_per_label": metadata.groupby("label")["source_group"].nunique().to_dict(),
        "labels_per_group_distribution": metadata.groupby("source_group")["label"].nunique().value_counts().to_dict(),
    }
    content_hashes = content_hash_audit(data_dir, metadata, sample, cache_dir / "content_hashes.csv")
    duplicated_hash_mask = content_hashes["content_sha256"].duplicated(keep=False)
    duplicate_hashes = content_hashes.loc[duplicated_hash_mask].sort_values("content_sha256")
    cross_split_hash_counts = duplicate_hashes.groupby("content_sha256")["split"].nunique() if len(duplicate_hashes) else pd.Series(dtype=int)
    cross_split_duplicate_hashes = set(cross_split_hash_counts[cross_split_hash_counts > 1].index)
    duplicate_hashes.to_csv(results_dir / "exact_content_duplicates.csv", index=False)
    audit.update(
        {
            "unique_content_sha256": int(content_hashes["content_sha256"].nunique()),
            "files_in_exact_duplicate_sets": int(duplicated_hash_mask.sum()),
            "cross_split_exact_duplicate_hashes": len(cross_split_duplicate_hashes),
            "file_id_equals_encoded_file_sha256": int(
                (content_hashes["file_id"].str.lower() == content_hashes["content_sha256"]).sum()
            ),
        }
    )
    save_json(audit, results_dir / "data_audit.json")

    group_summary = (
        metadata.groupby(["label", "source_group"], as_index=False)
        .size()
        .rename(columns={"size": "recordings"})
        .sort_values(["label", "recordings"], ascending=[True, False])
    )
    group_summary.to_csv(results_dir / "source_group_summary.csv", index=False)

    plt.figure(figsize=(8, 4.8))
    ax = sns.countplot(data=metadata, x="label", order=LABEL_ORDER, hue="label", palette=PALETTE, legend=False)
    ax.set(title="Distribusi kelas data latih", xlabel="Label", ylabel="Jumlah rekaman")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)
    save_figure(figures_dir / "target_distribution.png")

    plt.figure(figsize=(10, 5.8))
    plot_groups = group_summary.copy()
    plot_groups["source_group_short"] = plot_groups["source_group"].str[:8]
    ax = sns.barplot(
        data=plot_groups,
        x="recordings",
        y="source_group_short",
        hue="label",
        hue_order=LABEL_ORDER,
        palette=PALETTE,
        dodge=False,
    )
    ax.set(title="Ukuran source_group dan kelasnya", xlabel="Jumlah rekaman", ylabel="source_group (8 karakter awal)")
    save_figure(figures_dir / "source_group_sizes.png")

    features = build_feature_cache(
        data_dir=data_dir,
        cache_path=cache_dir / "audio_features.csv",
        n_jobs=args.n_jobs,
        force=args.force_features,
    )
    feature_columns = numeric_feature_columns(features)
    train_features = features.loc[features["split"] == "train"].reset_index(drop=True)
    test_features = features.loc[features["split"] == "test"].reset_index(drop=True)

    quality_columns = [
        "duration_sec",
        "raw_rms",
        "raw_peak",
        "silence_fraction",
        "spectral_centroid_mean",
        "spectral_flatness_mean",
    ]
    quality_rows = []
    quality_flags = pd.DataFrame(index=train_features.index)
    for column in quality_columns:
        series = train_features[column]
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flag = (series < lower) | (series > upper)
        quality_flags[column] = flag
        quality_rows.append(
            {
                "feature": column,
                "minimum": float(series.min()),
                "q1": float(q1),
                "median": float(series.median()),
                "q3": float(q3),
                "maximum": float(series.max()),
                "iqr_lower_fence": float(lower),
                "iqr_upper_fence": float(upper),
                "iqr_outlier_count": int(flag.sum()),
            }
        )
    quality_summary = pd.DataFrame(quality_rows)
    quality_summary.to_csv(results_dir / "audio_quality_summary.csv", index=False)
    quality_outliers = train_features[["file_id", "label", "source_group"]].copy()
    quality_outliers["flagged_feature_count"] = quality_flags.sum(axis=1)
    quality_outliers["flagged_features"] = quality_flags.apply(
        lambda row: ",".join(row.index[row.to_numpy()].tolist()), axis=1
    )
    quality_outliers = quality_outliers.loc[quality_outliers["flagged_feature_count"] > 0].sort_values(
        ["flagged_feature_count", "file_id"], ascending=[False, True]
    )
    quality_outliers.to_csv(results_dir / "audio_quality_outliers.csv", index=False)

    audio_properties = {
        "duration_by_split": features.groupby("split")["duration_sec"].describe(
            percentiles=[0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
        ).round(6).to_dict(orient="index"),
        "sample_rate_counts": features.groupby(["split", "sample_rate"]).size().to_dict(),
        "channel_counts": features.groupby(["split", "channels"]).size().to_dict(),
        "subtype_counts": features.groupby(["split", "subtype"]).size().to_dict(),
        "numeric_feature_count": len(feature_columns),
        "feature_missing_total": int(features[feature_columns].isna().sum().sum()),
        "train_files_flagged_by_quality_iqr_rule": int(len(quality_outliers)),
        "duration_iqr_outlier_count": int(
            quality_summary.loc[quality_summary["feature"] == "duration_sec", "iqr_outlier_count"].iloc[0]
        ),
    }
    # JSON requires string keys for tuple-keyed dictionaries.
    audio_properties["sample_rate_counts"] = {str(k): v for k, v in audio_properties["sample_rate_counts"].items()}
    audio_properties["channel_counts"] = {str(k): v for k, v in audio_properties["channel_counts"].items()}
    audio_properties["subtype_counts"] = {str(k): v for k, v in audio_properties["subtype_counts"].items()}
    save_json(audio_properties, results_dir / "audio_properties.json")

    plt.figure(figsize=(9, 5.2))
    ax = sns.violinplot(
        data=train_features,
        x="label",
        y="duration_sec",
        order=LABEL_ORDER,
        hue="label",
        palette=PALETTE,
        legend=False,
        inner="quartile",
        cut=0,
    )
    ax.set(title="Durasi rekaman menurut kelas", xlabel="Label", ylabel="Durasi (detik)")
    save_figure(figures_dir / "duration_by_label.png")

    plt.figure(figsize=(9, 5.2))
    ax = sns.kdeplot(data=features, x="duration_sec", hue="split", common_norm=False, fill=True, alpha=0.3)
    ax.set(title="Perbandingan durasi train dan test", xlabel="Durasi (detik)", ylabel="Kepadatan")
    save_figure(figures_dir / "train_test_duration_shift.png")

    X_all = features[feature_columns].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_all)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    domain_y = (features["split"] == "test").astype(int).to_numpy()
    domain_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2_000, random_state=SEED),
    )
    domain_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    domain_probability = cross_val_predict(
        domain_model,
        X_imputed,
        domain_y,
        cv=domain_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    domain_auc = float(roc_auc_score(domain_y, domain_probability))

    shift_rows = []
    n_train = len(train_features)
    for index, column in enumerate(feature_columns):
        train_values = X_imputed[:n_train, index]
        test_values = X_imputed[n_train:, index]
        pooled_std = float(np.std(X_imputed[:, index]))
        standardized_mean_difference = (
            float(np.mean(test_values) - np.mean(train_values)) / pooled_std if pooled_std > 0 else 0.0
        )
        ks = ks_2samp(train_values, test_values, method="auto")
        shift_rows.append(
            {
                "feature": column,
                "train_mean": float(np.mean(train_values)),
                "test_mean": float(np.mean(test_values)),
                "standardized_mean_difference": standardized_mean_difference,
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
    shift = pd.DataFrame(shift_rows).sort_values("ks_statistic", ascending=False)
    shift.to_csv(results_dir / "train_test_shift.csv", index=False)

    plt.figure(figsize=(9, 6.5))
    top_shift = shift.head(15).sort_values("ks_statistic")
    ax = sns.barplot(data=top_shift, x="ks_statistic", y="feature", color="#457b9d")
    ax.set(title=f"Fitur dengan shift train–test terbesar (domain AUC={domain_auc:.3f})", xlabel="Statistik KS", ylabel="Fitur")
    save_figure(figures_dir / "train_test_shift_top_features.png")

    pca = PCA(n_components=2, random_state=SEED)
    embedding = pca.fit_transform(X_scaled)
    pca_frame = pd.DataFrame(
        {
            "pc1": embedding[:, 0],
            "pc2": embedding[:, 1],
            "split": features["split"].to_numpy(),
        }
    )
    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(data=pca_frame, x="pc1", y="pc2", hue="split", alpha=0.55, s=24)
    ax.set(
        title=f"PCA fitur audio: train vs test ({100*pca.explained_variance_ratio_.sum():.1f}% variansi)",
        xlabel="PC1",
        ylabel="PC2",
    )
    save_figure(figures_dir / "pca_train_test.png")

    y_encoded = pd.Categorical(train_features["label"], categories=LABEL_ORDER).codes
    f_values, _ = f_classif(X_imputed[:n_train], y_encoded)
    top_indices = np.argsort(np.nan_to_num(f_values, nan=-np.inf))[-20:]
    top_columns = [feature_columns[index] for index in top_indices]
    corr = pd.DataFrame(X_imputed[:n_train, top_indices], columns=top_columns).corr()
    plt.figure(figsize=(13, 10))
    ax = sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8})
    ax.set(title="Korelasi 20 fitur paling terkait dengan label (ANOVA F)")
    save_figure(figures_dir / "feature_correlation_heatmap.png")

    # Approximate cross-split duplicate/overlap audit in standardized feature space.
    neighbors = NearestNeighbors(n_neighbors=1, metric="cosine")
    neighbors.fit(X_scaled[:n_train])
    distances, indices = neighbors.kneighbors(X_scaled[n_train:])
    nearest = pd.DataFrame(
        {
            "test_file_id": test_features["file_id"],
            "nearest_train_file_id": train_features.iloc[indices[:, 0]]["file_id"].to_numpy(),
            "nearest_train_label": train_features.iloc[indices[:, 0]]["label"].to_numpy(),
            "cosine_distance": distances[:, 0],
        }
    ).sort_values("cosine_distance")
    nearest.to_csv(results_dir / "nearest_train_test_pairs.csv", index=False)

    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "seed": SEED,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": sns.__version__,
        "soundfile": soundfile.__version__,
    }
    save_json(runtime, results_dir / "runtime_versions.json")
    summary = {
        **audit,
        "domain_classifier_auc": domain_auc,
        "pca_explained_variance_2d": float(pca.explained_variance_ratio_.sum()),
        "nearest_cross_split_cosine_distance_min": float(nearest["cosine_distance"].min()),
        "nearest_cross_split_cosine_distance_p01": float(nearest["cosine_distance"].quantile(0.01)),
        "top_shift_features": shift.head(10)[["feature", "ks_statistic", "standardized_mean_difference"]].to_dict(orient="records"),
    }
    save_json(summary, results_dir / "eda_summary.json")
    print(f"EDA complete. Domain classifier AUC: {domain_auc:.4f}", flush=True)
    print(f"Figures: {figures_dir}", flush=True)


if __name__ == "__main__":
    main()

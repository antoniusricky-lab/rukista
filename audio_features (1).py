"""Shared, deterministic audio feature extraction for the Schematics NPC data.

The module intentionally avoids pretrained weights and external data. Features are
computed from each complete FLAC file after deterministic mono conversion and
resampling to 16 kHz.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.fft import dct
from scipy.signal import resample_poly

SEED = 42
TARGET_SR = 16_000
N_FFT = 1_024
HOP = 512
N_MELS = 48
N_MFCC = 20


def _hz_to_mel(freq: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


@lru_cache(maxsize=8)
def mel_filterbank(
    sample_rate: int = TARGET_SR,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = 200.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Return a triangular mel filterbank with shape (n_mels, n_fft // 2 + 1)."""
    if fmax is None:
        fmax = sample_rate / 2.0
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for idx in range(1, n_mels + 1):
        left, center, right = bins[idx - 1], bins[idx], bins[idx + 1]
        if center <= left:
            center = min(left + 1, n_fft // 2)
        if right <= center:
            right = min(center + 1, n_fft // 2)
        if center > left:
            fb[idx - 1, left:center] = np.linspace(0.0, 1.0, center - left, endpoint=False)
        if right > center:
            fb[idx - 1, center:right] = np.linspace(1.0, 0.0, right - center, endpoint=False)
    norm = fb.sum(axis=1, keepdims=True)
    fb /= np.maximum(norm, 1e-12)
    return fb


def _summary(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"{prefix}_{name}": np.nan for name in ("mean", "std", "p10", "p50", "p90")}
    q10, q50, q90 = np.quantile(finite, [0.10, 0.50, 0.90])
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_p10": float(q10),
        f"{prefix}_p50": float(q50),
        f"{prefix}_p90": float(q90),
    }


def _load_mono(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    channels = int(audio.shape[1])
    mono = np.mean(audio, axis=1, dtype=np.float32)
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    if sr != target_sr:
        divisor = math.gcd(int(sr), int(target_sr))
        mono = resample_poly(mono, target_sr // divisor, sr // divisor).astype(np.float32)
    return mono, int(sr), channels


def extract_audio_features(path: Path) -> dict[str, float | int | str]:
    """Extract compact waveform, spectral, log-mel, and MFCC summaries."""
    info = sf.info(path)
    y, original_sr, channels = _load_mono(path)
    eps = 1e-12
    raw_rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64) + eps))
    raw_peak = float(np.max(np.abs(y))) if y.size else 0.0
    raw_abs_median = float(np.median(np.abs(y))) if y.size else 0.0
    silence_fraction = float(np.mean(np.abs(y) < max(raw_rms * 0.10, 1e-5))) if y.size else 1.0

    y = y - float(np.mean(y))
    emphasized = np.empty_like(y)
    emphasized[0] = y[0] if y.size else 0.0
    if y.size > 1:
        emphasized[1:] = y[1:] - 0.97 * y[:-1]
    if emphasized.size < N_FFT:
        emphasized = np.pad(emphasized, (0, N_FFT - emphasized.size))

    n_frames = 1 + (emphasized.size - N_FFT) // HOP
    frame_starts = np.arange(n_frames) * HOP
    frames = np.stack([emphasized[start : start + N_FFT] for start in frame_starts])
    raw_frames = np.stack([y[start : start + N_FFT] for start in frame_starts])
    window = np.hanning(N_FFT).astype(np.float32)
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float64) + eps
    magnitude = np.sqrt(power)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / TARGET_SR)
    power_sum = power.sum(axis=1) + eps

    centroid = (power * freqs).sum(axis=1) / power_sum
    bandwidth = np.sqrt((power * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / power_sum)
    cumulative = np.cumsum(power, axis=1)
    rolloff_idx = np.argmax(cumulative >= 0.85 * cumulative[:, -1, None], axis=1)
    rolloff = freqs[rolloff_idx]
    flatness = np.exp(np.mean(np.log(power), axis=1)) / (np.mean(power, axis=1) + eps)
    normalized_power = power / power_sum[:, None]
    entropy = -np.sum(normalized_power * np.log(normalized_power + eps), axis=1) / np.log(power.shape[1])
    rms_frames = np.sqrt(np.mean(np.square(raw_frames), axis=1) + eps)
    zcr = np.mean(np.diff(np.signbit(raw_frames), axis=1), axis=1)
    norm_mag = magnitude / (np.linalg.norm(magnitude, axis=1, keepdims=True) + eps)
    flux = np.concatenate([[0.0], np.sqrt(np.sum(np.diff(norm_mag, axis=0) ** 2, axis=1))])

    mel_energy = power @ mel_filterbank().T
    log_mel = np.log(mel_energy + 1e-10)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :N_MFCC]

    result: dict[str, float | int | str] = {
        "duration_sec": float(info.duration),
        "sample_rate": int(original_sr),
        "channels": channels,
        "frames_original": int(info.frames),
        "file_size_bytes": int(path.stat().st_size),
        "raw_rms": raw_rms,
        "raw_peak": raw_peak,
        "raw_abs_median": raw_abs_median,
        "silence_fraction": silence_fraction,
        "format": str(info.format),
        "subtype": str(info.subtype),
    }
    for prefix, values in (
        ("frame_rms", rms_frames),
        ("zcr", zcr),
        ("spectral_centroid", centroid),
        ("spectral_bandwidth", bandwidth),
        ("spectral_rolloff85", rolloff),
        ("spectral_flatness", flatness),
        ("spectral_entropy", entropy),
        ("spectral_flux", flux),
    ):
        result.update(_summary(prefix, values))

    band_edges = [0, 250, 500, 1_000, 2_000, 4_000, 6_000, 8_001]
    total_energy = power.sum(axis=1) + eps
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        ratio = power[:, mask].sum(axis=1) / total_energy
        result.update(_summary(f"band_{low}_{high}_ratio", ratio))

    for idx in range(N_MFCC):
        result.update(_summary(f"mfcc_{idx:02d}", mfcc[:, idx]))
    for idx in range(N_MELS):
        result[f"logmel_{idx:02d}_mean"] = float(np.mean(log_mel[:, idx]))
        result[f"logmel_{idx:02d}_std"] = float(np.std(log_mel[:, idx]))

    # Preserve coarse temporal structure without producing per-frame training rows.
    for segment_idx, frame_idx in enumerate(np.array_split(np.arange(n_frames), 3)):
        segment = mfcc[frame_idx]
        for mfcc_idx in range(13):
            result[f"segment_{segment_idx}_mfcc_{mfcc_idx:02d}_mean"] = float(np.mean(segment[:, mfcc_idx]))
    return result


def competition_records(data_dir: Path) -> pd.DataFrame:
    train = pd.read_csv(data_dir / "train" / "metadata.csv")
    train_records = train.assign(
        split="train",
        relative_path=train["audio_path"].astype(str),
    )[["split", "file_id", "relative_path", "label", "source_group"]]
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    test_records = pd.DataFrame(
        {
            "split": "test",
            "file_id": sample["file_id"].astype(str),
            "relative_path": "test/audio/" + sample["file_id"].astype(str) + ".flac",
            "label": pd.NA,
            "source_group": pd.NA,
        }
    )
    return pd.concat([train_records, test_records], ignore_index=True)


def build_feature_cache(
    data_dir: Path,
    cache_path: Path,
    n_jobs: int = 4,
    force: bool = False,
) -> pd.DataFrame:
    """Extract all file-level features, with a reusable CSV cache."""
    if cache_path.exists() and not force:
        cached = pd.read_csv(cache_path, low_memory=False)
        expected = competition_records(data_dir)
        if len(cached) == len(expected) and set(cached["file_id"]) == set(expected["file_id"]):
            print(f"Using feature cache: {cache_path}", flush=True)
            return cached

    records = competition_records(data_dir)
    missing = [rel for rel in records["relative_path"] if not (data_dir / rel).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} referenced audio files; first: {missing[0]}")

    n_jobs = max(1, min(int(n_jobs), os.cpu_count() or 1))
    print(f"Extracting features from {len(records)} audio files with {n_jobs} workers...", flush=True)
    tasks: Iterable[dict[str, float | int | str]] = joblib.Parallel(
        n_jobs=n_jobs,
        prefer="threads",
        return_as="generator",
        batch_size=8,
    )(
        joblib.delayed(extract_audio_features)(data_dir / row.relative_path)
        for row in records.itertuples(index=False)
    )
    extracted: list[dict[str, float | int | str]] = []
    for idx, features in enumerate(tasks, start=1):
        extracted.append(features)
        if idx % 100 == 0 or idx == len(records):
            print(f"  processed {idx}/{len(records)}", flush=True)
    feature_frame = pd.concat([records.reset_index(drop=True), pd.DataFrame(extracted)], axis=1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.to_csv(cache_path, index=False)
    print(f"Saved feature cache: {cache_path}", flush=True)
    return feature_frame


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"split", "file_id", "relative_path", "label", "source_group", "format", "subtype"}
    return [column for column in frame.columns if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])]

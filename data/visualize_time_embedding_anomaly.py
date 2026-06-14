#!/usr/bin/env python3
"""
Anomaly detection + visualization using time-embedding representations.

Steps:
1) Read raw CSV (label is the last column by default).
2) Annotate anomaly info (unknown by default) using anomaly_reasoning spec.
3) Learn time-embedding representations via a small GRU forecasting head.
4) Visualize counts, anomaly scores, and PCA embeddings.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from anomaly_reasoning import (
    _load_spec,
    _parse_base_label_arg,
    _resolve_base_labels,
    compute_baseline_stats,
    annotate_chunk,
)
from models.define_model import TimeEmbedding


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_time_col(df: pd.DataFrame, spec_time: str | None, override: str | None) -> str | None:
    if override:
        if override in df.columns:
            return override
        return None
    if spec_time and spec_time in df.columns:
        return spec_time
    if "second" in df.columns:
        return "second"
    if "time" in df.columns:
        return "time"
    return None


def _build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray | None,
    seq_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    total = len(features)
    if total < seq_len:
        raise ValueError(f"Not enough rows ({total}) for sequence_length={seq_len}")
    sequences = []
    seq_labels = []
    seq_times = []
    end_indices = []
    for start in range(0, total - seq_len + 1, stride):
        end = start + seq_len
        sequences.append(features[start:end])
        seq_labels.append(labels[end - 1])
        end_indices.append(end - 1)
        if times is not None:
            seq_times.append(times[end - 1])
    seq_times_arr = np.array(seq_times) if times is not None else None
    return np.stack(sequences), np.array(seq_labels), seq_times_arr, np.array(end_indices)


class TimeEmbedForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        time_emb_dim: int = 8,
        time_emb_type: str = "time2vec",
        time_emb_activation: str = "sin",
        time_emb_normalize: bool = True,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.time_embed = TimeEmbedding(
            max_len=seq_len,
            dim=time_emb_dim,
            emb_type=time_emb_type,
            activation=time_emb_activation,
            normalize=time_emb_normalize,
        )
        self.gru = nn.GRU(
            input_size=input_dim + time_emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, F]
        time_emb = self.time_embed(
            seq_len=x.size(1),
            batch_size=x.size(0),
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, time_emb], dim=-1)
        _, h = self.gru(x)
        h_last = h[-1]
        pred = self.head(h_last)
        return pred, h_last


def _standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std_safe = np.where(std == 0.0, 1.0, std)
    return ((features - mean) / std_safe).astype(np.float32, copy=False)


def _plot_counts(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="anomaly_type", order=df["anomaly_type"].value_counts().index)
    plt.title("Anomaly Type Counts")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "anomaly_type_counts.png"), dpi=200)
    plt.close()


def _plot_score_hist(scores: np.ndarray, labels: List[str], out_dir: str) -> None:
    plt.figure(figsize=(7, 4))
    for label in sorted(set(labels)):
        mask = np.array(labels) == label
        if mask.sum() == 0:
            continue
        sns.kdeplot(scores[mask], label=label, fill=True, alpha=0.3)
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Reconstruction MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "anomaly_score_distribution.png"), dpi=200)
    plt.close()


def _plot_score_over_time(times: np.ndarray, scores: np.ndarray, out_dir: str) -> None:
    if times is None:
        return
    order = np.argsort(times)
    times_sorted = times[order]
    scores_sorted = scores[order]
    plt.figure(figsize=(8, 4))
    plt.plot(times_sorted, scores_sorted, linewidth=0.8, alpha=0.8)
    plt.title("Anomaly Score Over Time")
    plt.xlabel("Time")
    plt.ylabel("Reconstruction MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "anomaly_score_over_time.png"), dpi=200)
    plt.close()


def _plot_pca(embeddings: np.ndarray, labels: List[str], out_dir: str) -> None:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": labels})
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="x", y="y", hue="label", s=18, alpha=0.7)
    plt.title("PCA of Time-Embedding Representations")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_time_embedding.png"), dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-embedding anomaly visualization")
    parser.add_argument(
        "--input",
        default="../../data/temporal_splits_routing_normal/raw_combined.csv",
        help="Input raw CSV (relative to IPv6_detect_attack_by_AI/)",
    )
    parser.add_argument(
        "--spec",
        default="../../data/temporal_splits_routing_normal/anomaly_reasoning_binary.yaml",
        help="Anomaly reasoning spec YAML",
    )
    parser.add_argument("--out-dir", default="outputs/time_embedding_viz")
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--time-emb-dim", type=int, default=8)
    parser.add_argument("--time-emb-type", default="time2vec", choices=["learned", "time2vec"])
    parser.add_argument("--time-emb-activation", default="sin", choices=["sin", "cos"])
    parser.add_argument("--no-time-emb-normalize", action="store_true", help="Disable time2vec normalization")
    parser.add_argument("--train-all", action="store_true", help="Train on all sequences (not only normal)")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    input_path = args.input if os.path.isabs(args.input) else os.path.join(base_dir, args.input)
    spec_path = args.spec if os.path.isabs(args.spec) else os.path.join(base_dir, args.spec)
    out_dir = args.out_dir
    _ensure_dir(out_dir)

    spec = _load_spec(spec_path, None)
    df_head = pd.read_csv(input_path, nrows=1)
    label_col = args.label_col or df_head.columns[-1]
    if label_col not in df_head.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV")

    time_col = _resolve_time_col(df_head, spec.get("time_col"), args.time_col)
    drop_cols = [label_col]
    feature_cols = [c for c in df_head.columns if c not in drop_cols]

    base_label_arg = _parse_base_label_arg("auto")
    base_labels = _resolve_base_labels(base_label_arg, spec, spec["label_map"])

    mean, std = compute_baseline_stats(
        input_path,
        feature_cols=feature_cols,
        label_col=label_col,
        base_labels=base_labels,
        chunksize=50000,
    )

    df = pd.read_csv(input_path)
    annotated = annotate_chunk(
        chunk=df,
        feature_cols=feature_cols,
        label_col=label_col,
        time_col=time_col,
        label_map=spec["label_map"],
        mean=mean,
        std=std,
        top_k=3,
        z_thresh=2.5,
        anomaly_type_map=spec["anomaly_types"],
        anomaly_family_map=spec["anomaly_families"],
        feature_hints_map=spec["feature_hints"],
        default_anomaly_type=spec["default_anomaly_type"],
        default_anomaly_family=spec["default_anomaly_family"],
        normal_labels=spec["normal_labels"],
    )
    annotated_path = os.path.join(out_dir, "raw_combined_annotated.csv")
    annotated.to_csv(annotated_path, index=False)

    features = annotated[feature_cols].to_numpy(dtype=np.float32)
    labels = annotated[label_col].to_numpy()
    times = annotated[time_col].to_numpy() if time_col else None
    features = _standardize(features, mean, std)

    sequences, seq_labels, seq_times, end_indices = _build_sequences(
        features, labels, times, args.sequence_length, args.stride
    )

    normal_mask = seq_labels == 0
    if not args.train_all and normal_mask.any():
        train_sequences = sequences[normal_mask]
    else:
        train_sequences = sequences

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeEmbedForecaster(
        input_dim=sequences.shape[-1],
        seq_len=args.sequence_length,
        time_emb_dim=args.time_emb_dim,
        time_emb_type=args.time_emb_type,
        time_emb_activation=args.time_emb_activation,
        time_emb_normalize=not args.no_time_emb_normalize,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(train_sequences))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            pred, _ = model(batch)
            target = batch[:, -1, :]
            loss = criterion(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * batch.size(0)
        avg_loss = running / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.6f}")

    model.eval()
    all_emb = []
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(sequences), args.batch_size):
            batch = torch.from_numpy(sequences[i : i + args.batch_size]).to(device)
            pred, emb = model(batch)
            target = batch[:, -1, :]
            mse = torch.mean((pred - target) ** 2, dim=1)
            all_emb.append(emb.cpu().numpy())
            all_scores.append(mse.cpu().numpy())
    embeddings = np.concatenate(all_emb, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    anomaly_type_seq = []
    if "anomaly_type" in annotated.columns:
        anomaly_type_seq = annotated["anomaly_type"].iloc[end_indices].tolist()
    else:
        anomaly_type_seq = ["unknown" if int(v) != 0 else "none" for v in seq_labels]

    embed_df = pd.DataFrame(embeddings)
    embed_df["label"] = seq_labels
    embed_df["anomaly_type"] = anomaly_type_seq
    embed_df["anomaly_score"] = scores
    if seq_times is not None:
        embed_df["time"] = seq_times
    embed_df.to_csv(os.path.join(out_dir, "sequence_embeddings.csv"), index=False)

    _plot_counts(annotated, out_dir)
    _plot_score_hist(scores, anomaly_type_seq, out_dir)
    _plot_score_over_time(seq_times, scores, out_dir)
    _plot_pca(embeddings, anomaly_type_seq, out_dir)

    print(f"Annotated CSV: {annotated_path}")
    print(f"Embeddings CSV: {os.path.join(out_dir, 'sequence_embeddings.csv')}")
    print(f"Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()

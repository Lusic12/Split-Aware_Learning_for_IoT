"""Smoke test: time-aware distribution matching on sequence windows.

This is a lightweight prototype to verify that time dependence can be
captured by matching sequence embeddings (GRU encoder) instead of treating
rows as independent samples.
"""

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.dataloader import SequenceDataset
from models.define_model import build_time_embedding


@dataclass
class DistillConfig:
    seq_len: int = 20
    stride: int = 10
    ipc: int = 50
    iterations: int = 200
    batch_real: int = 128
    hidden_dim: int = 64
    embed_dim: int = 32
    lr: float = 0.1
    use_time_embedding: bool = True


class SeqEncoder(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=64, embed_dim=32, time_cfg=None):
        super().__init__()
        self.seq_len = seq_len
        self.time_embed, self.time_dim = build_time_embedding(time_cfg, seq_len)
        self.gru = nn.GRU(
            input_dim + self.time_dim,
            hidden_dim,
            batch_first=True,
        )
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        if self.time_embed is not None:
            time_emb = self.time_embed(
                seq_len=x.size(1),
                batch_size=x.size(0),
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, time_emb], dim=-1)
        out, _ = self.gru(x)
        return self.proj(out[:, -1, :])


class TimeDistiller:
    def __init__(self, input_dim, num_classes, cfg: DistillConfig, device):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.cfg = cfg
        self.device = device
        self.S = None
        self.sy = None
        self.optimizer = None

    def _init_synthetic(self, X_seq, y_seq):
        rng = torch.Generator(device=self.device).manual_seed(42)
        synth = []
        for c in range(self.num_classes):
            idx = torch.where(y_seq == c)[0]
            if len(idx) == 0:
                raise ValueError(f"No samples for class {c}")
            pick = idx[torch.randperm(len(idx), generator=rng)[: self.cfg.ipc]]
            synth.append(X_seq[pick].clone())
        self.S = torch.cat(synth, dim=0).to(self.device)
        self.S.requires_grad = True
        self.sy = torch.cat(
            [torch.full((self.cfg.ipc,), c, dtype=torch.long) for c in range(self.num_classes)]
        ).to(self.device)
        self.optimizer = torch.optim.SGD([self.S], lr=self.cfg.lr, momentum=0.5)

    def distill(self, X_seq, y_seq):
        X_seq = X_seq.to(self.device)
        y_seq = y_seq.to(self.device)
        self._init_synthetic(X_seq, y_seq)

        class_indices = {c: torch.where(y_seq == c)[0] for c in range(self.num_classes)}
        time_cfg = {
            "enabled": self.cfg.use_time_embedding,
            "type": "time2vec",
            "dim": 8,
            "activation": "sin",
            "normalize": True,
        }

        for it in range(self.cfg.iterations):
            encoder = SeqEncoder(
                self.input_dim,
                self.cfg.seq_len,
                hidden_dim=self.cfg.hidden_dim,
                embed_dim=self.cfg.embed_dim,
                time_cfg=time_cfg,
            ).to(self.device)
            encoder.train()
            for p in encoder.parameters():
                p.requires_grad = False

            loss = torch.tensor(0.0, device=self.device)
            for c in range(self.num_classes):
                idx = class_indices[c]
                if len(idx) == 0:
                    continue
                if len(idx) > self.cfg.batch_real:
                    batch_idx = idx[torch.randperm(len(idx))[: self.cfg.batch_real]]
                else:
                    batch_idx = idx
                real = X_seq[batch_idx]
                synth = self.S[c * self.cfg.ipc : (c + 1) * self.cfg.ipc]

                emb_real = encoder(real).detach()
                emb_synth = encoder(synth)
                mean_real = emb_real.mean(dim=0)
                mean_synth = emb_synth.mean(dim=0)
                loss = loss + (mean_real - mean_synth).pow(2).sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if it % max(1, self.cfg.iterations // 10) == 0:
                print(f"Iter {it}/{self.cfg.iterations} | Loss={loss.item():.6f}")

        return self.S.detach().cpu(), self.sy.detach().cpu()


def build_sequences(csv_path, seq_len, stride, max_rows=None):
    df = pd.read_csv(csv_path, nrows=max_rows)
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].to_numpy(dtype=np.int64)
    dataset = SequenceDataset(X, y, seq_len, stride)
    if len(dataset) == 0:
        raise ValueError("SequenceDataset is empty. Check seq_len/stride or data size.")

    seq_list = []
    label_list = []
    for i in range(len(dataset)):
        seq, label = dataset[i]
        seq_list.append(seq.numpy())
        label_list.append(int(label))

    X_seq = torch.tensor(np.stack(seq_list), dtype=torch.float32)
    y_seq = torch.tensor(label_list, dtype=torch.long)
    return X_seq, y_seq


def main():
    parser = argparse.ArgumentParser(description="Time-aware data distillation smoke test")
    parser.add_argument("--data-path", required=True, help="CSV path (label is last column)")
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--ipc", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batch-real", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--no-time-embed", action="store_true")
    parser.add_argument("--save-npz", default=None, help="Optional output .npz path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_seq, y_seq = build_sequences(args.data_path, args.seq_len, args.stride, args.max_rows)
    num_classes = int(torch.unique(y_seq).numel())

    cfg = DistillConfig(
        seq_len=args.seq_len,
        stride=args.stride,
        ipc=args.ipc,
        iterations=args.iterations,
        batch_real=args.batch_real,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        lr=args.lr,
        use_time_embedding=not args.no_time_embed,
    )

    distiller = TimeDistiller(X_seq.shape[-1], num_classes, cfg, device)
    S, sy = distiller.distill(X_seq, y_seq)

    print("Done.")
    print(f"Synthetic sequences: {S.shape}")
    print(f"Labels: {sy.shape}")

    if args.save_npz:
        np.savez(args.save_npz, X=S.numpy(), y=sy.numpy())
        print(f"Saved synthetic sequences to {args.save_npz}")


if __name__ == "__main__":
    main()

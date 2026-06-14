"""Smoke test: time-dependent model distillation path (sequence + time embedding).

Runs a single forward/backward step to verify shapes and loss wiring.
"""

import argparse

import torch
import yaml

from data.dataloader import create_dataloaders
from models.define_model import create_model
from support.dl_helpers import DistillationLoss


def main():
    parser = argparse.ArgumentParser(description="Time-dependent model distillation smoke test")
    parser.add_argument("--config", default="configs/Deep_learning.yaml")
    parser.add_argument("--student", default="gru")
    parser.add_argument("--teacher", default="gru")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-time-embed", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data_config", {})
    seq_len = args.seq_len or data_cfg.get("sequence_length", 20)

    # Force sequence dataloader for time-dependent path
    train_loader, _, _ = create_dataloaders(
        args.config,
        normalize=False,
        force_sequence=True,
    )

    # Override batch size locally for quick smoke test
    batch = next(iter(train_loader))
    inputs, labels = batch
    inputs = inputs[: args.batch_size].to(device)
    labels = labels[: args.batch_size].to(device)

    input_dim = inputs.size(-1)
    all_labels = train_loader.dataset.labels
    num_classes = int(torch.unique(all_labels).numel())

    time_cfg = {
        "enabled": not args.no_time_embed,
        "type": "time2vec",
        "dim": 8,
        "activation": "sin",
        "normalize": True,
    }

    student = create_model(
        args.student,
        input_dim,
        num_classes,
        hidden_layer_list=[64, 32],
        sequence_length=seq_len,
        config=config,
        strict_input=True,
        time_embedding=time_cfg,
    ).to(device)

    teacher = create_model(
        args.teacher,
        input_dim,
        num_classes,
        hidden_layer_list=[128, 64],
        sequence_length=seq_len,
        config=config,
        strict_input=True,
        time_embedding=time_cfg,
    ).to(device)

    student.train()
    teacher.eval()

    with torch.no_grad():
        t_logits = teacher(inputs)
    s_logits = student(inputs)

    kd = DistillationLoss(temperature=4.0, alpha=0.5)
    loss = kd(s_logits, t_logits, labels)
    loss.backward()

    print("OK")
    print(f"inputs: {tuple(inputs.shape)}")
    print(f"student logits: {tuple(s_logits.shape)}")
    print(f"teacher logits: {tuple(t_logits.shape)}")
    print(f"loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()

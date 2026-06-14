from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from utils.function import count_parameters, plot_training_history


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _resolve_device(model, device):
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def plot_confusion_matrix(y_true, y_pred, class_names, dataset_type, model_name, out_dir="img"):
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        class_names = [str(l) for l in unique_labels]

    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(f"\nConfusion Matrix - {dataset_type.capitalize()} - {model_name.upper()}:")
    print(df_cm)

    print(f"Macro F1 Score: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"PR-AUC Score: {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", xticklabels="auto", yticklabels="auto")
    plt.xlabel("PREDICTED")
    plt.ylabel("TRUE")
    plt.title(f"CONFUSION MATRIX - {dataset_type.upper()} - {model_name.upper()}")

    ensure_dir(out_dir)
    fp = os.path.join(out_dir, f"dl_confusion_matrix_{dataset_type}_{model_name}.png")
    plt.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close()
    return fp


def evaluate_model(
    model,
    loader,
    criterion,
    class_names=None,
    dataset_type="test",
    model_name="model",
    device=None,
    out_dir="img",
):
    model.eval()
    device = _resolve_device(model, device)
    total_loss = 0.0
    total = 0
    correct = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {dataset_type}"):
            if isinstance(batch, dict):
                inputs = batch.get("sequence")
                labels = batch.get("label")
            else:
                inputs, labels = batch
            if inputs is None or labels is None:
                continue

            inputs = inputs.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    precision = precision_score(all_targets, all_predictions, average="macro", zero_division=0)
    recall = recall_score(all_targets, all_predictions, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0)

    print(f"\n{dataset_type.capitalize()} Results:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    try:
        plot_confusion_matrix(all_targets, all_predictions, class_names, dataset_type, model_name, out_dir=out_dir)
    except Exception as exc:
        print(f"Failed to plot confusion matrix: {exc}")

    print(f"\nClassification Report - {dataset_type.capitalize()}:")
    try:
        print(classification_report(all_targets, all_predictions, target_names=class_names, zero_division=0))
    except Exception:
        print(classification_report(all_targets, all_predictions, zero_division=0))

    return avg_loss, accuracy, precision, recall, f1


def train_model(
    model_name,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs=100,
    class_names=None,
    save_path="weights",
    best_model_path=None,
    device=None,
):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience = 5
    trigger_times = 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if best_model_path is None:
        best_model_path = os.path.join(save_path, f"best_{model_name}_model.pth")
    best_model_dir = os.path.dirname(best_model_path)
    if best_model_dir and not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir, exist_ok=True)

    device = _resolve_device(model, device)

    print(f"\nStarting training {model_name.upper()} for {epochs} epochs...")
    print(f"Device: {device}")
    print(f"Total trainable parameters: {count_parameters(model):,}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for data in train_bar:
            if isinstance(data, dict):
                inputs = data["sequence"]
                target = data["label"]
            else:
                inputs, target = data

            inputs = inputs.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predicted = torch.argmax(outputs, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            current_acc = 100 * correct / total
            train_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{current_acc:.2f}%",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        val_loss, val_acc, _, _, _ = evaluate_model(
            model,
            val_loader,
            criterion,
            class_names,
            "validation",
            model_name,
            device=device,
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch [{epoch+1}/{epochs}]:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            trigger_times = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "config": model_name,
            }, best_model_path)

            print(f"New best model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        else:
            trigger_times += 1
            print(f"No improvement for {trigger_times} epochs")

            if trigger_times >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("-" * 60)

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    plot_training_history(history, model_name=model_name)

    return history, best_val_loss, best_val_acc


class DistillationLoss(nn.Module):
    """Combine student CE (or focal) loss with KL divergence to teacher logits."""
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.3,
        class_weights=None,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.T = temperature
        self.alpha = alpha
        self.use_focal = use_focal_loss
        self.gamma = focal_gamma
        self.class_weights = class_weights
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_outputs, teacher_outputs, labels):
        soft_teacher = F.softmax(teacher_outputs / self.T, dim=1)
        log_soft_student = F.log_softmax(student_outputs / self.T, dim=1)

        distill_loss = self.kl(log_soft_student, soft_teacher) * (self.T ** 2)
        if self.use_focal:
            ce = F.cross_entropy(student_outputs, labels, weight=self.class_weights, reduction="none")
            pt = torch.exp(-ce)
            student_loss = ((1 - pt) ** self.gamma * ce).mean()
        else:
            student_loss = self.ce(student_outputs, labels)
        return self.alpha * student_loss + (1.0 - self.alpha) * distill_loss


def train_student_model(
    student_model,
    teacher_model,
    train_loader,
    val_loader,
    kd_criterion,
    ce_criterion,
    optimizer,
    num_epochs,
    model_name,
    class_names=None,
    save_dir="checkpoints",
    device=None,
):
    ensure_dir(save_dir)

    device = _resolve_device(student_model, device)
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            if isinstance(batch, dict):
                inputs = batch.get("sequence")
                labels = batch.get("label")
            else:
                inputs, labels = batch

            inputs = inputs.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

            student_inputs = inputs
            teacher_inputs = inputs
            if teacher_inputs.dim() == 2:
                teacher_inputs = teacher_inputs.unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                student_outputs = student_model(student_inputs)
                with torch.no_grad():
                    teacher_outputs = teacher_model(teacher_inputs)

                loss = kd_criterion(student_outputs, teacher_outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(student_outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        val_loss, val_acc, _, _, _ = evaluate_model(
            student_model,
            val_loader,
            ce_criterion,
            class_names,
            dataset_type="validation",
            model_name=model_name,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
            f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(save_dir, f"{model_name}_best_student.pth")
            torch.save(student_model.state_dict(), best_path)
            print(f"Saved best student model to {best_path} (val_acc={best_val_acc:.2f}%)")

    return history

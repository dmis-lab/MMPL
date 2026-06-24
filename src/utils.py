from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, directory: str | None = None):
        self.logger = SummaryWriter(log_dir=directory)

    def close(self):
        self.logger.flush()
        self.logger.close()

    def _log_scalars(self, summary: dict[str, Any], step: int):
        for k, v in summary.items():
            self.logger.add_scalar(k, v, step)

    def __call__(self, name: str, *args: Any, **kwargs: Any):
        if name == "log_scalars":
            return self._log_scalars(*args, **kwargs)
        return getattr(self.logger, name)(*args, **kwargs)


class MetricLogger:
    def __init__(self):
        self.y_probs, self.y_pred, self.y_true = [], [], []

    def log(self, probs, Y_hat: int, Y: int):
        self.y_probs.append(float(probs[1]))
        self.y_pred.append(int(Y_hat))
        self.y_true.append(int(Y))

    def summary(self, verbose: bool = False) -> dict[str, float]:
        acc = accuracy_score(self.y_true, self.y_pred)
        auc = roc_auc_score([1 if y == 1 else 0 for y in self.y_true], self.y_probs)
        f1 = f1_score(self.y_true, self.y_pred, average=None)
        wf1 = f1_score(self.y_true, self.y_pred, average="weighted")
        kappa = cohen_kappa_score(self.y_true, self.y_pred, weights="quadratic")

        if verbose:
            print("*** Metrics ***")
            print(f"* Accuracy: {acc}")
            print(f"* ROC AUC score: {auc}")
            for i in range(len(f1)):
                print(f"* Class {i} f1-score: {f1[i]}")
            print(f"* Weighted f1-score: {wf1}")
            print(f"* Kappa score: {kappa}")

        summary = {"accuracy": acc, "auc": auc, "weighted_f1": wf1, "kappa": kappa}
        for i in range(len(f1)):
            summary[f"class_{i}_f1"] = f1[i]
        return summary

    def get_confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(np.array(self.y_true), np.array(self.y_pred))


class ModelSaver:
    def __init__(self, directory: str, metric: str, mode: str = "min"):
        self.metric = metric
        self.mode = mode
        self.filename = os.path.join(directory, "model_best_loss.pt")
        self.best_value = np.inf if mode == "min" else 0.0

    def __call__(self, model: nn.Module, summary: dict[str, float]):
        if self.mode == "min" and summary[self.metric] > self.best_value:
            return
        if self.mode == "max" and summary[self.metric] < self.best_value:
            return

        print(
            f"{self.metric} {'decreased' if self.mode == 'min' else 'increased'}"
            f" ({self.best_value:.6f} --> {summary[self.metric]:.6f})."
            f" Saving model ..."
        )
        self.best_value = summary[self.metric]
        torch.save(model.state_dict(), self.filename)

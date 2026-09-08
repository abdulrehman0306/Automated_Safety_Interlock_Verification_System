"""
Stage 3. A small network that predicts whether an interlock survives a fault.

Written with an explicit training loop because I wanted to write one, and
because when something goes wrong in a training run you want to be able to see
every line that touched the gradients.

Two things in here exist purely to stop me fooling myself.

The split is by fault type, not by row. Two fault classes are held out entirely
and the network never sees a single example of them during training. A random
split would put nearly identical rows on both sides, since the same scenario is
reused across many sampled interlocks, and the test score would be worthless.

Every result is reported next to the majority class baseline. If the dataset is
70 percent unsafe then a model that always answers unsafe scores 70 percent, and
any number near that means the network learned nothing.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fault_sim import FEATURE_NAMES, build_dataset

# Held out at test time. Single phasing is the interesting one: the measured
# phase current only rises to about 1.7 times rated, which does not look
# alarming, while negative sequence current cooks the rotor in under a minute.
# Nothing in the training set behaves like that. The healthy load spike class is
# held out too so there is at least one non damaging class on the test side,
# otherwise the test set has no safe outcomes to get wrong.
HELDOUT_FAULTS = ["single_phasing", "healthy_load_spike"]

DEFAULT_SEED = 7


@dataclass
class TrainingMetrics:
    epochs: int
    final_train_loss: float
    final_val_loss: float
    first_epoch_train_loss: float
    heldout_accuracy: float
    heldout_majority_baseline: float
    heldout_recall_unsafe: float
    heldout_precision_unsafe: float
    heldout_predicted_positive_rate: float
    indist_accuracy: float
    per_fault_accuracy: Dict[str, float]


class SafetyClassifier(nn.Module):
    """Two hidden layers, and that is deliberate.

    I tried a wider network first and it made no measurable difference on the
    held out classes, which is unsurprising given there are sixteen inputs. A
    bigger model here would only make the results look more impressive to
    somebody who does not read the split.
    """

    def __init__(self, n_features: int, hidden: Sequence[int] = (64, 32)):
        super().__init__()
        layers: List[nn.Module] = []
        prev = n_features
        for size in hidden:
            layers += [nn.Linear(prev, size), nn.ReLU(), nn.Dropout(0.10)]
            prev = size
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class Standardiser:
    """Mean and standard deviation, fitted on training rows only.

    Fitting this on the whole dataset is the classic quiet leak. The held out
    fault classes have different current distributions, so letting them into the
    scaler tells the model something about them before it has seen one.
    """

    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(x: np.ndarray) -> "Standardiser":
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0
        return Standardiser(mean=x.mean(axis=0), std=std)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


def split_by_fault(
    data: Dict[str, np.ndarray],
    heldout: Optional[Sequence[str]] = None,
    seed: int = DEFAULT_SEED,
) -> Dict[str, np.ndarray]:
    """Split into train, validation and held out test.

    Test is whole fault classes. Validation is carved out of the remaining
    classes by scenario id, never by row, so the same simulated trace cannot
    appear on both sides of the validation boundary either.

    The default is None rather than HELDOUT_FAULTS because a default argument is
    evaluated once when the function is defined. The first version had
    HELDOUT_FAULTS sitting in the signature, the leave one fault out sweep tried
    to change it by rebinding the module global, and every one of the seven runs
    silently used the same holdout. The sweep printed seven identical rows and I
    nearly believed them.
    """
    heldout = list(heldout) if heldout is not None else list(HELDOUT_FAULTS)
    is_heldout = np.isin(data["fault_type"], heldout)
    train_pool = ~is_heldout

    scenarios = np.unique(data["scenario_id"][train_pool])
    rng = np.random.default_rng(seed)
    rng.shuffle(scenarios)
    n_val = max(1, int(0.20 * scenarios.size))
    val_scenarios = set(scenarios[:n_val].tolist())

    in_val = np.array([sid in val_scenarios for sid in data["scenario_id"]])
    train_mask = train_pool & ~in_val
    val_mask = train_pool & in_val

    return {
        "train": np.flatnonzero(train_mask),
        "val": np.flatnonzero(val_mask),
        "test": np.flatnonzero(is_heldout),
    }


def train_model(
    data: Dict[str, np.ndarray],
    epochs: int = 60,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
    heldout: Optional[Sequence[str]] = None,
) -> Tuple[SafetyClassifier, Standardiser, List[Dict[str, float]], TrainingMetrics]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    idx = split_by_fault(data, heldout=heldout, seed=seed)
    scaler = Standardiser.fit(data["X"][idx["train"]])

    def tensors(name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(scaler.apply(data["X"][idx[name]]))
        y = torch.from_numpy(data["y"][idx[name]].astype(np.float32))
        return x, y

    x_train, y_train = tensors("train")
    x_val, y_val = tensors("val")
    x_test, y_test = tensors("test")

    model = SafetyClassifier(x_train.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # Weighting the positive class by its inverse frequency. Without it the
    # network drifts toward whichever answer is more common and the recall on
    # unsafe cases, which is the number that actually matters for a safety
    # check, quietly collapses.
    n_pos = float(y_train.sum())
    n_neg = float(y_train.numel() - n_pos)
    pos_weight = torch.tensor(max(n_neg / max(n_pos, 1.0), 1e-3))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history: List[Dict[str, float]] = []
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(x_train.shape[0], generator=generator)
        running = 0.0
        n_batches = 0
        for start in range(0, order.numel(), batch_size):
            batch = order[start : start + batch_size]
            optimiser.zero_grad()
            logits = model(x_train[batch])
            loss = criterion(logits, y_train[batch])
            loss.backward()
            optimiser.step()
            running += float(loss.item())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val), y_val).item())
            val_acc = float(((model(x_val) > 0).float() == y_val).float().mean())

        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(n_batches, 1),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(
                f"epoch {epoch:3d}  train {history[-1]['train_loss']:.4f}  "
                f"val {val_loss:.4f}  val acc {val_acc:.3f}"
            )

    metrics = _evaluate(model, scaler, data, idx, history)
    return model, scaler, history, metrics


def _evaluate(
    model: SafetyClassifier,
    scaler: Standardiser,
    data: Dict[str, np.ndarray],
    idx: Dict[str, np.ndarray],
    history: List[Dict[str, float]],
) -> TrainingMetrics:
    model.eval()

    def predict(rows: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.from_numpy(scaler.apply(data["X"][rows])))
        return (logits > 0).numpy().astype(np.int64)

    test_rows = idx["test"]
    y_test = data["y"][test_rows]
    pred_test = predict(test_rows)

    # The number every accuracy figure in this project has to be read against.
    majority = float(max(y_test.mean(), 1.0 - y_test.mean())) if y_test.size else 0.0

    true_pos = int(((pred_test == 1) & (y_test == 1)).sum())
    false_pos = int(((pred_test == 1) & (y_test == 0)).sum())
    false_neg = int(((pred_test == 0) & (y_test == 1)).sum())

    per_fault: Dict[str, float] = {}
    for fault in np.unique(data["fault_type"][test_rows]):
        mask = data["fault_type"][test_rows] == fault
        per_fault[str(fault)] = float((pred_test[mask] == y_test[mask]).mean())

    val_rows = idx["val"]
    pred_val = predict(val_rows)

    return TrainingMetrics(
        epochs=len(history),
        final_train_loss=history[-1]["train_loss"],
        final_val_loss=history[-1]["val_loss"],
        first_epoch_train_loss=history[0]["train_loss"],
        heldout_accuracy=float((pred_test == y_test).mean()),
        heldout_majority_baseline=majority,
        heldout_recall_unsafe=true_pos / max(true_pos + false_neg, 1),
        heldout_precision_unsafe=true_pos / max(true_pos + false_pos, 1),
        # If this sits at 1.0 the model is answering unsafe to everything and the
        # accuracy figure above is meaningless no matter how high it is.
        heldout_predicted_positive_rate=float(pred_test.mean()),
        indist_accuracy=float((pred_val == data["y"][val_rows]).mean()),
        per_fault_accuracy=per_fault,
    )


def predict_unsafe(
    model: SafetyClassifier,
    scaler: Standardiser,
    features: np.ndarray,
) -> np.ndarray:
    """Probability that each row is unsafe."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(scaler.apply(np.atleast_2d(features))))
    return torch.sigmoid(logits).numpy()


def save_model(path: str, model: SafetyClassifier, scaler: Standardiser, n_features: int) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": scaler.mean,
            "std": scaler.std,
            "n_features": n_features,
        },
        path,
    )


def load_model(path: str) -> Tuple[SafetyClassifier, Standardiser]:
    blob = torch.load(path, weights_only=False)
    model = SafetyClassifier(blob["n_features"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, Standardiser(mean=blob["mean"], std=blob["std"])


if __name__ == "__main__":
    print("building dataset")
    data = build_dataset()
    print(f"rows {data['X'].shape[0]}, features {data['X'].shape[1]}")
    print(f"unsafe fraction overall {data['y'].mean():.3f}")
    print("verdict breakdown:")
    verdicts, counts = np.unique(data["verdict"], return_counts=True)
    for v, c in zip(verdicts, counts):
        print(f"  {v:20s} {c:7d}")
    print()
    model, scaler, history, metrics = train_model(data)
    print()
    for key, value in asdict(metrics).items():
        print(f"{key:34s} {value}")

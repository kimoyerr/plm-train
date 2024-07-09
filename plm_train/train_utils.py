# From torchtitan train.py

from dataclasses import dataclass, field
from typing import Any, Dict, List
from io import BytesIO

import torch
import torch.nn.functional as F
from torch.distributed.checkpoint.stateful import Stateful

from torchtitan.config_manager import JobConfig


@dataclass
class TrainState(Stateful):
    step: int = 0
    global_avg_losses: List[float] = field(default_factory=list)
    global_max_losses: List[float] = field(default_factory=list)
    log_steps: List[int] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        # Only checkpoint global_avg_losses and global_max_losses per log frequency
        # to avoid sync overhead in every iteration.
        global_avg_losses_bytes = BytesIO()
        torch.save(self.global_avg_losses, global_avg_losses_bytes)
        global_max_losses_bytes = BytesIO()
        torch.save(self.global_max_losses, global_max_losses_bytes)
        log_steps_bytes = BytesIO()
        torch.save(self.log_steps, log_steps_bytes)
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "global_avg_losses": global_avg_losses_bytes,
            "global_max_losses": global_max_losses_bytes,
            "log_steps": log_steps_bytes,
        }

    def load_state_dict(self, state_dict) -> None:
        self.step = state_dict["step"].item()
        state_dict["global_avg_losses"].seek(0)
        self.global_avg_losses = torch.load(
            state_dict["global_avg_losses"], weights_only=False
        )
        state_dict["global_max_losses"].seek(0)
        self.global_max_losses = torch.load(
            state_dict["global_max_losses"], weights_only=False
        )
        state_dict["log_steps"].seek(0)
        self.log_steps = torch.load(state_dict["log_steps"], weights_only=False)


def build_optimizer(model, job_config: JobConfig):
    # build optimizer
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    if name == "Adam":
        # TODO: make the optimizer options configurable by toml/cmd args
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1, foreach=True
        )
    elif name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1, foreach=True
        )
    else:
        raise NotImplementedError(f"Optimizer {name} not added.")

    return optimizer


# loss fn can be shared by pipeline-parallel or non-pp execution
def loss_fn(pred, labels, reduction="mean"):
        return F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1), reduction=reduction)

# loss fn can be shared by pipeline-parallel or non-pp execution
# Weighted to handle class imbalance
def loss_fn(pred, labels, class_weights, reduction="mean"):
        preds = pred.flatten(0, 1)
        labels = labels.flatten(0, 1)
        class_weights = class_weights.to(preds.device)

        return F.cross_entropy(preds, labels, weight=class_weights, reduction=reduction)

# F1-score        
def calc_f1_score(pred, labels, pos_label, neg_label):
    # Get probabilities
    log_probs = F.softmax(pred, dim=-1)
    # Assign multi-class labels
    pred_labels = log_probs.argmax(dim=-1)
    # Filter only to labels that are either 4 or 5
    sel_label_mask = (labels==pos_label) | (labels==neg_label)

    # True positive if labels==pos_labels
    true_positives = torch.sum((pred_labels[sel_label_mask] == pos_label).float() * (labels[sel_label_mask] == pos_label).float())
    false_positives = torch.sum((pred_labels[sel_label_mask] == pos_label).float() * (labels[sel_label_mask] == neg_label).float())
    true_negatives = torch.sum((pred_labels[sel_label_mask] == neg_label).float() * (labels[sel_label_mask] == neg_label).float())
    false_negatives = torch.sum((pred_labels[sel_label_mask] == neg_label).float() * (labels[sel_label_mask] == pos_label).float())
    precision = true_positives/(true_positives + false_positives + 1e-6)
    sensitivity = true_positives/(true_positives + false_negatives + 1e-6)
    
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-6)
    return f1_score, precision, sensitivity, true_positives, false_positives, true_negatives, false_negatives

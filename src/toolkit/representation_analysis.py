#!/usr/bin/env python3
"""Shared helpers for offline representation analysis (LP and SNR).

Loads the periodic checkpoints saved by ``CheckpointEveryNStepsPlugin`` and
extracts frozen backbone features (the 160-d vector fed to the multi-head
classifier of ``MTSlimResNet18``), reconstructing the multi-head structure from
the checkpoint's own keys so any intermediate checkpoint can be loaded.
"""
import glob
import os
import re

import torch
import avalanche.models as models
from avalanche.models.dynamic_modules import avalanche_model_adaptation
from torch.nn.functional import avg_pool2d, relu
from torch.utils.data import DataLoader

_STEP_RE = re.compile(r"model_step(\d+)\.ckpt$")
_HEAD_RE = re.compile(r"linear\.classifiers\.(\d+)\.")


def list_checkpoints(logdir):
    """Return [(step, path), ...] sorted by step for a run's checkpoints dir."""
    ckpt_dir = os.path.join(logdir, "checkpoints")
    out = []
    for p in glob.glob(os.path.join(ckpt_dir, "model_step*.ckpt")):
        m = _STEP_RE.search(os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def load_mt_checkpoint(ckpt_path, n_classes_per_exp, task_experiences, device="cpu"):
    """Rebuild MTSlimResNet18 with the heads present in the checkpoint and load it.

    ``task_experiences`` maps task_label -> a benchmark experience (e.g. built from
    ``benchmark.train_stream``); adapting with the *real* experiences reproduces the
    exact per-head sizes from training (head size follows the task's class labels,
    which depend on the benchmark's class-id scheme). Returns (model, task_ids).
    """
    sd = torch.load(ckpt_path, map_location=device)
    task_ids = sorted({int(m.group(1)) for k in sd
                       for m in [_HEAD_RE.match(k)] if m})
    model = models.MTSlimResNet18(n_classes_per_exp)
    for t in task_ids:
        avalanche_model_adaptation(model, task_experiences[t])
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, task_ids


def task_experience_map(stream):
    """{task_label: experience} from a benchmark stream (train or test)."""
    return {exp.task_labels[0]: exp for exp in stream}


def backbone_features(model, x):
    """ϕ(x): the 160-d representation fed to the multi-head classifier."""
    bsz = x.size(0)
    out = relu(model.bn1(model.conv1(x.view(bsz, 3, 32, 32))))
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.layer4(out)
    out = avg_pool2d(out, 4)
    return out.view(out.size(0), -1)


@torch.no_grad()
def extract_features(model, dataset, device="cpu", batch_size=256, max_samples=None):
    """Return (features [N,160], labels [N]) for a task's dataset.

    Datasets from avalanche yield (x, y, task_id); we keep x and y.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=max_samples is not None)
    feats, labels, seen = [], [], 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        feats.append(backbone_features(model, x).cpu())
        labels.append(y)
        seen += len(y)
        if max_samples is not None and seen >= max_samples:
            break
    return torch.cat(feats), torch.cat(labels)

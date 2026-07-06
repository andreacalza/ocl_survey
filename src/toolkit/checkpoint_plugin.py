#!/usr/bin/env python3
"""Save a model checkpoint every N *global* training steps.

Used to enable post-hoc (offline) representation analysis — linear probing (LP,
deep forgetting) and SNR curves — which need the frozen backbone features at many
points along training. Keyed off the global ``clock.train_iterations`` counter, so
the checkpoint steps line up exactly with the anytime-eval checkpoints (same N).

Only the ``state_dict`` is saved (small; the offline loader reconstructs the
multi-head structure from the checkpoint keys). Files are written to
``<logdir>/checkpoints/model_step<K>.ckpt``.
"""
import os

import torch
from avalanche.core import SupervisedPlugin


class CheckpointEveryNStepsPlugin(SupervisedPlugin):
    def __init__(self, logdir: str, every: int = 50):
        super().__init__()
        assert every > 0
        self.every = every
        self.ckpt_dir = os.path.join(logdir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def _save(self, strategy):
        step = strategy.clock.train_iterations
        path = os.path.join(self.ckpt_dir, f"model_step{step}.ckpt")
        torch.save(strategy.model.state_dict(), path)

    def after_training_iteration(self, strategy, **kwargs):
        # global step counter -> one checkpoint every `every` steps. (No
        # after_training_exp hook: in the online setting each micro-experience
        # ends every few steps, which would checkpoint far too often.)
        if strategy.clock.train_iterations % self.every == 0:
            self._save(strategy)

#!/usr/bin/env python3
"""
Smoke test for the task-incremental MTSlimResNet18 setup with anytime
(periodic, every-N-steps) inference. Trains on a single CIFAR100 experience
for a small number of iterations on CPU, just to confirm the multi-head
model, task labels and periodic evaluation plumbing work end to end.

Builds the Avalanche strategy directly (Naive + ReplayPlugin), the same way
src.factories.method_factory does for the "er" strategy, instead of going
through the factory itself, since some unrelated strategies in
src/strategies (e.g. icarl.py) are not compatible with newer avalanche-lib
releases and would otherwise block the import.

NOTE: uses val_size=0 (no validation split) and evaluates on the test stream
instead. avalanche-lib's `benchmark_with_validation_stream` (used by
benchmark_factory.create_benchmark when val_size > 0) drops task labels from
experiences on avalanche-lib>=0.5 (returns plain `DatasetExperience` objects
without `task_labels`), which breaks multi-head/task-aware models. This is a
pre-existing avalanche-lib version incompatibility in benchmark_factory.py,
unrelated to the mt_slim_resnet18 changes here - see the conversation notes.
"""
import os

from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import TextLogger
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer

import src.factories.benchmark_factory as benchmark_factory
import src.factories.model_factory as model_factory

DATASET_ROOT = os.path.expanduser("~/.avalanche/data/cifar100")
N_EXPERIENCES = 10
N_CLASSES_PER_EXP = 100 // N_EXPERIENCES


def test_til_mtslimresnet_anytime_inference(tmp_logdir="/tmp/ocl_survey_til_smoke"):
    os.makedirs(tmp_logdir, exist_ok=True)

    scenario = benchmark_factory.create_benchmark(
        benchmark_name="split_cifar100",
        n_experiences=N_EXPERIENCES,
        val_size=0,
        return_task_id=True,
        dataset_root=DATASET_ROOT,
    )

    model = model_factory.create_model(
        model_type="mt_slim_resnet18",
        input_size=(3, 32, 32),
        n_classes_per_exp=N_CLASSES_PER_EXP,
    )

    optimizer, _ = model_factory.get_optimizer(
        model,
        optimizer_type="SGD",
        kwargs_optimizer={"lr": 0.1, "momentum": 0.0, "weight_decay": 0.0},
    )

    storage_policy = ClassBalancedBuffer(max_size=200, adaptive_size=True)
    replay_plugin = ReplayPlugin(mem_size=200, storage_policy=storage_policy)

    log_file = open(os.path.join(tmp_logdir, "logs.txt"), "w")
    evaluator = EvaluationPlugin(
        accuracy_metrics(stream=True, experience=True),
        loss_metrics(stream=True),
        loggers=[TextLogger(log_file)],
    )

    strategy = Naive(
        model=model,
        optimizer=optimizer,
        train_mb_size=32,
        train_epochs=1,
        eval_mb_size=32,
        device="cpu",
        eval_every=50,
        peval_mode="iteration",
        plugins=[replay_plugin],
        evaluator=evaluator,
    )

    first_experience = scenario.train_stream[0]
    strategy.train(
        first_experience,
        eval_streams=[scenario.test_stream[:1]],
        num_workers=0,
    )

    results = strategy.eval(scenario.test_stream[:1])
    assert results is not None

    print("Smoke test passed: MTSlimResNet18 + TIL + anytime inference works.")


if __name__ == "__main__":
    test_til_mtslimresnet_anytime_inference()

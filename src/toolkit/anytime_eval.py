#!/usr/bin/env python3
"""Anytime (every-N-steps) evaluation for the online setting."""
from avalanche.core import SupervisedPlugin


class AnytimeEvalPlugin(SupervisedPlugin):
    """Evaluate every ``eval_every`` *global* training iterations.

    The built-in avalanche ``PeriodicEval`` keys its periodic evaluation off
    ``clock.train_exp_iterations``, which is reset at the start of every
    experience. In the online setting each ``OnlineCLScenario`` sub-experience
    is a single mini-batch, so that per-experience counter never grows and the
    built-in periodic eval instead fires once per sub-experience (i.e. at almost
    every step). This plugin keys off the *global* ``clock.train_iterations``
    counter, which is never reset, giving a true "anytime inference every N
    steps" schedule regardless of how the online stream is chunked.

    Intended usage: disable the built-in periodic eval (``eval_every=-1`` on the
    strategy) and add this plugin instead. It evaluates on the same streams that
    were passed to ``strategy.train(..., eval_streams=...)``.
    """

    def __init__(self, eval_every: int = 50):
        super().__init__()
        assert eval_every > 0, "eval_every must be a positive number of steps"
        self.eval_every = eval_every

    def after_training_iteration(self, strategy, **kwargs):
        if strategy.clock.train_iterations % self.eval_every == 0:
            for stream in strategy._eval_streams:
                strategy.eval(stream)

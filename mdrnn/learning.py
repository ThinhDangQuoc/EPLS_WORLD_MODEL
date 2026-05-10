from functools import partial
from typing import Callable


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        mode: 'min' stops when metric stops decreasing; 'max' when it stops increasing.
        patience: epochs to wait before stopping after the last improvement.
        threshold: minimum change to count as an improvement.
        threshold_mode: 'rel' or 'abs' threshold interpretation.
    """

    def __init__(self, mode='min', patience=10, threshold=1e-4, threshold_mode='rel'):
        self.patience = patience
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best: float = 0.0
        self.num_bad_epochs = 0
        self.mode_worse: float = 0.0
        self.is_better: Callable[[float, float], bool] = lambda a, b: False
        self.last_epoch = -1
        self._init_is_better(mode, threshold, threshold_mode)
        self._reset()

    def _reset(self):
        self.best = self.mode_worse
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

    @property
    def stop(self):
        return self.num_bad_epochs > self.patience

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            return a < best * (1.0 - threshold)
        if mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold
        if mode == 'max' and threshold_mode == 'rel':
            return a > best * (1.0 + threshold)
        return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError(f'mode {mode!r} is unknown')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError(f'threshold_mode {threshold_mode!r} is unknown')
        self.mode_worse = float('inf') if mode == 'min' else -float('inf')
        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'is_better'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(self.mode, self.threshold, self.threshold_mode)

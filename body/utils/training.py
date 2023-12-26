import functools
import logging
from contextlib import contextmanager
import inspect
import time
import sys

import torch

logger = logging.getLogger(__name__)

def capture_init(init):
    """capture_init.

    Decorate `__init__` with this, and you can then
    recover the *args and **kwargs passed to it in `self._init_args_kwargs`
    """
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__


def deserialize_model(package, strict=False):
    """deserialize_model.

    """
    klass = package['class']
    kwargs = package['kwargs']
    if 'sample_rate' not in kwargs:
        logger.warning(
            "Training sample rate not available!, 16kHz will be assumed. "
            "If you used a different sample rate at train time, please fix your checkpoint "
            "with the command `./train.py [TRAINING_ARGS] save_again=true.")
    if strict:
        model = klass(*package['args'], **kwargs)
    else:
        sig = inspect.signature(klass)
        kw = package['kwargs']
        for key in list(kw):
            if key not in sig.parameters:
                logger.warning("Dropping inexistant parameter %s", key)
                del kw[key]
        model = klass(*package['args'], **kw)
    model.load_state_dict(package['state'])
    return model


def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


def serialize_model(model):
    state = copy_state(model.state_dict())
    return {"class": model.__class__, "state": state}


@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


def pull_metric(history, name):
    out = []
    for metrics in history:
        if name in metrics:
            out.append(metrics[name])
    return out
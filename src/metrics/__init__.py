import importlib


def get_metric(name, **kwargs):
    return importlib.import_module('src.metrics.{}.evaluator'.format(name)).EvaluateTool(**kwargs)

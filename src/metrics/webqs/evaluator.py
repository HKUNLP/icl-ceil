# encoding=utf8
from src.utils.misc import parallel_run
from .answer_util import max_em


class EvaluateTool:
    def __init__(self):
        pass

    def evaluate(self, preds, golds):
        golds = [gold["answers"] for gold in golds]
        eval_results = parallel_run(eval_single, list(zip(preds, golds)))
        metrics = {"em": sum(eval_results) / len(golds)}
        return metrics


def eval_single(args):
    pred, gold_answers = args
    return max_em(pred, gold_answers)

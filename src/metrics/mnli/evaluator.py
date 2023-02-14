import evaluate
import logging
logger = logging.getLogger(__name__)


class EvaluateTool(object):
    def __init__(self):
        pass

    def evaluate(self, preds, golds):
        golds = [gold["label"] for gold in golds]
        metric = evaluate.load("accuracy")
        return metric.compute(references=golds, predictions=preds)

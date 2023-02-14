# encoding=utf8
from datasets import load_metric


class EvaluateTool:
    def __init__(self):
        pass

    def evaluate(self, preds, golds):
        # character-level 4-bleu
        bleu = load_metric('bleu')
        predictions = [[ch for ch in text] for text in preds]
        references = [[[ch for ch in entry['bash']]] for entry in golds]
        return bleu.compute(predictions=predictions, references=references)
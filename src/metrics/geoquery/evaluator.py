from src.utils.misc import parallel_run


class EvaluateTool(object):
    def __init__(self):
        pass

    def evaluate(self, preds, golds):
        golds = [gold["funql"] for gold in golds]
        em_results = parallel_run(eval_em_single, list(zip(preds, golds)))
        return {"em": sum(em_results) / len(golds)}


def eval_em_single(args):
    pred, gold = args
    return pred.strip() == gold



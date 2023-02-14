from src.utils.misc import parallel_run
import regex as re
import sys
sys.path.append("third_party")

from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.logical_form_matcher import \
    LogicalFromStructuralMatcher
from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.qdmr_to_logical_form_tokens import \
    QDMRToQDMRStepTokensConverter
from qdecomp_with_dependency_graphs.evaluation.normal_form.normalized_graph_matcher import \
    NormalizedGraphMatchScorer
from qdecomp_with_dependency_graphs.scripts.eval.evaluate_predictions import format_qdmr
from dataclasses import dataclass


@dataclass
class GlobalState:
    converter = None
    matcher = None
    scorer = None

    def __post_init__(self):
        self.converter = QDMRToQDMRStepTokensConverter()
        self.matcher = LogicalFromStructuralMatcher()
        self.scorer = NormalizedGraphMatchScorer()


global_state = {}


def set_global_object():
    global global_state
    global_state = GlobalState()


def renorm(text):
    text = text.split("\n")[0]
    text = re.sub("[\d]+\#\) ", ";", text)
    return text.strip().replace("  ", " ").lower()


class EvaluateTool(object):
    def __init__(self):
        pass

    def evaluate(self, preds, golds):
        questions = [gold['question_text'] for gold in golds]
        golds = [gold["decomposition"] for gold in golds]
        mrange = list(range(len(preds)))
        eval_results = parallel_run(eval_single, initializer=set_global_object, initargs={},
                                    args_list=list(zip(questions, preds, golds, mrange)))
        metrics = {"lf-em": sum(eval_results) / len(golds)}
        return metrics


def eval_single(args):
    question, generated, decomposition, index = args
    try:
        # print(f"Starting: {index}")
        if "#13" in generated:
            return False

        gold = format_qdmr(decomposition)
        pred = format_qdmr(renorm(generated))

        decomp_lf = global_state.converter.convert(question_id=str(index), question_text=question,
                                                   decomposition=pred.to_break_standard_string())
        gold_lf = global_state.converter.convert(question_id=str(index), question_text=question,
                                                 decomposition=gold.to_break_standard_string())
        s = global_state.matcher.is_match(question_id=str(index), question_text=question, graph1=decomp_lf,
                                          graph2=gold_lf)
        return s
    except Exception as ex:
        return False

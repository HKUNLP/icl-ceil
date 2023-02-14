from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *
import logging

logger = logging.getLogger(__name__)
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    return entry["question"]


@field_getter.add("qa")
def get_qa(entry):
    return "Q: {text}\tA: {label}".format(
        text=get_q(entry),
        label=get_a(entry)
    )


@field_getter.add("a")
def get_a(entry):
    return get_choices(entry)[ord(entry['answerKey'])-ord('A')]


@field_getter.add("gen_a")
def get_gen_a(entry):
    # hypothesis, premise = get_q(entry)
    return "{ice_prompt}Q: {text}\tA: ".format(
        ice_prompt="{ice_prompt}",
        text=get_q(entry))


@field_getter.add("choices")
def get_choices(entry):
    return entry['choices']['text']


class DatasetWrapper(ABC):
    name = "cmsqa"
    ice_separator = "\n"
    question_field = "question"
    answer_field = "answerKey"
    hf_dataset = "commonsense_qa"
    field_getter = field_getter

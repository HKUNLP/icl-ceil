from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *
import logging

logger = logging.getLogger(__name__)
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    return entry["sentence1"]+" "+entry["sentence2"]


@field_getter.add("qa")
def get_qa(entry):
    return "{sentence1} Can we say \"{sentence2}\"? {label}".format(
            sentence1=entry["sentence1"],
            label=get_a(entry),
            sentence2=entry["sentence2"]
            )


@field_getter.add("a")
def get_a(entry):
    return get_choices(entry)[entry['label']]


@field_getter.add("gen_a")
def get_gen_a(entry):
    # hypothesis, premise = get_q(entry)
    return "{ice_prompt}{sentence1} Can we say \"{sentence2}\"? ".format(
            ice_prompt="{ice_prompt}",
            sentence1=entry["sentence1"],
            sentence2=entry["sentence2"])


@field_getter.add("choices")
def get_choices(entry):
    return ["No", "Yes"]


class DatasetWrapper(ABC):
    name = "mrpc"
    ice_separator = "\n"
    question_field = ["sentence1", "sentence2"]
    answer_field = "label"
    hf_dataset = "glue"
    hf_dataset_name = "mrpc"
    field_getter = field_getter

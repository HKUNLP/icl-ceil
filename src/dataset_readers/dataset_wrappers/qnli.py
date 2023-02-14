from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *
import logging

logger = logging.getLogger(__name__)
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    return entry["sentence"] + " " + entry["question"]


@field_getter.add("qa")
def get_qa(entry):
    return "{sentence} Can we know \"{question}\"? {label}".format(
            sentence=entry["sentence"],
            label=get_a(entry),
            question=entry["question"]
            )


@field_getter.add("a")
def get_a(entry):
    return get_choices(entry)[entry['label']]


@field_getter.add("gen_a")
def get_gen_a(entry):
    # hypothesis, premise = get_q(entry)
    return "{ice_prompt}{sentence} Can we know \"{question}\"? ".format(
            ice_prompt="{ice_prompt}",
            sentence=entry["sentence"],
            question=entry["question"])


@field_getter.add("choices")
def get_choices(entry):
    return ["Yes", "No"]


class DatasetWrapper(ABC):
    name = "qnli"
    ice_separator = "\n"
    question_field = ["question", "sentence"]
    answer_field = "label"
    hf_dataset = "glue"
    hf_dataset_name = "qnli"
    field_getter = field_getter

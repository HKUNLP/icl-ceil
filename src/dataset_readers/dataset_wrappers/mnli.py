from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *
import logging

logger = logging.getLogger(__name__)
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    return entry["premise"] + " " + entry["hypothesis"]


@field_getter.add("qa")
def get_qa(entry):
    return "{premise} Can we say \"{hypothesis}\"? {label}".format(
            hypothesis=entry["hypothesis"],
            label=get_a(entry),
            premise=entry["premise"]
            )


@field_getter.add("a")
def get_a(entry):
    return get_choices(entry)[entry['label']]


@field_getter.add("gen_a")
def get_gen_a(entry):
    # hypothesis, premise = get_q(entry)
    return "{ice_prompt}{premise} Can we say \"{hypothesis}\"? ".format(
            ice_prompt="{ice_prompt}",
            hypothesis=entry["hypothesis"],
            premise=entry["premise"])


@field_getter.add("choices")
def get_choices(entry):
    return ["Yes", "Maybe", "No"]


class DatasetWrapper(ABC):
    name = "mnli"
    ice_separator = "\n"
    question_field = ["hypothesis", "premise"]
    answer_field = "label"
    hf_dataset = "LysandreJik/glue-mnli-train"
    hf_dataset_name = "glue-mnli-train"
    field_getter = field_getter

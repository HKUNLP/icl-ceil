from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *
import logging
logger = logging.getLogger(__name__)
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    return entry['startphrase']


@field_getter.add("qa")
def get_qa(entry):
    return f"{get_q(entry)} {get_a(entry)}"


@field_getter.add("a")
def get_a(entry):
    return entry[f"ending{entry['label']}"]


@field_getter.add("gen_a")
def get_gen_a(entry):
    return "{ice_prompt}{question} ".format(
        question=get_q(entry),
        ice_prompt="{ice_prompt}")


@field_getter.add("choices")
def get_choices(entry):
    return [entry[f"ending{i}"] for i in range(4)]


class DatasetWrapper(ABC):
    name = "swag"
    ice_separator = "\n"
    question_field = "startphrase"
    answer_field = "label"
    hf_dataset = "swag"
    hf_dataset_name = "regular"
    field_getter = field_getter
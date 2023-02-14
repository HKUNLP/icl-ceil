from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *
import logging
logger = logging.getLogger(__name__)
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    return entry['text']


@field_getter.add("qa")
def get_qa(entry):
    return f"{get_q(entry)} It is {get_a(entry)}"


@field_getter.add("a")
def get_a(entry):
    return get_choices(entry)[entry['label']]


@field_getter.add("gen_a")
def get_gen_a(entry):
    return "{ice_prompt}{question} It is ".format(
        question=get_q(entry),
        ice_prompt="{ice_prompt}")


@field_getter.add("choices")
def get_choices(entry):
    return ["terrible", "bad", "OK", "good", "great"]


class DatasetWrapper(ABC):
    name = "sst5"
    ice_separator = "\n"
    question_field = "text"
    answer_field = "label"
    hf_dataset = "SetFit/sst5"
    hf_dataset_name = None
    field_getter = field_getter

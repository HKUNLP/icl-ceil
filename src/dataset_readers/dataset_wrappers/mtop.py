from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *

field_getter = App()


@field_getter.add("q")
def get_q(entry):
    return entry['question']


@field_getter.add("a")
def get_a(entry):
    return entry['logical_form']


@field_getter.add("qa")
def get_qa(entry):
    return f"{get_q(entry)}\t{get_a(entry)}"


@field_getter.add("gen_a")
def get_gen_a(entry):
    return "{ice_prompt}{question}\t".format(question=get_q(entry), ice_prompt="{ice_prompt}")


class DatasetWrapper(ABC):
    name = "mtop"
    ice_separator = "\n"
    question_field = "question"
    answer_field = "logical_form"
    hf_dataset = "iohadrubin/mtop"
    hf_dataset_name = "mtop"
    field_getter = field_getter
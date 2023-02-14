from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *

field_getter = App()


@field_getter.add("q")
def get_q(entry):
    return entry['user_utterance']


@field_getter.add("a")
def get_a(entry):
    return entry['lispress']


@field_getter.add("qa")
def get_qa(entry):
    return f"{get_q(entry)}\t{get_a(entry)}"


@field_getter.add("gen_a")
def get_gen_a(entry):
    return "{ice_prompt}{question}\t".format(question=get_q(entry), ice_prompt="{ice_prompt}")


class DatasetWrapper(ABC):
    name = "smcalflow"
    ice_separator = "\n"
    question_field = "user_utterance"
    answer_field = "lispress"
    hf_dataset = "iohadrubin/smcalflow"
    hf_dataset_name = "smcalflow"
    field_getter = field_getter
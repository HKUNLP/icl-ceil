from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import *
import regex as re


field_getter = App()


def remove_double_space(string):
    return re.sub("[ ]{2,}", " ", string)


def reformat(text):
    return " ".join([f"{i + 1}#) {x.strip()}" for i, x in enumerate(text.split(";"))])


@field_getter.add("q")
def get_q(entry):
    return remove_double_space(entry['question_text'])


@field_getter.add("qa")
def get_qa(entry):
    return f"{get_q(entry)}\t{get_a(entry)}"


@field_getter.add("a")
def get_a(entry):
    return remove_double_space(reformat(entry['decomposition']))


@field_getter.add("gen_a")
def get_gen_a(entry):
    return "{ice_prompt}{question}\t".format(question=get_q(entry), ice_prompt="{ice_prompt}")


class DatasetWrapper(ABC):
    name = "break"
    ice_separator = "\n"
    question_field = "question_text"
    answer_field = "decomposition"
    hf_dataset = "break_data"
    hf_dataset_name = "QDMR"
    field_getter = field_getter
    a_prefix = "1#)"

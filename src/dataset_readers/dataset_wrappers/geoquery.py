from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import ABC

field_getter = App()


@field_getter.add("q")
def get_q(entry):
    return entry['question']


@field_getter.add("a")
def get_a(entry):
    return entry['funql']


@field_getter.add("qa")
def get_qa(entry):
    return f"{get_q(entry)}\t{get_a(entry)}"


@field_getter.add("gen_a")
def get_gen_a_instruction(entry):
    return "{ice_prompt}{question}\t".format(question=entry['question'], ice_prompt='{ice_prompt}')


class DatasetWrapper(ABC):
    name = "geoquery"
    ice_separator = "\n"
    question_field = "question"
    answer_field = "funql"
    hf_dataset = 'src/hf_datasets/geoquery.py'
    hf_dataset_name = 'standard'  # standard, tmcd, template, length; currently hard coded
    field_getter = field_getter

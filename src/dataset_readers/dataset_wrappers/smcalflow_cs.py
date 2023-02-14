import datasets
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


# modify n_shot to use n compositional demonstrations, n in [8, 16, 32], currently hard coded
few_shots = datasets.load_dataset("src/hf_datasets/smcalflow_cs.py")['8_shot']
few_shots = "\n".join([get_qa(entry) for entry in few_shots][::-1]) + '\n'


@field_getter.add("gen_a")
def get_gen_a(entry):
    return "{extra}{ice_prompt}{question}\t".format(extra=few_shots, question=get_q(entry), ice_prompt="{ice_prompt}")
    # return "{ice_prompt}{question}\t".format(question=get_q(entry), ice_prompt="{ice_prompt}")


class DatasetWrapper(ABC):
    name = "smcalflow_cs"
    ice_separator = "\n"
    question_field = "user_utterance"
    answer_field = "lispress"
    hf_dataset = "src/hf_datasets/smcalflow_cs.py"
    hf_dataset_name = None
    field_getter = field_getter
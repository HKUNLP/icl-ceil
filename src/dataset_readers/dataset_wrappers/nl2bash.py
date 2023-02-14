from src.utils.misc import App
from src.dataset_readers.dataset_wrappers.base_dsw import ABC

field_getter = App()


@field_getter.add("q")
def get_q(entry):
    # in-context example for few-shot generating question
    return f"{entry['nl']}"


@field_getter.add("a")
def get_a(entry):
    return entry['bash']


@field_getter.add("qa")
def get_qa(entry):
    return f"{get_q(entry)}\t{get_a(entry)}"


@field_getter.add("gen_a")
def get_gen_a_instruction(entry):
    prompt = "{ice_prompt}{question}\t"
    prompt = prompt.format(
        question=get_q(entry),
        ice_prompt="{ice_prompt}")
    return prompt


class DatasetWrapper(ABC):
    name = "nl2bash"
    ice_separator = "\n"
    question_field = "nl"
    answer_field = "bash"
    hf_dataset = "src/hf_datasets/nl2bash.py"
    hf_dataset_name = 'nl2bash'
    field_getter = field_getter

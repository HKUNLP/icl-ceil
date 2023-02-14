import pandas as pd
from datasets import Dataset
from copy import deepcopy

from src.dataset_readers.base_dsr import encode_field
from src.dataset_readers.inference_dsr import InferenceDatasetReader
from src.dataset_readers.dataset_wrappers import get_dataset_wrapper


class ScorerDatasetReader(InferenceDatasetReader):

    def init_dataset(self, task_name, field, dataset_path, dataset_split, ds_size, truncation=False):
        def get_instance(idx, entry):
            # todo, note here we may overwrite original idx field (if exists)
            entry['idx'] = idx  # unique id of original instances, used for grouping instances
            ctxs_candidates = entry.pop("ctxs_candidates")
            for exp in ctxs_candidates:
                example = deepcopy(entry)
                example['ctxs'] = exp
                yield example

        def get_dataset(data):
            for idx, entry in enumerate(data):
                yield from get_instance(idx, entry)

        self.dataset_wrapper = get_dataset_wrapper(task_name, dataset_path=dataset_path,
                                                   dataset_split=dataset_split,
                                                   ds_size=ds_size)
        df = pd.DataFrame(list(get_dataset(self.dataset_wrapper.dataset)))
        self.dataset_wrapper.dataset = Dataset.from_pandas(df)
        self.encoded_dataset = encode_field(self.tokenizer, self.dataset_wrapper, field, truncation)

    def __getitem__(self, index):
        entry = self.dataset_wrapper[index]
        prompt_len = self.encoded_dataset[index]['metadata']['len']
        prompt = self.encoded_dataset[index]['metadata']['text']

        answer = self.dataset_wrapper.get_field(entry=entry, field="a")
        tokenized_labels = self.tokenizer.encode_plus(answer, truncation=False, add_special_tokens=False,
                                                      return_tensors='pt')
        answer_len = tokenized_labels.attention_mask.shape[1]

        ice_prompt, trunc_ice_prompts_list = self.get_ice_prompt(entry, prompt_len+answer_len)
        # do not use format, as some prompts also contains {xxx} :(
        prompt = prompt.replace("{ice_prompt}", ice_prompt)

        entry['prompt'] = prompt + answer

        entry['ice_prompts_list'] = trunc_ice_prompts_list

        tokenized_example = self.tokenizer.encode_plus(entry['prompt'], truncation=False, return_tensors='pt',
                                                       add_special_tokens=False)

        return {
            'input_ids': tokenized_example.input_ids[0],
            'labels': tokenized_labels.attention_mask[0],
            "metadata": entry
        }
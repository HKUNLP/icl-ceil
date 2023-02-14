import numpy as np
from src.dataset_readers.base_dsr import BaseDatasetReader
from src.utils.tokenizer_util import get_tokenizer
import logging
logger = logging.getLogger(__name__)


class InferenceDatasetReader(BaseDatasetReader):

    def __init__(self, index_reader, n_tokens, task_name, model_name, field, dataset_path=None,
                 dataset_split=None, ds_size=None):
        self.tokenizer = get_tokenizer(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.index_reader = index_reader
        self.n_tokens_in_prompt = n_tokens
        # set truncation to false so that metadata['len'] corresponds to metadata['text']
        self.init_dataset(task_name, field, dataset_path, dataset_split, ds_size, truncation=False)

    def get_ice_prompt(self, entry, prompt_len):
        if 'ctxs' in entry:
            ctx = [self.index_reader[i] for i in entry['ctxs']]

            ice_prompts_list = [i['metadata']['text'] for i in ctx]
            ice_lengths_list = [i['metadata']['len'] for i in ctx]

            trunc_ice_prompts_list = self.truncate(prompt_len, ice_lengths_list, ice_prompts_list)
            ice_separator = self.dataset_wrapper.ice_separator
            ice_prompt = ice_separator.join(trunc_ice_prompts_list) + ice_separator
        else:
            trunc_ice_prompts_list = []
            ice_prompt = ""
        return ice_prompt, trunc_ice_prompts_list

    def __getitem__(self, index):
        entry = self.dataset_wrapper[index]
        prompt_len = self.encoded_dataset[index]['metadata']['len']
        prompt = self.encoded_dataset[index]['metadata']['text']

        ice_prompt, trunc_ice_prompts_list = self.get_ice_prompt(entry, prompt_len)
        # do not use format, as some prompts also contains {xxx} :(
        prompt = prompt.replace("{ice_prompt}", ice_prompt)

        entry['prompt'] = prompt + self.dataset_wrapper.a_prefix
        entry['ice_prompts_list'] = trunc_ice_prompts_list

        tokenized_example = self.tokenizer.encode_plus(entry['prompt'], truncation=False, return_tensors='pt',
                                                       add_special_tokens=False)

        return {
            'input_ids': tokenized_example.input_ids[0],
            'attention_mask': tokenized_example.attention_mask[0],
            "metadata": entry
        }

    def __len__(self):
        return len(self.dataset_wrapper)

    def shard(self, accelerator):
        self.dataset_wrapper.dataset = self.dataset_wrapper.dataset.shard(
            num_shards=accelerator.num_processes,
            index=accelerator.process_index
        )
        self.encoded_dataset = self.encoded_dataset.shard(
            num_shards=accelerator.num_processes,
            index=accelerator.process_index
        )
        assert len(self.dataset_wrapper.dataset) == len(self.encoded_dataset)

    def truncate(self, test_input_len, lengths_list, prompts_list):
        max_prompts = np.searchsorted(np.cumsum(lengths_list), self.n_tokens_in_prompt - test_input_len)
        # logger.info(self.n_tokens_in_prompt, max_prompts)
        trunc_prompts_list = prompts_list[:max_prompts][::-1]  # more similar more close
        return trunc_prompts_list

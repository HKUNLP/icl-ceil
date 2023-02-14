import torch
from transformers import AutoTokenizer
from src.dataset_readers.dataset_wrappers import get_dataset_wrapper
import collections
from copy import deepcopy

InputSample = collections.namedtuple(
    "InputSample",
    [
        "question_ids",
        "ctxs_candidates"
    ]
)


def encode_field(example, **kwargs):
    field_getter = kwargs['field_getter']
    tokenizer = kwargs['tokenizer']
    question = field_getter(example)
    question_ids = tokenizer.encode(question, truncation=True)
    return {
        "question_ids": question_ids,
        "ctxs_candidates": example['ctxs_candidates']
    }


class TrainingDatasetReader(torch.utils.data.Dataset):

    def __init__(self, task_name, model_name, field, dataset_path, ds_size=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset_wrapper = get_dataset_wrapper(task_name, dataset_path=dataset_path, ds_size=ds_size)
        self.encoded_dataset = self.encode_field(dataset_wrapper, field)

    def encode_field(self, dataset_wrapper, field):
        remove_columns = [col for col in dataset_wrapper.dataset.column_names]
        encoded_dataset = dataset_wrapper.dataset.map(
            encode_field,
            load_from_cache_file=False,
            remove_columns=remove_columns,
            fn_kwargs={'field_getter': dataset_wrapper.field_getter.functions[field],
                       'tokenizer': self.tokenizer}
        )
        return encoded_dataset

    def __getitem__(self, index) -> InputSample:
        return InputSample(**self.encoded_dataset[index])

    def __len__(self):
        return len(self.encoded_dataset)

    def split_dataset(self, test_size=0.1, seed=42):
        dataset = self.encoded_dataset.train_test_split(test_size=test_size, seed=seed)
        train_dataset, eval_dataset = dataset['train'], dataset['test']

        cache_self = {k: self.__dict__[k] for k in self.__dict__.keys()}
        for k in self.__dict__.keys():
            self.__dict__[k] = None

        trainset_cls = deepcopy(self)
        for k, v in cache_self.items():
            trainset_cls.__dict__[k] = v
        trainset_cls.encoded_dataset = train_dataset

        evalset_cls = deepcopy(self)
        for k, v in cache_self.items():
            evalset_cls.__dict__[k] = v
        evalset_cls.encoded_dataset = eval_dataset

        self.__dict__ = cache_self
        return trainset_cls, evalset_cls
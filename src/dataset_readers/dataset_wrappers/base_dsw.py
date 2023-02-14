#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import os
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import logging
import random

logger = logging.getLogger(__name__)


class ABC:
    name = "base"
    ice_separator = None
    question_field = None
    answer_field = None
    hf_dataset = None
    hf_dataset_name = None
    field_getter = None
    a_prefix = ""  # an answer prefix as beginning word, e.g., SELECT

    def __init__(self, dataset_path=None, dataset_split=None, ds_size=None):
        if dataset_path is None or not os.path.exists(dataset_path):
            self.dataset = load_dataset(self.hf_dataset, self.hf_dataset_name)
        else:
            self.dataset = Dataset.from_pandas(pd.read_json(dataset_path))
            logger.info(f"Loading dataset from {dataset_path}, size {len(self.dataset)}")

        if dataset_split is not None and isinstance(self.dataset, DatasetDict):
            self.dataset = self.dataset[dataset_split]

        if ds_size is not None:
            self.dataset = load_partial_dataset(self.dataset, size=ds_size)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def get_field(self, entry, field):
        return self.field_getter[field](entry)

    def get_corpus(self, field):
        return [self.get_field(entry, field) for entry in self.dataset]


def load_partial_dataset(dataset, size=1):
    if size == 1 or size >= len(dataset):
        return dataset

    total_size = len(dataset)
    size = int(size * total_size) if size < 1 else size

    rand = random.Random(x=size)
    index_list = list(range(total_size))
    rand.shuffle(index_list)
    dataset = dataset.select(index_list[:size])
    return dataset
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def show_statistics(encoded_dataset, dataset_name):
    all_lens = [item['metadata']['len'] for item in encoded_dataset]
    all_lens = pd.Series(all_lens, dtype=int)
    logger.info(f"length of {dataset_name}: {str(all_lens.describe())}")
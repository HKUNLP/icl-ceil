import logging
import pandas as pd
import os
from src.dataset_readers.base_dsr import BaseDatasetReader, encode_field
from src.utils.misc import save_json

logger = logging.getLogger(__name__)


def deduplicate(dataset_wrapper, encoded_dataset):
    """deduplication """
    df = pd.DataFrame(encoded_dataset)
    df['uid'] = df['input_ids'].astype(str)
    is_dup = df.duplicated(subset=['uid'], keep='first')
    keep_idx = is_dup[~is_dup].index.values

    dataset_wrapper.dataset = dataset_wrapper.dataset.select(keep_idx)
    encoded_dataset = encoded_dataset.select(keep_idx)

    encoded_dataset = encoded_dataset.map(reassign_idx, load_from_cache_file=False, with_indices=True)
    logger.info(f"Keeping {len(keep_idx)}/{len(df)} instances after deduplicating")
    return dataset_wrapper, encoded_dataset


def reassign_idx(example, index):
    example['metadata']['id'] = index
    return example


class IndexDatasetReader(BaseDatasetReader):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        dataset_path = kwargs['dataset_path']
        # if not create index file, we create it by deduplication q field
        if dataset_path is None or not os.path.exists(dataset_path):
            if kwargs['field'] == 'q':
                self.dataset_wrapper, self.encoded_dataset = deduplicate(self.dataset_wrapper, self.encoded_dataset)
            else:
                # use field q for deduplication
                encoded_dataset = encode_field(self.tokenizer, self.dataset_wrapper, 'q', truncation=True)
                # make sure all items in index are unique
                self.dataset_wrapper, _ = deduplicate(self.dataset_wrapper, encoded_dataset)
                # re-encode using deduplicated dataset_wrapper
                self.encoded_dataset = encode_field(self.tokenizer, self.dataset_wrapper, kwargs['field'], truncation=True)

            if dataset_path is not None:
                save_json(dataset_path, list(self.dataset_wrapper))
                logger.info(f"index dataset has been saved to {dataset_path}")
import logging
import hydra
import hydra.utils as hu
import numpy as np
from omegaconf import DictConfig
from transformers import set_seed
from tqdm import tqdm
from src.dataset_readers.dataset_wrappers import get_dataset_wrapper
from src.utils.misc import save_json

logger = logging.getLogger(__name__)


class RandomFinder:
    def __init__(self, cfg: DictConfig) -> None:
        self.output_file = cfg.output_file
        self.is_train = cfg.dataset_split == "train"
        self.num_candidates = cfg.num_candidates
        self.num_ice = cfg.num_ice
        self.index_dataset = hu.instantiate(cfg.index_reader).dataset_wrapper
        self.dataset_wrapper = get_dataset_wrapper(cfg.task_name, dataset_split=cfg.dataset_split, ds_size=cfg.ds_size)

    def find(self):
        num_index = len(self.index_dataset)
        ctxs_candidates = []
        while len(ctxs_candidates) < self.num_candidates:
            candidate = np.random.choice(num_index, self.num_ice, replace=False).tolist()
            if candidate not in ctxs_candidates:
                ctxs_candidates.append(candidate)
        return ctxs_candidates[0], ctxs_candidates


def find(cfg):
    finder = RandomFinder(cfg)
    entries = []
    for entry in tqdm(finder.dataset_wrapper):
        ctxs, ctxs_candidates = finder.find()
        entry['ctxs'] = ctxs
        entry['ctxs_candidates'] = ctxs_candidates
        entries.append(entry)

    save_json(finder.output_file, entries)


@hydra.main(config_path="configs", config_name="bm25_retriever")
def main(cfg):
    logger.info(cfg)
    set_seed(43)
    find(cfg)


if __name__ == "__main__":
    main()

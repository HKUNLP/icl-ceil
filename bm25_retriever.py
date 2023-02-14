import logging
import json
import hydra
import hydra.utils as hu
import numpy as np
from tqdm import tqdm
import multiprocessing
from transformers import set_seed
from rank_bm25 import BM25Okapi
from omegaconf import DictConfig
from nltk.tokenize import word_tokenize
from src.dataset_readers.dataset_wrappers import get_dataset_wrapper

logger = logging.getLogger(__name__)


class BM25Finder:
    def __init__(self, cfg: DictConfig) -> None:
        self.output_file = cfg.output_file
        self.is_train = cfg.dataset_split == "train"
        self.num_candidates = cfg.num_candidates
        self.num_ice = cfg.num_ice
        self.index_dataset = hu.instantiate(cfg.index_reader).dataset_wrapper
        self.dataset_wrapper = get_dataset_wrapper(cfg.task_name, dataset_split=cfg.dataset_split, ds_size=cfg.ds_size)

        logger.info("started creating the corpus")
        self.index_corpus = [word_tokenize(i) for i in self.index_dataset.get_corpus(cfg.index_reader.field)]
        self.bm25 = BM25Okapi(self.index_corpus)
        self.tokenized_queries = [word_tokenize(i) for i in self.dataset_wrapper.get_corpus(cfg.query_field)]
        logger.info("finished creating the corpus")


def knn_search(tokenized_query, is_train, idx, num_candidates=1, num_ice=1):
    bm25 = bm25_global
    scores = bm25.get_scores(tokenized_query)
    near_ids = list(np.argsort(scores)[::-1][:max(num_candidates, num_ice)])
    near_ids = near_ids[1:] if is_train else near_ids
    near_ids = [int(a) for a in near_ids]
    return near_ids[:num_ice], [[i] for i in near_ids[:num_candidates]], idx


def search(tokenized_query, is_train, idx, num_candidates, num_ice):
    """for BM25, we simply random select subsets"""
    if num_ice == 1 or num_candidates == 1:
        return knn_search(tokenized_query, is_train, idx,
                          num_candidates=num_candidates, num_ice=num_ice)

    candidates = knn_search(tokenized_query, is_train, idx, num_ice=100)[0]
    # add topk as one of the candidates
    ctxs_candidates = [candidates[:num_ice]]
    while len(ctxs_candidates) < num_candidates:
        # ordered by sim score
        samples_ids = np.random.choice(len(candidates), num_ice, replace=False)
        samples_ids = sorted(samples_ids)
        candidate = [candidates[i] for i in samples_ids]
        if candidate not in ctxs_candidates:
            ctxs_candidates.append(candidate)
    return ctxs_candidates[0], ctxs_candidates, idx


def _search(args):
    return search(*args)


def find(cfg):
    global bm25_global
    knn_finder = BM25Finder(cfg)
    bm25_global = knn_finder.bm25

    def set_global_object(bm25):
        global bm25_global
        bm25_global = bm25

    pool = multiprocessing.Pool(processes=16, initializer=set_global_object, initargs=(knn_finder.bm25,))

    cntx_pre = [[tokenized_query, knn_finder.is_train, idx, knn_finder.num_candidates, knn_finder.num_ice]
                for idx, tokenized_query in enumerate(knn_finder.tokenized_queries)]

    data_list = list(knn_finder.dataset_wrapper.dataset)
    cntx_post = []
    with tqdm(total=len(cntx_pre)) as pbar:
        for i, res in enumerate(pool.imap_unordered(_search, cntx_pre)):
            pbar.update()
            cntx_post.append(res)
    for ctxs, ctxs_candidates, idx in cntx_post:
        data_list[idx]['ctxs'] = ctxs
        data_list[idx]['ctxs_candidates'] = ctxs_candidates

    with open(cfg.output_file, "w") as f:
        json.dump(data_list, f)
    return data_list


@hydra.main(config_path="configs", config_name="bm25_retriever")
def main(cfg):
    logger.info(cfg)
    set_seed(43)
    find(cfg)


if __name__ == "__main__":
    main()

import json
import logging
import faiss
import hydra
import hydra.utils as hu
import numpy as np
import torch
import tqdm
import os
from transformers import set_seed
from torch.utils.data import DataLoader
from src.utils.dpp_map import fast_map_dpp, k_dpp_sampling
from src.utils.misc import parallel_run, partial
from src.utils.collators import DataCollatorWithPaddingAndCuda
from src.models.biencoder import BiEncoder

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, cfg) -> None:
        self.cuda_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.cuda_device)
        self.dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)

        model_config = hu.instantiate(cfg.model_config)
        if cfg.pretrained_model_path is not None:
            self.model = BiEncoder.from_pretrained(cfg.pretrained_model_path, config=model_config)
        else:
            self.model = BiEncoder(model_config)

        self.model = self.model.to(self.cuda_device)
        self.model.eval()

        self.output_file = cfg.output_file
        self.num_candidates = cfg.num_candidates
        self.num_ice = cfg.num_ice
        self.is_train = cfg.dataset_reader.dataset_split == "train"

        self.dpp_search = cfg.dpp_search
        self.dpp_topk = cfg.dpp_topk
        self.mode = cfg.mode
        # if os.path.exists(cfg.faiss_index):
        #     logger.info(f"Loading faiss index from {cfg.faiss_index}")
        #     self.index = faiss.read_index(cfg.faiss_index)
        # else:
        self.index = self.create_index(cfg)

    def create_index(self, cfg):
        logger.info("Building faiss index...")
        index_reader = hu.instantiate(cfg.index_reader)
        co = DataCollatorWithPaddingAndCuda(tokenizer=index_reader.tokenizer, device=self.cuda_device)
        dataloader = DataLoader(index_reader, batch_size=cfg.batch_size, collate_fn=co)

        index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        res_list = self.forward(dataloader, encode_ctx=True)
        id_list = np.array([res['metadata']['id'] for res in res_list])
        embed_list = np.stack([res['embed'] for res in res_list])
        index.add_with_ids(embed_list, id_list)
        faiss.write_index(index, cfg.faiss_index)
        logger.info(f"Saving faiss index to {cfg.faiss_index}, size {len(index_reader)}")
        return index

    def forward(self, dataloader, **kwargs):
        res_list = []
        for i, entry in enumerate(tqdm.tqdm(dataloader)):
            with torch.no_grad():
                metadata = entry.pop("metadata")
                res = self.model.encode(**entry, **kwargs)
            res = res.cpu().detach().numpy()
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res_list

    def find(self):
        res_list = self.forward(self.dataloader)
        for res in res_list:
            res['entry'] = self.dataset_reader.dataset_wrapper[res['metadata']['id']]

        if self.dpp_search:
            logger.info(f"Using scale_factor={self.model.scale_factor}; mode={self.mode}")
            func = partial(dpp, num_candidates=self.num_candidates, num_ice=self.num_ice,
                           mode=self.mode, dpp_topk=self.dpp_topk, scale_factor=self.model.scale_factor)
        else:
            func = partial(knn, num_candidates=self.num_candidates, num_ice=self.num_ice)
        data = parallel_run(func=func, args_list=res_list, initializer=set_global_object,
                            initargs=(self.index, self.is_train))

        with open(self.output_file, "w") as f:
            json.dump(data, f)


def set_global_object(index, is_train):
    global index_global, is_train_global
    index_global = index
    is_train_global = is_train


def knn(entry, num_candidates=1, num_ice=1):
    embed = np.expand_dims(entry['embed'], axis=0)
    near_ids = index_global.search(embed, max(num_candidates, num_ice)+1)[1][0].tolist()
    near_ids = near_ids[1:] if is_train_global else near_ids

    entry = entry['entry']
    entry['ctxs'] = near_ids[:num_ice]
    entry['ctxs_candidates'] = [[i] for i in near_ids[:num_candidates]]
    return entry


def get_kernel(embed, candidates, scale_factor):
    near_reps = np.stack([index_global.index.reconstruct(i) for i in candidates], axis=0)
    # normalize first
    embed = embed / np.linalg.norm(embed)
    near_reps = near_reps / np.linalg.norm(near_reps, keepdims=True, axis=1)

    rel_scores = np.matmul(embed, near_reps.T)[0]
    # to make kernel-matrix non-negative
    rel_scores = (rel_scores + 1) / 2
    # to prevent overflow error
    rel_scores -= rel_scores.max()
    # to balance relevance and diversity
    rel_scores = np.exp(rel_scores / (2 * scale_factor))
    sim_matrix = np.matmul(near_reps, near_reps.T)
    # to make kernel-matrix non-negative
    sim_matrix = (sim_matrix + 1) / 2
    kernel_matrix = rel_scores[None] * sim_matrix * rel_scores[:, None]
    return near_reps, rel_scores, kernel_matrix


def random_sampling(num_total, num_ice, num_candidates, pre_results=None):
    ctxs_candidates_idx = [] if pre_results is None else pre_results
    while len(ctxs_candidates_idx) < num_candidates:
        # ordered by sim score
        samples_ids = np.random.choice(num_total, num_ice, replace=False).tolist()
        samples_ids = sorted(samples_ids)
        if samples_ids not in ctxs_candidates_idx:
            ctxs_candidates_idx.append(samples_ids)
    return ctxs_candidates_idx


def dpp(entry, num_candidates=1, num_ice=1, mode="map", dpp_topk=100, scale_factor=0.1):
    candidates = knn(entry, num_ice=dpp_topk)['ctxs']
    embed = np.expand_dims(entry['embed'], axis=0)
    near_reps, rel_scores, kernel_matrix = get_kernel(embed, candidates, scale_factor)

    if mode == "cand_random" or np.isinf(kernel_matrix).any() or np.isnan(kernel_matrix).any():
        if np.isinf(kernel_matrix).any() or np.isnan(kernel_matrix).any():
            logging.info("Inf or NaN detected in Kernal_matrix, using random sampling instead!")
        topk_results = list(range(num_ice))
        ctxs_candidates_idx = [topk_results]
        ctxs_candidates_idx = random_sampling(num_total=dpp_topk,  num_ice=num_ice,
                                              num_candidates=num_candidates,
                                              pre_results=ctxs_candidates_idx)
    elif mode == "pure_random":
        ctxs_candidates_idx = [candidates[:num_ice]]
        ctxs_candidates_idx = random_sampling(num_total=index_global.ntotal,  num_ice=num_ice,
                                              num_candidates=num_candidates,
                                              pre_results=ctxs_candidates_idx)
        entry = entry['entry']
        entry['ctxs'] = ctxs_candidates_idx[0]
        entry['ctxs_candidates'] = ctxs_candidates_idx
        return entry
    elif mode == "cand_k_dpp":
        topk_results = list(range(num_ice))
        ctxs_candidates_idx = [topk_results]
        ctxs_candidates_idx = k_dpp_sampling(kernel_matrix=kernel_matrix, rel_scores=rel_scores,
                                             num_ice=num_ice, num_candidates=num_candidates,
                                             pre_results=ctxs_candidates_idx)
    else:
        # MAP inference
        map_results = fast_map_dpp(kernel_matrix, num_ice)
        map_results = sorted(map_results)
        ctxs_candidates_idx = [map_results]

    ctxs_candidates = []
    for ctxs_idx in ctxs_candidates_idx:
        ctxs_candidates.append([candidates[i] for i in ctxs_idx])
    assert len(ctxs_candidates) == num_candidates

    entry = entry['entry']
    entry['ctxs'] = ctxs_candidates[0]
    entry['ctxs_candidates'] = ctxs_candidates
    return entry


@hydra.main(config_path="configs", config_name="dense_retriever")
def main(cfg):
    logger.info(cfg)
    set_seed(43)
    dense_retriever = DenseRetriever(cfg)
    dense_retriever.find()


if __name__ == "__main__":
    main()

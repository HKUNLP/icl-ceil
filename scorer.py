import glob
import json
import os
import logging
import hydra
import torch
import tqdm
from transformers import set_seed
from accelerate import Accelerator
from inferencer import Inferencer
from src.utils.misc import save_json


logger = logging.getLogger(__name__)


class Scorer(Inferencer):

    def forward(self):
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader

        res = []
        for i, entry in enumerate(dataloader):
            metadata = entry.pop("metadata")
            # fix position id error when using left padding, note that calling generate() doesn't affected by this,
            # but here we don't generate new tokens, so we have to fix it manually.
            position_ids = entry.attention_mask.long().cumsum(-1) - 1
            # replace -1 with 1, final position id is like [1, 1, ..., 1, 0, 1, 2, 3...],
            # where prior 1s is ignored in self-attention
            position_ids.masked_fill_(entry.attention_mask == 0, 1)

            with torch.no_grad():
                output = self.model(input_ids=entry.input_ids, attention_mask=entry.attention_mask,
                                    position_ids=position_ids)

                loss = self.nll_loss(entry=entry, output=output)

            for mdata, loss in zip(metadata, loss):
                mdata['score'] = loss

            if i == 0:
                logger.info(f"Prompt: {metadata[0]['prompt']}")
                logger.info(f"Number of ICE: {len(metadata[0]['ice_prompts_list'])}")

            res.extend(metadata)

        with open(f"{self.output_file}tmp_{self.accelerator.device}.bin", "w") as f:
            json.dump(res, f)

    def nll_loss(self, entry, output):
        shift_logits = output.logits[..., :-1, :].contiguous()
        shift_labels = entry.input_ids[..., 1:].contiguous()
        pad_token_id = self.dataset_reader.tokenizer.pad_token_id
        # entry.labels is already padded with pad_token_id, we further pad it to full length
        pad_mask = torch.nn.functional.pad(entry.labels,
                                           (shift_labels.shape[-1] - entry.labels.shape[-1], 0),
                                           value=pad_token_id)
        shift_labels.masked_fill_(pad_mask == pad_token_id, pad_token_id)

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())
        answer_lens = (entry.labels != pad_token_id).sum(-1)
        loss = loss.sum(-1) / answer_lens
        loss = loss.cpu().detach().numpy().tolist()
        return loss

    def write_results(self):
        data = []
        for i, path in enumerate(glob.glob(f"{self.output_file}tmp_*.bin")):
            with open(path) as f:
                one_device = json.load(f)
                logger.info(f"device: {i}, idx {[i['idx'] for i in one_device][:200]}...")
                data.extend(one_device)

        # grouping results by uid
        example_dict = {}
        uid_field = 'idx'
        for entry in data:
            ctxs = {"ctxs": entry.pop('ctxs'), "score": entry.pop("score")}
            if entry[uid_field] not in example_dict:
                entry['ctxs_candidates'] = [ctxs]
                example_dict[entry[uid_field]] = entry
            else:
                example_dict[entry[uid_field]]['ctxs_candidates'].append(ctxs)

        example_list = list(example_dict.values())
        mrr = 0
        num_candidates = len(example_list[0]['ctxs_candidates'])
        for entry in example_list:
            assert len(entry['ctxs_candidates']) == num_candidates, f"{len(entry['ctxs_candidates'])}!={num_candidates}"

            sorted_tuple = sorted(enumerate(entry['ctxs_candidates']), key=lambda x: x[1]['score'])
            entry['ctxs_candidates'] = [i[1]['ctxs'] for i in sorted_tuple]
            entry['ctxs'] = entry['ctxs_candidates'][0]  # set top-scored cand to ctxs
            mrr += 1/([i[0] for i in sorted_tuple].index(0)+1)
        logger.info(f"MRR: {mrr/len(example_list)}")

        save_json(self.output_file, example_list)

        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            os.remove(path)


@hydra.main(config_path="configs", config_name="scorer")
def main(cfg):
    logger.info(cfg)
    set_seed(43)
    accelerator = Accelerator()
    scorer = Scorer(cfg, accelerator)

    scorer.forward()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        scorer.write_results()


if __name__ == "__main__":
    main()

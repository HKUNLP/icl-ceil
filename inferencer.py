import glob
import json
import os
import logging
import hydra
import hydra.utils as hu
import torch
import tqdm
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import set_seed
from src.metrics import get_metric
from src.utils.collators import DataCollatorWithPaddingAndCuda
from src.utils.statistics import show_statistics
from src.models.api_client import run_api
from src.utils.misc import parallel_run, save_json
from src.models.model import ppl_generate

logger = logging.getLogger(__name__)


class Inferencer:
    def __init__(self, cfg, accelerator=None) -> None:
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        self.gen_field = cfg.dataset_reader.field

        self.accelerator = accelerator
        self.output_file = cfg.output_file
        # OmegaConf DictConfig to dict
        self.generation_kwargs = OmegaConf.to_object(cfg.model_config.generation_kwargs)
        self.evaluator = get_metric(cfg.task_name)

        self.model, self.dataloader = self.init_model_dataloader(cfg)

    def init_model_dataloader(self, cfg):
        self.dataset_reader.shard(self.accelerator)

        if self.accelerator.is_main_process:
            logger.info(f"Statistics after sharding: ")
            show_statistics(self.dataset_reader.encoded_dataset, "main dataset")
            show_statistics(self.dataset_reader.index_reader.encoded_dataset, "index dataset")

        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.accelerator.device)
        dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)

        model = hu.instantiate(cfg.model_config.model).eval()
        model = self.accelerator.prepare(model)

        if hasattr(model, "module"):
            model = model.module

        return model, dataloader

    def forward(self):
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader

        avg_ice_num = 0
        res = []
        for i, entry in enumerate(dataloader):
            metadata = entry.pop("metadata")
            if 'choices' in self.dataset_reader.dataset_wrapper.field_getter:
                # for classification tasks, we compare the ppl of provided generation_choices as generation
                choices = [self.dataset_reader.dataset_wrapper.get_field(meta, 'choices') for meta in metadata]
                choices_list = list(zip(*choices))
                preds = ppl_generate([meta['prompt'] for meta in metadata],
                                     model=self.model,
                                     tokenizer=self.dataset_reader.tokenizer,
                                     choices_list=choices_list,
                                     device=self.accelerator.device)
                for mdata, pred in zip(metadata, preds):
                    mdata['generated'] = pred
                    avg_ice_num += len(mdata['ice_prompts_list'])
            else:
                with torch.no_grad():
                    outputs = self.model.generate(input_ids=entry.input_ids,
                                                  attention_mask=entry.attention_mask,
                                                  eos_token_id=self.dataset_reader.tokenizer.encode("\n")[0],
                                                  pad_token_id=self.dataset_reader.tokenizer.pad_token_id,
                                                  do_sample=False,  # always use greedy decode here
                                                  **self.generation_kwargs)
                    prompt_len = int(entry.attention_mask.shape[1])
                    for mdata, output in zip(metadata, outputs.tolist()):
                        generated = self.dataset_reader.tokenizer.decode(output[prompt_len:])
                        mdata['generated'] = generated.strip(self.dataset_reader.tokenizer.pad_token).strip()
                        avg_ice_num += len(mdata['ice_prompts_list'])

            res.extend(metadata)

            if i == 0:
                logger.info(f"Prompt: {metadata[0]['prompt']}")
                logger.info(f"Generated: {metadata[0]['generated']}")
                logger.info(f"Number of ICE: {len(metadata[0]['ice_prompts_list'])}")

        save_json(f"{self.output_file}tmp_{self.accelerator.device}.bin", res)

        logger.info(f"Average number of in-context examples after truncating is {avg_ice_num / len(res)}")

    def write_results(self):
        data = []
        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            with open(path) as f:
                data.extend(json.load(f))
        # from src.utils.misc import load_json
        # data = load_json(self.output_file)
        preds = [i['generated'] for i in data]
        metric = self.evaluator.evaluate(preds, data)
        logger.info(f"metric: {str(metric)}")

        save_json(self.output_file, data)

        for path in glob.glob(f"{self.output_file}tmp_*.bin"):
            os.remove(path)
        return data


class APInferencer(Inferencer):

    def init_model_dataloader(self, cfg):
        model = hu.instantiate(cfg.model_config.model)
        dataloader = self.dataset_reader
        return model, dataloader

    def forward(self):
        prompts = [entry['metadata']['prompt'] for entry in self.dataloader]
        if 'choices' in self.dataset_reader.dataset_wrapper.field_getter:
            choices = [self.dataset_reader.dataset_wrapper.get_field(entry['metadata'], 'choices')
                       for entry in self.dataloader]
            args_list = list(zip(prompts, choices))
        else:
            args_list = prompts
        logger.info(str(prompts[0]))
        responses = parallel_run(run_api, args_list=args_list,
                                 n_processes=self.model.n_processes,
                                 client=self.model,
                                 **self.generation_kwargs)

        data = []
        for i, (entry, response) in enumerate(zip(self.dataloader, responses)):
            if i == 0:
                logger.info(prompts[i])
                logger.info('\n***\n'.join([str(i) for i in response][:3]))
            entry['metadata']['generated'] = response[0]['text']
            data.append(entry['metadata'])

        save_json(self.output_file, data)

        avg_ice_num = sum([len(i['ice_prompts_list']) for i in data])/len(data)
        logger.info(f"Average number of in-context examples after truncating is {avg_ice_num}")
        preds = [i['generated'] for i in data]
        metric = self.evaluator.evaluate(preds, data)
        logger.info(f"metric: {str(metric)}")


@hydra.main(config_path="configs", config_name="inferencer")
def main(cfg):
    logger.info(cfg)
    set_seed(43)
    if cfg.model_config.model_type == 'hf':
        accelerator = Accelerator()
        inferencer = Inferencer(cfg, accelerator)
        inferencer.forward()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            inferencer.write_results()
    else:
        inferencer = APInferencer(cfg)
        inferencer.forward()


if __name__ == "__main__":
    main()

#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY=  # change to your wandb account
export WANDB_API_KEY=  # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

gpu=2
method=dpp-epr-random
num_ice=50
port=9927

#model_name=gpt2-large
#n_tokens=700
#scr_batch_size=128
#inf_batch_size=48

model_name=EleutherAI/gpt-neo-2.7B
n_tokens=1600
scr_batch_size=8
inf_batch_size=8

task_name=mrpc
#for scale_factor in 0.01 0.05 0.1
for scale_factor in 0.1
do
  export WANDB_TAGS="${method},${task_name},${model_name}"
  run_dir=output/${method}/${task_name}/${model_name}
  index_data=index_data/${task_name}/index_dataset.json
  mkdir -p ${run_dir}
  mkdir -p index_data/${task_name}

  epr_model=output/epr/${task_name}/${model_name}/bert-fix_ctx-shared-bs64

  retrieve_file=${run_dir}/retrieved.json
  python dense_retriever.py \
      hydra.run.dir=${run_dir}/dense_retriever \
      output_file=${retrieve_file} \
      task_name=${task_name} \
      dataset_reader.dataset_split=train \
      +dataset_reader.ds_size=44000 \
      index_reader.dataset_path=${index_data} \
      pretrained_model_path=${epr_model} \
      model_config.norm_embed=true \
      faiss_index=${run_dir}/index \
      dpp_search=true \
      dpp_topk=100 \
      num_ice=16 \
      num_candidates=50 \
      model_config.scale_factor=${scale_factor} \
      mode=cand_random


  scored_file=${run_dir}/scored.json
  accelerate launch --num_processes ${gpu} --main_process_port ${port}  scorer.py \
      hydra.run.dir=${run_dir}/scorer \
      task_name=${task_name} \
      output_file=${scored_file} \
      batch_size=${scr_batch_size} \
      model_name=${model_name} \
      dataset_reader.dataset_path=${retrieve_file} \
      dataset_reader.n_tokens=${n_tokens} \
      index_reader.dataset_path=${index_data}


  run_name=base-mg0.02-s${scale_factor}-fix
  run_dir=${run_dir}/${run_name}
  accelerate launch --num_processes ${gpu} --main_process_port ${port}  retriever_trainer.py \
      hydra.run.dir=${run_dir}/trainer \
      task_name=${task_name} \
      pair_wise=true \
      dataset_reader.dataset_path=${scored_file} \
      index_reader.dataset_path=${index_data} \
      training_args.output_dir=${run_dir} \
      training_args.run_name=${run_name} \
      pretrained_model_path=${epr_model} \
      training_args.num_train_epochs=30 \
      training_args.per_device_train_batch_size=64 \
      training_args.per_device_eval_batch_size=64 \
      training_args.gradient_accumulation_steps=1 \
      model_config.dpp_training=true \
      model_config.norm_embed=true \
      model_config.margin=0.02 \
      model_config.scale_factor=${scale_factor}


  retrieve_file=${run_dir}/train_retrieved.json
  python dense_retriever.py \
      hydra.run.dir=${run_dir}/dense_retriever \
      output_file=${retrieve_file} \
      num_ice=${num_ice} \
      task_name=${task_name} \
      index_reader.dataset_path=${index_data} \
      pretrained_model_path=${run_dir} \
      faiss_index=${run_dir}/index \
      model_config.norm_embed=true \
      model_config.scale_factor=${scale_factor} \
      dpp_search=true \
      dpp_topk=100 \
      mode=map


  pred_file=${run_dir}/pred.json
  accelerate launch --num_processes ${gpu} --main_process_port ${port}  inferencer.py \
      hydra.run.dir=${run_dir}/inferencer \
      task_name=${task_name} \
      dataset_reader.dataset_path=${retrieve_file} \
      dataset_reader.n_tokens=${n_tokens} \
      index_reader.dataset_path=${index_data} \
      output_file=${pred_file} \
      model_name=${model_name} \
      batch_size=${inf_batch_size}
done



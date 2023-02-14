#!/bin/bash
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

method=bm25
port=2041

model_config=hf-gen_a
model_name=EleutherAI/gpt-neo-2.7B
n_tokens=1600
inf_batch_size=8
gpu=2

#model_config=api-gen_a
#model_name=code-davinci-002
#n_tokens=7000
#inf_batch_size=8
#ds_size=1000
#gpu=1

for task_name in mrpc
do
  run_dir=output/${method}/${task_name}/${model_name}
  index_data=index_data/${task_name}/index_dataset.json
  retrieve_file=${run_dir}/retrieved.json
  pred_file=${run_dir}/pred.json
  mkdir -p ${run_dir}
  mkdir -p index_data/${task_name}

  python bm25_retriever.py \
     hydra.run.dir=${run_dir}/bm25_retriever \
     output_file=${retrieve_file} \
     task_name=${task_name} \
     index_reader.dataset_path=${index_data}
#      ds_size=${ds_size}

  accelerate launch --num_processes ${gpu} --main_process_port ${port}  inferencer.py \
      hydra.run.dir=${run_dir}/inferencer \
      task_name=${task_name} \
      dataset_reader.dataset_path=${retrieve_file} \
      dataset_reader.n_tokens=${n_tokens} \
      index_reader.dataset_path=${index_data} \
      output_file=${pred_file} \
      model_name=${model_name} \
      batch_size=${inf_batch_size} \
      model_config=${model_config}
done



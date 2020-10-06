import fire
from examples.deebert.run_glue_deebert_pp import dbpmain


def main(
        data_dir="",
        model_type="",
        model_name_or_path="",
        task_name="",
        output_dir="",
        plot_data_dir="./plotting/",
        config_name="",
        tokenizer_name="",
        cache_dir="",
        max_seq_length=128,
        do_train=False,
        do_eval=False,
        evaluate_during_training=False,
        do_lower_case=False,
        eval_each_highway=False,
        eval_after_first_stage=False,
        eval_highway=False,
        per_gpu_train_batch_size=8,
        per_gpu_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        num_train_epochs=3.0,
        max_steps=-1,  # "If > 0: set total number of training steps to perform. Override num_train_epochs."
        warmup_steps=0,  # "Linear warmup over warmup_steps."
        early_exit_entropy=-1.,  # "Entropy threshold for early exit."
        logging_steps=50,  # "Log every X updates steps."
        save_steps=50,  # "Save checkpoint every X updates steps."
        eval_all_checkpoints=False,
        # "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
        no_cuda=False,  # "Avoid using CUDA when available"
        overwrite_output_dir=False,  # "Overwrite the content of the output directory"
        overwrite_cache=False,  # "Overwrite the cached training and evaluation sets"
        seed=42,  # "random seed for initialization"
        fp16=False,  # "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
        fp16_opt_level="O1",
        # "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']." "See details at https://nvidia.github.io/apex/amp.html",
        local_rank=-1,  # "For distributed training: local_rank"
        server_ip="",  # "For distant debugging."
        server_port="",  # " For distant debugging."
):
      dbpmain(**locals().copy())


if __name__ == '__main__':
    fire.Fire(main)

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=/h/xinji/projects/GLUE

MODEL_TYPE=bert  # bert or roberta
MODEL_SIZE=base  # base or large
DATASET=MRPC  # SST-2, MRPC, RTE, QNLI, QQP, or MNLI

MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
if [ $MODEL_TYPE = 'bert' ]
then
  MODEL_NAME=${MODEL_NAME}-uncased
fi


python -u run_glue_deebert.py  \
  --model_type $MODEL_TYPE \
  --model_name_or_path ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/two_stage \
  --task_name $DATASET \
  --do_eval \
  --do_lower_case \
  --data_dir $PATH_TO_DATA/$DATASET \
  --output_dir ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/two_stage \
  --plot_data_dir ./results/ \
  --max_seq_length 128 \
  --eval_each_highway \
  --eval_highway \
  --overwrite_cache \
  --per_gpu_eval_batch_size=1

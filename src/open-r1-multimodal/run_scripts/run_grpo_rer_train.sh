# cd src/open-r1-multimodal

export DEBUG_MODE="true"

RUN_NAME="Qwen2.5-VL-3B-GRPO-RER"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

WORKDIR=${1:-./output/$RUN_NAME}
echo $WORKDIR
mkdir -p $WORKDIR

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open-r1-multimodal/src/open_r1/grpo_rer_multiturn.py \
    --deepspeed src/open-r1-multimodal/local_scripts/zero3_offload.json \
    --output_dir $WORKDIR \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name src/open-r1-multimodal/data_config/rer.yaml \
    --image_root data \
    --max_prompt_length 1024 \
    --max_completion_length 256 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --dataloader_num_workers 2 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    | tee $WORKDIR/train.log

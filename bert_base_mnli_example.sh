export CUDA_VISIBLE_DEVICES=0
export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0

# FIXME : batch_size
export batch_size=64
export num_workers=8
export gradient_accumulation_steps=1
export lr_scheduler_type="cosine"
export warmup_steps=100
export max_seq_length=256
export learning_rate="2e-5"

export logging_steps=0
export eval_steps=200
export save_steps=100
export save_total_limit=5

export WANDB_ENTITY="taehyunzzz"
export WANDB_PROJECT="diffmoebert"

# FIXME : task
task_name=$1
if [[ ${task_name} == "mnli" ]]; then
    export num_train_epochs=5
elif [[ ${task_name} == "sst2" ]]; then
    export num_train_epochs=10
elif [[ ${task_name} == "rte" ]]; then
    export num_train_epochs=10
elif [[ ${task_name} == "cola" ]]; then
    export num_train_epochs=10
else
    export task_name=mnli
    export num_train_epochs=5
fi

# FIXME : model
export model_name_or_path="bert-base-cased"

# FIXME : MODE
export MODE="dense"
# export MODE="importance"
# export MODE="moe"

export run_name="${task_name}_${MODE}_ba${batch_size}"
export output_dir="ckpt/${MODE}/${task_name}"
mkdir -p ${output_dir}
mkdir -p ${output_dir}/model
mkdir -p ${output_dir}/log

# -m \
# torch.distributed.launch \
# --nproc_per_node=$num_gpus \

CMD="
    python \
    examples/text-classification/run_glue.py \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --warmup_steps ${warmup_steps} \
    --model_name_or_path ${model_name_or_path} \
    --task_name ${task_name} \
    --max_seq_length ${max_seq_length} \
    --dataloader_num_workers ${num_workers} \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${num_train_epochs} \
    --output_dir $output_dir/model \
    --overwrite_output_dir \

    --logging_steps ${logging_steps} \
    --logging_dir $output_dir/log \

    --report_to wandb \
    --run_name ${run_name} \

    --evaluation_strategy steps \
    --eval_steps ${eval_steps} \

    --save_strategy steps \
    --save_steps ${save_steps} \
    --save_total_limit ${save_total_limit} \

    --warmup_ratio 0.0 \
    --seed 0 \
    --weight_decay 0.1 \
    --fp16 \
"

echo ${CMD}

if [[ ${MODE} == "dense" ]]; then
    echo "Train dense model"
    CMD+="
            --do_eval \
            --do_train
        "

elif [[ ${MODE} == "importance" ]]; then
    echo "Preprocess importance scores"
    CMD+="
            --do_eval \
            --preprocess_importance y
        "

elif [[ ${MODE} == "moe" ]]; then
    echo "Finetune MoEBERT"
    CMD+="
            --moebert_load_importance importance_files/importance_mnli.pkl \
            --moebert moe \
            --moebert_distill 5.0 \
            --moebert_expert_num 4 \
            --moebert_expert_dim 768 \
            --moebert_expert_dropout 0.1 \
            --moebert_load_balance 0.0 \
            --moebert_route_method hash-random \
            --moebert_share_importance 512 \
            --do_train
        "
fi

${CMD}
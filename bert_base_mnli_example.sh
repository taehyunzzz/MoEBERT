export CUDA_VISIBLE_DEVICES=1
export MASTER_ADDR=localhost
export MASTER_PORT=9100
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0

###########################################
# FIXME : TRAINER
###########################################
export num_workers=8
export lr_scheduler_type="linear"

export gradient_accumulation_steps=1
export warmup_steps=100
export max_seq_length=256

export logging_steps=0
export eval_steps=200
export save_steps=100
export save_total_limit=5

export WANDB_ENTITY="taehyunzzz"
export WANDB_PROJECT="diffmoebert"

###########################################
# FIXME : TASK (set as arg)
###########################################
task_name=$1
if [[ 1 ]]; then
    if [[ ${task_name} == "cola" ]]; then
        export learning_rate="2e-5"
        export batch_size=8
        export num_train_epochs=10
        export weight_decay=0.0
        export moebert_distill=3.0

    elif [[ ${task_name} == "mnli" ]]; then
        export learning_rate="5e-5"
        export batch_size=64
        export num_train_epochs=5
        export weight_decay=0.00
        export moebert_distill=5.0

    elif [[ ${task_name} == "mrpc" ]]; then
        export learning_rate="3e-5"
        export batch_size=8
        export num_train_epochs=5
        export weight_decay=0.0
        export moebert_distill=2.0

    elif [[ ${task_name} == "qnli" ]]; then
        export learning_rate="1e-5"
        export batch_size=32
        export num_train_epochs=5
        export weight_decay=0.00
        export moebert_distill=2.0

    elif [[ ${task_name} == "qqp" ]]; then
        export learning_rate="3e-5"
        export batch_size=64
        export num_train_epochs=5
        export weight_decay=0.00
        export moebert_distill=1.0

    elif [[ ${task_name} == "rte" ]]; then
        export learning_rate="1e-5"
        export batch_size=8
        export num_train_epochs=10
        export weight_decay=0.01
        export moebert_distill=1.0

    elif [[ ${task_name} == "sst2" ]]; then
        export learning_rate="2e-5"
        export batch_size=16
        export num_train_epochs=5
        export weight_decay=0.0
        export moebert_distill=1.0

    else
        echo "Wrong task ${task_name}. Running MNLI"
        export learning_rate="5e-5"
        export batch_size=64
        export num_train_epochs=5
        export weight_decay=0.00
        export moebert_distill=5.0

    fi
fi

###########################################
# FIXME : MOE CONFIG
###########################################
random_seed=0
moebert_expert_num=8

# moebert_expert_dim=768
# moebert_expert_dim=1024
# moebert_expert_dim=2048
moebert_expert_dim=3072

# moebert_share_importance=768
# moebert_share_importance=1024
moebert_share_importance=2048
# moebert_share_importance=3072

moebert_expert_dropout=0.1
moebert_load_balance=0.0
moebert_route_method=hash-random

###########################################
# FIXME : MODE
###########################################
# export MODE="dense"
# export MODE="importance"
# export MODE="dense2moe"
export MODE="moe"

###########################################
# FIXME : CHECKPOINT
###########################################
export ckpt_name="checkpoint-3100"


export run_name="${task_name}_${MODE}_ba${batch_size}_${lr_scheduler_type}"
export output_dir="ckpt/${MODE}/${task_name}"

if [[ ${MODE} == "dense" ]]; then
    export model_name_or_path="bert-base-cased"
    # export model_name_or_path="/home/kimth/workspace/MoEBERT/ckpt/dense/${task_name}/model/${ckpt_name}/"

elif [[ ${MODE} == "importance" ]]; then
    export model_name_or_path="/home/kimth/workspace/MoEBERT/ckpt/dense/${task_name}/model/${ckpt_name}/"

elif [[ ${MODE} == "dense2moe" ]]; then
    export model_name_or_path="/home/kimth/workspace/MoEBERT/ckpt/dense/${task_name}/model/${ckpt_name}/"

elif [[ ${MODE} == "moe" ]]; then
    export model_name_or_path="/home/kimth/workspace/MoEBERT/ckpt/dense/${task_name}/model/${ckpt_name}/"

fi

mkdir -p ${output_dir}
mkdir -p ${output_dir}/model
mkdir -p ${output_dir}/log


export importance_file=importance_files/importance_${task_name}.pkl

# -m \
# torch.distributed.launch \
# --nproc_per_node=$num_gpus \

CMD="
    python \
    examples/text-classification/run_glue.py \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
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
    --seed ${random_seed} \
    --weight_decay ${weight_decay} \
    --fp16 \
"

if [[ ${MODE} == "dense" ]]; then
    echo "Train dense model"
    CMD+="
            --do_eval \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size ${batch_size} \
        "

elif [[ ${MODE} == "importance" ]]; then
    echo "Preprocess importance scores"
    CMD+="
            --do_eval \
            --per_device_eval_batch_size 16 \
            --preprocess_importance y
        "

elif [[ ${MODE} == "dense2moe" ]]; then
    echo "Preprocess importance scores"

    # ACTIVATE moebert_load_expert
    # DEACTIVATE moebert_distill 5.0
    CMD+="
            --moebert_load_importance ${importance_file} \
            --moebert_load_expert True \
            --moebert moe \
            --moebert_expert_num ${moebert_expert_num} \
            --moebert_expert_dim ${moebert_expert_dim} \
            --moebert_expert_dropout ${moebert_expert_dropout} \
            --moebert_load_balance ${moebert_load_balance} \
            --moebert_route_method ${moebert_route_method} \
            --moebert_share_importance ${moebert_share_importance} \
            --do_train
        "
        # Keep do_train, but it will finish automatically after evaluation instead

elif [[ ${MODE} == "moe" ]]; then
    echo "Finetune MoEBERT"
    CMD+="
            --moebert_load_importance ${importance_file} \
            --moebert moe \
            --moebert_load_expert True \
            --moebert_distill ${moebert_distill} \
            --moebert_expert_num ${moebert_expert_num} \
            --moebert_expert_dim ${moebert_expert_dim} \
            --moebert_expert_dropout ${moebert_expert_dropout} \
            --moebert_load_balance ${moebert_load_balance} \
            --moebert_route_method ${moebert_route_method} \
            --moebert_share_importance ${moebert_share_importance} \
            --do_train \
            --do_eval
        "
fi

# RUN CMD
echo ${CMD}
${CMD}
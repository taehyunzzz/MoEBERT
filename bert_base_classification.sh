#/bin/bash

task_name=$1
cuda_device=$2
port_num=$3
mode=$4
moebert_expert_num=$5
moebert_expert_dim=$6
moebert_share_importance=$7
moebert_target_sparsity=$8

export MASTER_ADDR=localhost
export MASTER_PORT=${port_num}
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0

###########################################
# FIXME : TRAINER
###########################################
export num_workers=4
export lr_scheduler_type="linear"

export gradient_accumulation_steps=1
export warmup_steps=100
export max_seq_length=256

export logging_steps=0
export eval_steps=100
# export save_strategy=no
export save_strategy=steps
export save_steps=100
export save_total_limit=1

export WANDB_ENTITY="taehyunzzz"
export WANDB_PROJECT="diffmoebert-ablation"

###########################################
# FIXME : TASK (set as arg)
###########################################
if [[ 1 ]]; then
    if [[ ${task_name} == "rte" ]]; then
        export learning_rate="1e-5"
        export batch_size=8
        export num_train_epochs=10
        export weight_decay=0.01
        export moebert_distill=1.0
        export ckpt_name="checkpoint-3100"

    elif [[ ${task_name} == "cola" ]]; then
        export learning_rate="2e-5"
        export batch_size=8
        export num_train_epochs=10
        export weight_decay=0.0
        export moebert_distill=3.0
        export ckpt_name="checkpoint-10600"

    elif [[ ${task_name} == "mrpc" ]]; then
        export learning_rate="3e-5"
        export batch_size=8
        export num_train_epochs=10 # modified 5->10 to make loss smaller
        export weight_decay=0.0
        export moebert_distill=2.0
        export ckpt_name="checkpoint-4500"

    elif [[ ${task_name} == "sst2" ]]; then
        export learning_rate="2e-5"
        export batch_size=16
        export num_train_epochs=5
        export weight_decay=0.0
        export moebert_distill=1.0
        export ckpt_name="checkpoint-21000"

    elif [[ ${task_name} == "qnli" ]]; then
        export learning_rate="1e-5"
        export batch_size=32
        export num_train_epochs=10
        export weight_decay=0.00
        export moebert_distill=2.0
        export ckpt_name="checkpoint-32650"

    elif [[ ${task_name} == "mnli" ]]; then
        export learning_rate="5e-5"
        export batch_size=64
        export num_train_epochs=10
        export weight_decay=0.00
        export moebert_distill=5.0
        export ckpt_name="checkpoint-61350"

        export eval_steps=$(( ${eval_steps} * 4 ))
        export save_steps=$(( ${save_steps} * 4 ))

    elif [[ ${task_name} == "qqp" ]]; then
        export learning_rate="3e-5"
        export batch_size=64
        export num_train_epochs=10
        export weight_decay=0.00
        export moebert_distill=1.0
        export ckpt_name="checkpoint-5600"

        export eval_steps=$(( ${eval_steps} * 4 ))
        export save_steps=$(( ${save_steps} * 4 ))

    else
        echo "Wrong task ${task_name}. Exiting"
        exit 1
    fi
fi


###########################################
# FIXME : MODE
###########################################
if [[ 1 ]]; then
# export MODE="dense"
# export MODE="importance"
# export MODE="moe"
# export MODE="diffmoe"
export MODE=${mode}
fi

if [[ ${MODE} == "moe" || ${MODE} == "diffmoe" ]]; then
export num_train_epochs=$(( ${num_train_epochs} * 5 ))
echo "Increasing training epochs to ${num_train_epochs}"
fi


###########################################
# FIXME : MOE CONFIG
###########################################
if [[ 1 ]]; then

    random_seed=0
    # moebert_expert_num=16
    # moebert_expert_dim=3072
    # moebert_share_importance=2048

    moebert_expert_dropout=0.1
    moebert_load_balance=0.0
    moebert_route_method=hash-random

    moebert_fixmask_init=False
    moebert_alpha_init=5.0
    moebert_concrete_lower=-1.5
    moebert_concrete_upper=1.5
    moebert_structured=True
    moebert_sparsity_pen=1.25e-7
    
    # diff pruning params
    moebert_learning_rate_alpha=1e-3
    moebert_l0_loss_scale=1e1

fi

###########################################
# FIXME : CHECKPOINT
###########################################

export run_name="${task_name}_${MODE}_ba${batch_size}_${lr_scheduler_type}"

if [[ ${MODE} == "dense" ]]; then
    export model_name_or_path="bert-base-cased"
    # export model_name_or_path="/home/kimth/workspace/MoEBERT/ckpt/dense/${task_name}/model/${ckpt_name}/"

else 
    if [[ -z ${ckpt_name} ]]; then
        echo "[ERROR] VARIABLE ckpt_name not specified for task ${task_name}. Aborting"
        exit 1
    fi

    export model_name_or_path="/home/kimth/workspace/MoEBERT/ckpt/dense/${task_name}/model/${ckpt_name}/"

    if [[ ${MODE} == "moe" || ${MODE} == "diffmoe" ]]; then
        export run_name+="_ex${moebert_expert_num}"
        export run_name+="_dff${moebert_expert_dim}_share${moebert_share_importance}"
    fi

    if [[ ${MODE} == "diffmoe" ]]; then
        export run_name+="_alphalr${moebert_learning_rate_alpha}_sparsity${moebert_target_sparsity}"
    fi

fi

export output_dir="ckpt/${MODE}/${task_name}"
mkdir -p ${output_dir}
mkdir -p ${output_dir}/model
mkdir -p ${output_dir}/log


export importance_file=importance_files/importance_${task_name}.pkl

# -m \
# torch.distributed.launch \
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

    --save_strategy ${save_strategy} \
    --save_steps ${save_steps} \
    --save_total_limit ${save_total_limit} \

    --warmup_ratio 0.0 \
    --seed ${random_seed} \
    --weight_decay ${weight_decay} \

    --local_rank ${cuda_device} \
    --moebert_device ${cuda_device} \
"
    # --fp16 \

if [[ ${MODE} == "dense" ]]; then
    echo "Train dense model"
    CMD+="
            --do_train \
            --do_eval \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size ${batch_size} \
        "

elif [[ ${MODE} == "importance" ]]; then
    echo "Preprocess importance scores"
    CMD+="--do_eval \
         --per_device_eval_batch_size 16 \
         --preprocess_importance y"

    # Merge importance
    mkdir -p importance_files
    CMD2="python merge_importance.py --num_files 1 --task_name ${task_name}"
    CMD3="mv importance_${task_name}.pkl importance_files/importance_${task_name}.pkl"

elif [[ ${MODE} == "moe" || ${MODE} == "diffmoe" ]]; then
    echo "Finetune MoEBERT"
    CMD+="
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size ${batch_size} \
            --moebert_load_importance ${importance_file} \
            --moebert_load_expert True \
            --moebert ${MODE} \
            --moebert_distill ${moebert_distill} \
            --moebert_expert_num ${moebert_expert_num} \
            --moebert_expert_dim ${moebert_expert_dim} \
            --moebert_expert_dropout ${moebert_expert_dropout} \
            --moebert_load_balance ${moebert_load_balance} \
            --moebert_route_method ${moebert_route_method} \
            --moebert_share_importance ${moebert_share_importance} \
            --moebert_fixmask_init ${moebert_fixmask_init} \
            --moebert_alpha_init ${moebert_alpha_init} \
            --moebert_concrete_lower ${moebert_concrete_lower} \
            --moebert_concrete_upper ${moebert_concrete_upper} \
            --moebert_structured ${moebert_structured} \
            --moebert_sparsity_pen ${moebert_sparsity_pen} \
            --moebert_learning_rate_alpha ${moebert_learning_rate_alpha} \
            --moebert_l0_loss_scale ${moebert_l0_loss_scale} \
            --moebert_target_sparsity ${moebert_target_sparsity} \
            --do_train \
            --do_eval
        "
fi

# RUN CMD
echo ${CMD}
${CMD}

if [[ ${MODE} == "importance" ]]; then
    echo ${CMD2}
    ${CMD2}

    echo ${CMD3}
    ${CMD3}
fi

export CUDA_VISIBLE_DEVICES=1
export MASTER_ADDR=localhost
export MASTER_PORT=9101
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
export eval_steps=50
export save_steps=50
export save_total_limit=3

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
        export batch_size=16
        export num_train_epochs=10
        export weight_decay=0.01
        export moebert_distill=1.0
        export ckpt_name="checkpoint-1550"

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
# FIXME : MODE
###########################################
if [[ 1 ]]; then
    # export MODE="dense"
    # export MODE="importance"
    # export MODE="dense2moe"
    export MODE="moe"
    # export MODE="diffmoe"
fi

if [[ ${MODE} == "moe" ]]; then
    export num_train_epochs=$(( ${num_train_epochs} * 2 ))
    echo "Increasing training epochs to ${num_train_epochs}"
elif [[ ${MODE} == "diffmoe" ]]; then
    export num_train_epochs=$(( ${num_train_epochs} * 3 ))
    echo "Increasing training epochs to ${num_train_epochs}"
fi


###########################################
# FIXME : MOE CONFIG
###########################################
if [[ 1 ]]; then
    random_seed=0
    moebert_expert_num=4

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

    moebert_is_diffmoe=$(( ${MODE} == "diffmoe" ))
    moebert_fixmask_init=False
    moebert_alpha_init=5.0
    moebert_concrete_lower=-1.5
    moebert_concrete_upper=1.5
    moebert_structured=True
    moebert_sparsity_pen=1.25e-7
    
    # diff pruning params
    moebert_learning_rate_alpha=3e-2
    moebert_l0_loss_scale=1e1
fi

###########################################
# FIXME : CHECKPOINT
###########################################

export output_dir="ckpt/${MODE}/${task_name}"
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
        export run_name+="_alphalr${moebert_learning_rate_alpha}"
    fi

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
    CMD2="python merge_importance.py --num_files 1"
    CMD3="mv importance.pkl importance_files/importance_${task_name}.pkl"


elif [[ ${MODE} == "dense2moe" ]]; then
    echo "Preprocess importance scores"
    echo "DEPRECATED : EXITING"
    exit 1

    # ACTIVATE moebert_load_expert
    # DEACTIVATE moebert_distill 5.0
    CMD+="
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size ${batch_size} \
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
            --moebert_is_diffmoe ${moebert_is_diffmoe} \
            --moebert_fixmask_init ${moebert_fixmask_init} \
            --moebert_alpha_init ${moebert_alpha_init} \
            --moebert_concrete_lower ${moebert_concrete_lower} \
            --moebert_concrete_upper ${moebert_concrete_upper} \
            --moebert_structured ${moebert_structured} \
            --moebert_sparsity_pen ${moebert_sparsity_pen} \
            --moebert_learning_rate_alpha ${moebert_learning_rate_alpha} \
            --moebert_l0_loss_scale ${moebert_l0_loss_scale} \
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
#!/bin/bash
#SBATCH -J FULL                               # 作业名为 test
#SBATCH -o FULL_%j.out                           # stdout 重定向到 test.out
#SBATCH -e FULL_%j.err                           # stderr 重定向到 test.err
#SBATCH -p gpu02                            # 作业提交的分区为 gpu
#SBATCH -N 1      
#SBATCH --mem 600G                            # 作业申请 1 个节点
#SBATCH -t 1:00:00                            # 任务运行的最长时间为 12 小时
#SBATCH --gres=gpu:8         # 申请 1 卡 A100 80GB

# source ~/gjh_ws/venv/cu12_2/bin/activate

beta=$1
reward_type=$2
safe_loss_type=kl

NUM_TRAIN_EPOCHS=1
# Hyperparameters
NUM_GPUS=$(nvidia-smi -L | wc -l)
TRAIN_BATCH_SIZE=2
EVAL_BATCH_SIZE=2
GRAD_ACCUMULATION_STEPS=2
LEARNING_RATE=2e-5
NUM_TRAIN_EPOCHS=1
MAX_SOURCE_LENGTH=512
MAX_TARGET_LENGTH=128
ds_name=RoleBench/$3

# Environment setup
SRC_ROOT_DIR=SaRFT/sarft
RUN_NAME=FULL_epochs${NUM_TRAIN_EPOCHS}_lr${LEARNING_RATE}_bs$((TRAIN_BATCH_SIZE*NUM_GPUS*GRAD_ACCUMULATION_STEPS))_beta${beta}_${reward_type}_${safe_loss_type}
BACKBONE=Meta-Llama-3-8B-Instruct
BACKBONE_ID=llama3
DATA_DIR=$SRC_ROOT_DIR/datasets/$ds_name
OUTPUT_DIR=$SRC_ROOT_DIR/results/$BACKBONE_ID/$(basename "$DATA_DIR")/$RUN_NAME
DS_CONFIG=$SRC_ROOT_DIR/ds_configs/stage2.config

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
PORT=$(( $RANDOM % 1000 + 32768 ))

# Job execution
export NCCL_P2P_LEVEL=NVL
deepspeed --num_gpus 8 --master_port $PORT $SRC_ROOT_DIR/src/run.py \
    --do_train \
    --model_name_or_path $BACKBONE \
    --output_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --bf16 True \
    --tf32 True \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length $MAX_TARGET_LENGTH \
    --lr_scheduler_type constant \
    --gradiant_update gd \
    --overwrite_output_dir \
    --save_strategy no \
    --evaluation_strategy no \
    --prediction_loss_only True \
    --logging_strategy steps \
    --logging_steps 5 \
    --deepspeed $DS_CONFIG \
    --load_best_model_at_end True \
    --do_lora False \
    --beta $beta \
    --reward_type $reward_type \
    --safe_loss_type $safe_loss_type
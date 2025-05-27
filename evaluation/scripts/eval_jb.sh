#!/bin/bash
#SBATCH -J eval_all_ds                               # 作业名为 test
#SBATCH -o eval_all_ds_%j.out                           # stdout 重定向到 test.out
#SBATCH -e eval_all_ds_%j.err                           # stderr 重定向到 test.err
#SBATCH -p gpu02                           # 作业提交的分区为 gpu
#SBATCH -N 1      
#SBATCH --mem 100G                            # 作业申请 1 个节点
#SBATCH -t 0:05:00                            # 任务运行的最长时间为 12 小时
#SBATCH --gres=gpu:1         # 申请 1 卡 A100 80GB

# Usage: bash eval.sh <model_id>
# Example: bash eval.sh llama3.1_SafeLoRA_LORA_epochs1_lr1e-4_bs32

model_id=$1

decoding_result_dir=SaRFT/decoding/results
datasets=( aim Autodan GCG Codechameleon Cipher )

for ds in ${datasets[@]}; do
    decoding_result_path=$decoding_result_dir/$ds/$model_id.json
    # check if the decoding result exists
    if [ ! -f $decoding_result_path ]; then
        echo "decoding result not found: $decoding_result_path"
        continue
    fi
    python SaRFT/Evaluation/run.py \
        --decoding_result_path $decoding_result_path \
        --dataset_name $ds
done
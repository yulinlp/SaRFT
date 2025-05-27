#!/bin/bash
#SBATCH -J generate_all_ds                               # 作业名为 test
#SBATCH -o generate_all_ds_%j.out                           # stdout 重定向到 test.out
#SBATCH -e generate_all_ds_%j.err                           # stderr 重定向到 test.err
#SBATCH -p gpu02                           # 作业提交的分区为 gpu
#SBATCH -N 1      
#SBATCH --mem 100G                            # 作业申请 1 个节点
#SBATCH -t 1:00:00                            # 任务运行的最长时间为 12 小时
#SBATCH --gres=gpu:1         # 申请 1 卡 A100 80GB

BACKBONE_ID=$1
adapter_path=$2
role=$3
decoding_method=$4
# sys_prompt=""

datasets=( HEx-PHI RoleBench/$role/RoleBench_SPE RoleBench/$role/RoleBench_RAW AdvBench BeaverTails )
# datasets=( RoleBench/$role/RoleBench_SPE RoleBench/$role/RoleBench_RAW )

if [[ $BACKBONE_ID == "llama3.1" ]]; then
    BACKBONE=/Llama-3.1-8B-Instruct-hf
elif [[ $BACKBONE_ID == "llama3" ]]; then
    BACKBONE=/Meta-Llama-3-8B-Instruct
elif [[ $BACKBONE_ID == "gemma2" ]]; then
    BACKBONE=/gemma-2-9b-it
elif [[ $BACKBONE_ID == "mistralv0.3" ]]; then
    BACKBONE=/Mistral-7B-Instruct-v0.3
elif [[ $BACKBONE_ID == "qwen2.5" ]]; then
    BACKBONE=/Qwen2.5-7B-Instruct
else
    BACKBONE=$BACKBONE_ID
    # 尝试从BACKBONE_ID中提取模型名
    if [[ $BACKBONE_ID == *"llama3"* ]]; then
        BACKBONE_ID=llama3
    elif [[ $BACKBONE_ID == *"gemma2"* ]]; then
        BACKBONE_ID="gemma2"
    elif [[ $BACKBONE_ID == *"mistralv0.3"* ]]; then
        BACKBONE_ID="mistralv0.3"
    elif [[ $BACKBONE_ID == *"qwen2.5"* ]]; then
        BACKBONE_ID="qwen2.5"
    fi
fi

if [ $decoding_method == "None" ]; then
    echo "<BACKBONE_ID>: $BACKBONE_ID"
    echo "<adapter_path>: $adapter_path"
    echo "<role>: $role"
    echo "<decoding_method>: $decoding_method"
    echo "<sys_prompt>: $sys_prompt"
    if [[ $adapter_path == "None" ]]; then
        for ds in ${datasets[@]}; do
            bash SaRFT/decoding/scripts/generate.sh \
                --model_name_or_path $BACKBONE \
                --ds $ds 
                # --sys_prompt $sys_prompt
        done
    else
        for ds in ${datasets[@]}; do
            bash SaRFT/decoding/scripts/generate.sh \
                --model_name_or_path $BACKBONE \
                --ds $ds \
                --adapter_path $adapter_path \
                # --sys_prompt $sys_prompt
        done
    fi
else
    echo "<BACKBONE_ID>: $BACKBONE_ID"
    echo "<adapter_path>: $adapter_path"
    echo "<role>: $role"
    echo "<decoding_method>: $decoding_method"
    echo "<sys_prompt>: $sys_prompt"
    if [[ $adapter_path == "None" ]]; then
        for ds in ${datasets[@]}; do
            bash SaRFT/decoding/scripts/generate.sh \
                --model_name_or_path $BACKBONE \
                --ds $ds \
                --decoding_method $decoding_method \
                --model_id ${BACKBONE_ID}_${role}_${decoding_method} \
                # --sys_prompt $sys_prompt
        done
    else
        for ds in ${datasets[@]}; do
            bash SaRFT/decoding/scripts/generate.sh \
                --model_name_or_path $BACKBONE \
                --ds $ds \
                --adapter_path $adapter_path \
                --decoding_method $decoding_method \
                # --sys_prompt $sys_prompt
        done
    fi
fi
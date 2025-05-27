#!/bin/bash
#SBATCH -J Decode                               # 作业名为 Decode
#SBATCH -o Decode_%j.out                           # stdout 重定向到 Decode_<jobid>.out
#SBATCH -e Decode_%j.err                           # stderr 重定向到 Decode_<jobid>.err
#SBATCH -p gpu02                            # 作业提交的分区为 gpu
#SBATCH -N 1                                # 作业申请 1 个节点
#SBATCH --mem 100G                          # 每个节点申请 100GB 内存
#SBATCH -t 0:05:00                          # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:1                        # 申请 1 卡 A100 80GB

# 初始化参数
model_name_or_path=""
save_dir="SaRFT/decoding/results"
dataset_dir="SaRFT/evaluation/datasets"
adapter_path=""
decoding_method=""
few_shot=""
sys_prompt=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name_or_path)
            model_name_or_path="$2"
            shift 2
            ;;
        --save_dir)
            save_dir="$2"
            shift 2
            ;;
        --ds)
            dataset_dir=$dataset_dir/"$2"
            shift 2
            ;;
        --adapter_path)
            adapter_path="$2"
            shift 2
            ;;
        --decoding_method)
            decoding_method="$2"
            shift 2
            ;;
        --few_shot)
            few_shot="$2"
            shift 2
            ;;
        --sys_prompt)
            sys_prompt="$2"
            shift 2
            ;;
        *)  # 未知参数
            extra_args+=("$1")
            shift
            ;;
    esac
done

# 激活虚拟环境


# 构建Python命令
python_cmd=(python SaRFT/decoding/src/generate.py)

# 添加已知参数
[[ -n "$model_name_or_path" ]] && python_cmd+=("--model_name_or_path" "$model_name_or_path")
[[ -n "$save_dir" ]] && python_cmd+=("--save_dir" "$save_dir")
[[ -n "$dataset_dir" ]] && python_cmd+=("--dataset_dir" "$dataset_dir")
[[ -n "$adapter_path" ]] && python_cmd+=("--adapter_path" "$adapter_path")
[[ -n "$decoding_method" ]] && python_cmd+=("--decoding_method" "$decoding_method")
[[ -n "$few_shot" ]] && python_cmd+=("--few_shot" "$few_shot")

# 添加额外参数
if [[ ${#extra_args[@]} -gt 0 ]]; then
    python_cmd+=("${extra_args[@]}")
fi

# 打印Python命令
echo "${python_cmd[@]}"

# 执行Python命令
"${python_cmd[@]}"
#!/bin/bash
# Swift SFT 但节点无流水线并行启动脚本
# 适用于单机训练

set -eo pipefail

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检查配置文件参数
if [ $# -eq 0 ] || [ -z "$1" ]; then
    echo "Error: Config file is required!"
    echo "Usage: $0 <config_file.yaml> [additional_swift_args...]"
    echo ""
    echo "Example:"
    echo "  $0 configs/train_configs/sft/Qwen2_5-VL-7B-MergeSFT.yaml"
    echo ""
    exit 1
fi

CONFIG_FILE=$1

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    echo "Usage: $0 <config_file.yaml> [additional_swift_args...]"
    echo ""
    exit 1
fi

# 设置环境 - 使用脚本所在目录作为项目根目录
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# 从 YAML 配置中解析并设置环境变量
echo "Loading environment variables from config..."
ENV_VARS_SCRIPT=$(python3 "${SCRIPT_DIR}/parse_args/parse_env_vars.py" "$CONFIG_FILE" 2>/dev/null || echo "")
if [ -n "$ENV_VARS_SCRIPT" ]; then
    echo "Setting environment variables from YAML:"
    echo "$ENV_VARS_SCRIPT"
    eval "$ENV_VARS_SCRIPT"
else
    echo "No env_vars found in config, using default environment variables"
    # 默认环境变量 (只设置通用的，模型特定参数通过 model_kwargs 设置)
    # 注意：wandb相关配置已移除默认值，需要在YAML中配置或手动设置
    export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-"1"}
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
    export NPROC_PER_NODE=${NPROC_PER_NODE:-"8"}
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
    export PYTHONWARNINGS=${PYTHONWARNINGS:-"ignore"}
fi

# 解析并设置 Wandb 配置
echo "Loading Wandb configuration from config..."
WANDB_CONFIG_SCRIPT=$(python3 "${SCRIPT_DIR}/parse_args/parse_wandb_config.py" "$CONFIG_FILE" 2>/dev/null || echo "")
if [ -n "$WANDB_CONFIG_SCRIPT" ]; then
    echo "Setting Wandb environment variables from YAML:"
    echo "$WANDB_CONFIG_SCRIPT"
    eval "$WANDB_CONFIG_SCRIPT"
else
    echo "No Wandb config found in YAML"
    echo "Note: If you want to use Wandb, please configure it in your YAML file or set environment variables manually"
fi

# 设置GPU使用
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # 如果没有设置，默认使用所有可用GPU
    echo "CUDA_VISIBLE_DEVICES not set, using all available GPUs"
else
    echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
fi

# 创建日志目录
mkdir -p logs
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")

echo "========================================"
echo "Swift SFT Simple Training"
echo "========================================"
echo "Config: $CONFIG_FILE"
echo "Timestamp: $TIMESTAMP"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-all}"
echo "Starting training..."
echo "========================================"

# 从 YAML 转换为命令行参数
echo "Converting YAML config to command line arguments..."
SWIFT_ARGS=$(python3 "${SCRIPT_DIR}/parse_args/yaml_to_args.py" "$CONFIG_FILE")
echo "Swift arguments: $SWIFT_ARGS"

# 启动训练
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}" \
eval "python3 -m swift.cli.sft $SWIFT_ARGS ${@:2}" 2>&1 | tee logs/swift_sft_${TIMESTAMP}.log

echo "========================================"
echo "Training completed!"
echo "Log: logs/swift_sft_${TIMESTAMP}.log"
echo "========================================"

#!/bin/bash

# ==============================
# Swift SFT DeepSpeed 训练启动脚本
# ==============================

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_yaml> [additional_args...]"
    echo "Example: $0 configs/exp_0_0.yaml"
    exit 1
fi

CONFIG_FILE="$1"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

# 加载 .env 文件（如果存在）
if [ -f ".env" ]; then
    echo "Loading .env file..."
    set -a
    source .env
    set +a
fi

# 解析并设置环境变量
echo "Loading environment variables from config..."
if [ -f "$(dirname "$0")/parse_args/parse_env_vars.py" ]; then
    echo "Setting environment variables from YAML:"
    eval $(python3 "$(dirname "$0")/parse_args/parse_env_vars.py" "$CONFIG_FILE")
else
    echo "Warning: parse_env_vars.py not found, skipping env_vars from YAML"
fi

# 解析并设置 Wandb 配置
echo "Loading Wandb configuration from config..."
if [ -f "$(dirname "$0")/parse_args/parse_wandb_config.py" ]; then
    echo "Setting Wandb environment variables from YAML:"
    eval $(python3 "$(dirname "$0")/parse_args/parse_wandb_config.py" "$CONFIG_FILE")
else
    echo "Warning: parse_wandb_config.py not found, skipping Wandb config from YAML"
fi

# 设置默认环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-"1"}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}
export PYTHONWARNINGS=${PYTHONWARNINGS:-"ignore"}

# GPU 设置
NUM_GPUS=${NPROC_PER_NODE:-8}
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES not set, using first $NUM_GPUS GPUs"
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
    export CUDA_VISIBLE_DEVICES
fi

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Number of processes: $NUM_GPUS"

# 创建日志目录
mkdir -p logs
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")

echo "========================================"
echo "Swift SFT DeepSpeed Training"
echo "========================================"
echo "Config: $CONFIG_FILE"
echo "Timestamp: $TIMESTAMP"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Processes: $NUM_GPUS"
echo "Starting training..."
echo "========================================"

# 从 YAML 转换为命令行参数
echo "Converting YAML config to command line arguments..."
SWIFT_ARGS=$(python3 $(dirname "$0")/parse_args/yaml_to_args.py "$CONFIG_FILE")
echo "Swift arguments: $SWIFT_ARGS"

# 启动多进程训练
# 设置环境 - 使用脚本所在目录作为项目根目录
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
torchrun --nproc_per_node=$NUM_GPUS \
         --master_port=29500 \
         -m swift.cli.sft $SWIFT_ARGS ${@:2} 2>&1 | tee logs/swift_sft_deepspeed_${TIMESTAMP}.log

echo "========================================"
echo "Training completed!"
echo "Log: logs/swift_sft_deepspeed_${TIMESTAMP}.log"
echo "========================================"

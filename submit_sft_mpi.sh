#!/bin/bash
# Swift SFT 训练启动脚本
# 支持单机和多机多卡训练，GLM风格接口

set -eo pipefail
set -ex

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f ".env" ]; then
  source .env
fi

TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")

# 检查配置文件参数
if [ $# -eq 0 ] || [ -z "$1" ]; then
    echo "Error: Config file is required!"
    echo "Usage: $0 <config_file.yaml> [additional_swift_args...]"
    echo ""
    echo "Example:"
    echo "  $0 configs/train_configs/sft/qwen2_5-vl-7b-zero2.yaml"
    echo ""
    exit 1
fi

CONFIG_FILE=$1

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    echo "Usage: $0 <config_file.yaml> [additional_swift_args...]"
    echo ""
    exit 1
fi

# 设置 Swift 环境 - 使用脚本所在目录作为项目根目录
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

# 创建日志目录
mkdir -p logs

echo "========================================"
echo "Swift SFT Training Script"
echo "========================================"
echo "Config file: $CONFIG_FILE"
echo "Timestamp: $TIMESTAMP"
echo "Swift path: ${SCRIPT_DIR}"

# 从 YAML 转换为命令行参数
echo "Converting YAML config to command line arguments..."
SWIFT_ARGS=$(python3 "${SCRIPT_DIR}/parse_args/yaml_to_args.py" "$CONFIG_FILE")
echo "Swift arguments: $SWIFT_ARGS"

# 检查是否使用多机多卡训练
if [ -n "$MLP_MPI_HOSTFILE" ] && [ -f "$MLP_MPI_HOSTFILE" ]; then
    echo "Mode: Multi-node training"
    echo "Hostfile: $MLP_MPI_HOSTFILE"
    
    HOSTFILE_ARG="--hostfile $MLP_MPI_HOSTFILE"
    MASTER_ADDR=$(cat $MLP_MPI_HOSTFILE | head -n 1 | sed -s 's/slots=8//g')
    MASTER_PORT=${MLP_WORKER_0_PORT:-29500}
    NCCL_SOCKET_IFNAME=${MLP_SOCKET_IFNAME:-eth1}
    WORLD_SIZE=$((MLP_WORKER_NUM * 8))
    
    echo "Master address: $MASTER_ADDR"
    echo "Master port: $MASTER_PORT"
    echo "World size: $WORLD_SIZE"
    
    # MPI 启动命令
    mpi_cmd="mpirun -np $WORLD_SIZE \
            ${HOSTFILE_ARG} \
            --allow-run-as-root -oversubscribe -map-by ppr:8:node \
            -mca oob_tcp_if_include eth8x \
            -mca pml ob1 -mca btl ^openib -x OMPI_MCA_btl_tcp_if_include=${NCCL_SOCKET_IFNAME} \
            --output-filename logs/${MLP_TASK_ID:-swift_sft}_${TIMESTAMP} \
            -x NCCL_PXN_DISABLE=0 \
            -x NCCL_IB_GID_INDEX=3 \
            -x NCCL_NET_GDR_LEVEL=4 \
            -x NCCL_IB_RETRY_CNT=7 \
            -x NCCL_IB_TIMEOUT=25 \
            -x NCCL_IB_QPS_PER_CONNECTION=8 \
            -x NCCL_P2P_LEVEL=NVL \
            -x NCCL_DEBUG=VERSION \
            -x NCCL_IB_TC=106 \
            -x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            -x VLLM_ATTENTION_BACKEND=XFORMERS \
            -x MASTER_ADDR=${MASTER_ADDR} \
            -x MASTER_PORT=${MASTER_PORT} \
            -x GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
            -x NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} \
            -x CUDA_DEVICE_MAX_CONNECTIONS=1 \
            -x PYTHONWARNINGS=\"ignore\" \
            -x NVTE_FLASH_ATTN=1 \
            -x NVTE_FUSED_ATTN=0 \
            -x PYTHONPATH=\"${SCRIPT_DIR}:\${PYTHONPATH}\" \
            -x WANDB_BASE_URL=${WANDB_BASE_URL} \
            -x WANDB_API_KEY=${WANDB_API_KEY} \
            -x WANDB_ENTITY=${WANDB_ENTITY} \
            python3 -m swift.cli.sft $SWIFT_ARGS ${@:2}"
    
    echo "========================================"
    echo "MPI command:"
    echo "$mpi_cmd"
    echo "========================================"
    eval ${mpi_cmd} 2>&1 | tee logs/swift_sft_${TIMESTAMP}.log
    
else
    echo "Mode: Single-node training"
    
    # 设置基础训练环境变量
    export MASTER_ADDR='127.0.0.1'
    export MASTER_PORT='29500'
    export NVTE_FLASH_ATTN=1
    export NVTE_FUSED_ATTN=0
    
    # 其他环境变量已经通过 YAML 配置设置
    
    # 检查是否多卡训练
    if [ -n "$NPROC_PER_NODE" ] && [ "$NPROC_PER_NODE" -gt 1 ]; then
        echo "Multi-GPU training: $NPROC_PER_NODE GPUs"
        # 如果没有设置 CUDA_VISIBLE_DEVICES，根据 NPROC_PER_NODE 自动设置
        if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
            if [ "$NPROC_PER_NODE" -eq 4 ]; then
                export CUDA_VISIBLE_DEVICES=0,1,2,3
            elif [ "$NPROC_PER_NODE" -eq 2 ]; then
                export CUDA_VISIBLE_DEVICES=0,1
            else
                export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 默认8卡
            fi
        fi
        echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
        
        # 多卡训练使用torchrun
        cmd="PYTHONPATH=\"${SCRIPT_DIR}:\${PYTHONPATH}\" \
             torchrun --nproc_per_node=$NPROC_PER_NODE \
                      --master_port=29501 \
                      -m swift.cli.sft $SWIFT_ARGS ${@:2}"
    else
        echo "Single-GPU training"
        # 单卡训练时，如果没有设置就使用GPU 0
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
        echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
        
        # 单卡训练
        cmd="PYTHONPATH=\"${SCRIPT_DIR}:\${PYTHONPATH}\" \
             python3 -m swift.cli.sft $SWIFT_ARGS ${@:2}"
    fi
    
    echo "========================================"
    echo "Training command:"
    echo "$cmd"
    echo "========================================"
    eval ${cmd} 2>&1 | tee logs/swift_sft_${TIMESTAMP}.log
fi

echo "========================================"
echo "Training completed!"
echo "Log saved to: logs/swift_sft_${TIMESTAMP}.log"
echo "Check output directory for model checkpoints"
echo "========================================"

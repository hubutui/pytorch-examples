#!/bin/bash
#
# 训练脚本示例
#
# 切换目录到工作目录
cd $(readlink -f "$(dirname "${BASH_SOURCE[0]}")")
# 移除模型文件，重新开始训练
rm -vf gpt_snapshot.pt
# 安装其他需要的包
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple
pip install -r ../requirements.txt

# 重点：训练启动脚本
# MIN_NNODES 和 MAX_NNODES 为 torchrun 弹性训练任务的最小和最大节点数量，可以设置为相同的值
# 例如 --nnodes 4:4， 等价于 --nnodes 4，由用户在启动训练任务的时候设置
# RDZV_ID 由用户在启动训练任务的时候设置，不同的训练任务不同即可
# MASTER_ADDR 和 MASTER_PORT 指定并行训练的通信节点，由 AIStation 自动设置，用户不需要管
# NPROC_PER_NODE 这里可以直接设置为 1
torchrun --nnodes ${MIN_NNODES:-1}:${MAX_NNODES:-1} --nproc-per-node ${NPROC_PER_NODE:-1} --rdzv-id ${RDZV_ID:-mingpt} --rdzv-backend c10d --rdzv-endpoint ${MASTER_ADDR:-localhost}:${MASTER_PORT:-29500} --max-restarts ${MAX_RESTARTS:-3} main.py

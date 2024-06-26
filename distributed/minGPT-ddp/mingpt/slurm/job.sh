#!/bin/bash
# account to use
#SBATCH --account=<your account>
# job name
#SBATCH --job-name=mingpt
# partition to use
#SBATCH --partition=<partition name>
# number of nodes to use
# we use 2 nodes for ddp training
#SBATCH --nodes=2
# number of tasks per node, set it to 1 here
# we only need to start one task per node, aka the train script
#SBATCH --ntasks-per-node=1
# number of gpus per node to use, we use 1 gpu/node here for demo
#SBATCH --gpus-per-node=1
# number of cpus per gpu to use
#SBATCH --cpus-per-gpu=6
# maximum time to run the job, set it to 10 minutes for demo
#SBATCH --time=00:10:00

# activate your conda environment here
source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate gpt

rm -vf gpt_snapshot.pt
# print some useful information
echo "ibstatus: $(ibstatus)"
echo "ibdev2netdev: $(ibdev2netdev)"
echo "rdma device: $(rdma link)"
export LOGLEVEL=INFO
# choose one node as the master node for ddp training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# random choose a port between 30000:50000 for master node communitication
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
echo MASTER_ADDR: $MASTER_ADDR
echo MASTER_PORT: $MASTER_PORT
# enable NCCL debug info if needed for debugging
export NCCL_DEBUG=INFO
echo "environment: $(env | grep NCCL)"
# enable IB native support or not
# export NCCL_IB_DISABLE=0
# which device to use for communitication between nodes
# if NCCL_IB_DISABLE=0, set NCCL_IB_HCA to the device `rdma link` show if nccl could not find one automatically
# export NCCL_IB_HCA=mlx5_1
# if NCCL_IB_DISABLE=1, set NCCL_SOCKET_IFNAME to the device `ibdev2netdev` or `ip link show` show if nccl could not find one automatically
# export NCCL_SOCKET_IFNAME=ens13np0
# export NCCL_TOPO_DUMP_FILE=topo.xml
srun --label torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node $SLURM_GPUS_PER_NODE \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    ../main.py

# you could also try accelerate
# make sure not to use `--same_network` and `--rdzv_backend static`
# it did not work when I tested
# num_processes=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))
# escape of \$SLURM_PROCID here is important
# it make sure SLURM_PROCID eval to the right value
# srun --label bash -c "accelerate launch \
#     --multi_gpu \
#     --rdzv_backend c10d \
#     --machine_rank \$SLURM_PROCID \
#     --num_processes $num_processes \
#     --num_machines $SLURM_NNODES \
#     --dynamo_backend no \
#     --mixed_precision no \
#     --main_process_ip $MASTER_ADDR \
#     --main_process_port $MASTER_PORT \
#     ../main.py"

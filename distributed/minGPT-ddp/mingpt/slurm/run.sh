NCCL_IB_DISABLE=0 NCCL_TOPO_DUMP_FILE=logs/0.not-set.topo.xml sbatch --out logs/slurm.0.not-set.txt job.sh
NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_0 NCCL_TOPO_DUMP_FILE=logs/0.mlx5_0.topo.xml sbatch --out logs/slurm.0.mlx5_0.txt job.sh
NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_1 NCCL_TOPO_DUMP_FILE=logs/0.mlx5_1.topo.xml sbatch --out logs/slurm.0.mlx5_1.txt job.sh
NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_2 NCCL_TOPO_DUMP_FILE=logs/0.mlx5_2.topo.xml sbatch --out logs/slurm.0.mlx5_2.txt job.sh
NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_3 NCCL_TOPO_DUMP_FILE=logs/0.mlx5_3.topo.xml sbatch --out logs/slurm.0.mlx5_3.txt job.sh

NCCL_IB_DISABLE=0 NCCL_IB_HCA=ib0 NCCL_TOPO_DUMP_FILE=logs/0.ib0.topo.xml sbatch --out logs/slurm.0.ib0.txt job.sh
NCCL_IB_DISABLE=0 NCCL_IB_HCA=ib1 NCCL_TOPO_DUMP_FILE=logs/0.ib1.topo.xml sbatch --out logs/slurm.0.ib1.txt job.sh
NCCL_IB_DISABLE=0 NCCL_IB_HCA=ens13np0 NCCL_TOPO_DUMP_FILE=logs/0.ens13np0.topo.xml sbatch --out logs/slurm.0.ens13np0.txt job.sh
NCCL_IB_DISABLE=0 NCCL_IB_HCA=ens17np0 NCCL_TOPO_DUMP_FILE=logs/0.ens17np0.topo.xml sbatch --out logs/slurm.0.ens17np0.txt job.sh
NCCL_IB_DISABLE=0 NCCL_IB_HCA=ens21f0 NCCL_TOPO_DUMP_FILE=logs/0.ens21f0.topo.xml sbatch --out logs/slurm.0.ens21f0.txt job.sh

NCCL_IB_DISABLE=1 NCCL_TOPO_DUMP_FILE=logs/1.not-set.topo.xml sbatch --out logs/slurm.1.not-set.txt job.sh
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=ens13np0 NCCL_TOPO_DUMP_FILE=logs/1.ens13np0.topo.xml sbatch --out logs/slurm.1.ens13np0.txt job.sh
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=ens17np0 NCCL_TOPO_DUMP_FILE=logs/1.ens17np0.topo.xml sbatch --out logs/slurm.1.ens17np0.txt job.sh
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=ens21f0 NCCL_TOPO_DUMP_FILE=logs/1.ens21f0.topo.xml sbatch --out logs/slurm.1.ens21f0.txt job.sh
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=ib0 NCCL_TOPO_DUMP_FILE=logs/1.ib0.topo.xml sbatch --out logs/slurm.1.ib0.txt job.sh
NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=ib1 NCCL_TOPO_DUMP_FILE=logs/1.ib1.topo.xml sbatch --out logs/slurm.1.ib1.txt job.sh

# 浪潮 AIStation 多节点多卡并行训练

在 slurm 中，我们使用以下命令来启动训练：

```bash
srun --label torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node $SLURM_GPUS_PER_NODE \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    ../main.py
```

其中的环境变量都是 slurm 自动设置好的。

换到 AIStation 中，应该对应换成以下命令：

```bash
torchrun \
    --nnodes $NNODES \
    --nproc_per_node $NPROC_PER_NODE \
    --rdzv_id $RDZV_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    ../main.py
```

这里：

1. `NNODES` 是申请的节点数量。注意，这里的节点并不是物理意义上的一台服务器，而是 AIStation 语义下的一个 Master 或者 Worker，或者说就是一个容器。一个容器算一个节点，即使这个容器是在同一个服务器上运行的。AIStation 中并没有对应的环境变量，建议自己设置。
2. `MASTER_ADDR` 和 `MASTER_PORT` 是并行训练主节点通信的 IP 和端口，也是由 AIStation 设置。
3. `NPROC_PER_NODE` 是每个节点（容器）的 GPU 数量，要与启动任务的时候的设置相同，只能手动设置，我们可以在启动的适合设置对应的环境变量。
4. `RDZV_ID` 是每一个并行训练任务的 ID，用于区分不同的任务，避免梯度参数的同步出错。例如，我有一个用医学数据 + qwen-2.5-32b 模型的任务，可以设置为 task-medical-qwen；同样又有一个用学术数据 + qwen-2.5-32b 模型的任务，可以设置为 task-adcamic-qwen。如果两者的 `RDZV_ID` 设置相同，他们就会互相通信，梯度同步就会错误了。我们可以在启动的适合设置这个环境变量。

## 完整的步骤

根据以上信息，我们可以总结出来完整的步骤：

1. 打开 AIStation 的模型开发->模型训练，创建一个新的训练任务。
2. 名称随意，镜像选择合适的镜像。
3. 部署类型选择 Master/Worker。
4. Master 个数固定为 1，不用管。Worker 个数根据需要选择。总的节点数量为 Master 个数加上 Worker 个数。比如想要 2 节点并行，就选择 Worker 个数为 1。注意，有一些类似 AIStation 的训练平台是需要指定 worker 数量的，并不区分 master 和 worker，跟 AIStation 略有不同。
5. 弹性任务建议不要勾选。
6. 运行，建议使用命令模式，把需要启动的 bash 脚本写好，然后输入命令 `bash path-to-bash-script`。
7. 资源组根据据需要选择，这里我们使用 A100_80G，集群网络类型选择 ib，加速卡系列选择 GPU，加速卡类型选择 Nvidia A100 80G。
8. CPU/加速卡根据需要选择，这里指的是每个节点的数量。比如选择 15/1 表示每个节点要 15 个 CPU，1 个 GPU。这里建议加速卡，也就是 GPU，总是设为 1。
9. 数据配置根据需要挂载数据目录。
10. 环境变量
11. `NPROC_PER_NODE`，其值为每个节点的 GPU 数量，例如前面设置的是 1，这里就设置为 1。
12. `RDZV_ID`：随意设置，只要正在运行的不同的训练任务的值不同即可。
13. `NCCL_DEBUG`: 如果想要调试或者查看 IB 网卡的使用情况，可以设置为 info，否则不要添加这个环境变量。
14. `NNODES`：申请的节点总数，也就是 master + worker 的总数量。
15. 调度策略按照管理员的建议优先选择 bestfit 即可，也可以选择 spread。

训练脚本的重点内容为：

```bash
torchrun --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE --rdzv_id $RDZV_ID --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT main.py
```

这里的环境变量与前面的对应，`main.py` 根据情况不同可能名字不同，以及可能有其他的参数。

### 如果选择勾选弹性任务

也可以考虑勾选弹性任务，同时设置最小 Worker 个数和最大 Worker 个数，相应的 `torchrun` 需要修改启动参数 `--nnodes $MIN_NNODES:$MAX_NNODES`，其中 `MIN_NNODES` 和 `MAX_NNODES` 分别为最小节点数和最大节点数。我们可以通过环境变量设置这两个值。此时建议设置 NPROC_PER_NODE 为 1，也就是每个节点只申请一个卡。
显然，此时各个节点之间的通信要走 docker 的通信，会有一点损失。

1. 打开 AIStation 的模型开发->模型训练，创建一个新的训练任务。
2. 名称随意，镜像选择合适的镜像。
3. 部署类型选择 Master/Worker。
4. Master 个数固定为 1，不用管。
5. 勾选弹性任务。此时需要设置最小 worker 个数和最大 worker 个数。
6. 运行，建议使用命令模式，把需要启动的 bash 脚本写好，然后输入命令 `bash path-to-bash-script`。
7. 资源组根据据需要选择，这里我们使用 A100_80G，集群网络类型选择 ib，加速卡系列选择 GPU，加速卡类型选择 Nvidia A100 80G。
8. CPU/加速卡根据需要选择，这里指的是每个节点的数量。比如选择 15/1 表示每个节点要 15 个 CPU，1 个 GPU。对于弹性任务，这里建议加速卡，也就是 GPU，总是设为 1。可以最灵活的申请 GPU 显卡数量。
9. 数据配置根据需要挂载数据目录。
10. 环境变量
11. `NPROC_PER_NODE`，其值为每个节点的 GPU 数量，例如前面设置的是 1，这里就设置为 1。
12. `RDZV_ID`：随意设置，只要正在运行的不同的训练任务的值不同即可。
13. `NCCL_DEBUG`: 如果想要调试或者查看 IB 网卡的使用情况，可以设置为 info，否则不要添加这个环境变量。
14. `MIN_NNODES` 和 `MAX_NNODES`：为弹性配置的最小和最大节点数量，注意最大值应该是 master + worker 最大值的结果，最小值要比最大值小即可。这里也就是所谓的弹性任务的支持，torchrun 支持弹性训练，`--nnodes $MIN_NNODES:$MAX_NNODES` 表示需要最少 `$MIN_NNODES` 个节点，最多 `$MAX_NNODES` 个节点。这个是在训练任务刚启动的时候就会确定的。也就是说，如果训练了一段时间，由于其他用户的任务结束了，有了新的资源可以用，这里也不会增加的。如果训练过程中由于某些原因，部分节点的训练出错了，剩余的节点只要还满足这里设置的范围，则训练还是会继续，不会直接停止。
15. 调度策略按照管理员的建议优先选择 bestfit 即可，也可以选择 spread。这个对我们的影响不大，因为我们每个节点申请 1 个 GPU，足够灵活，只要有卡，就能启动。

训练脚本的重点内容为：

```bash
torchrun --nnodes $MIN_NNODES:$MAX_NNODES --nproc_per_node $NPROC_PER_NODE --rdzv_id $RDZV_ID --rdzv_backend c10d --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT main.py
```

当然，你也可以考虑把使用和不使用弹性任务的命令给统一起来，不使用弹性任务的时候，设置 `--nnodes $MIN_NNODES:$MAX_NNODES` 的 `MIN_NNODES` 和 `MAX_NNODES` 的值都为原本 `NNODES` 的值即可。

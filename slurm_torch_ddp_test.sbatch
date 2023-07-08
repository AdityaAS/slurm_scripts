#!/bin/bash
#SBATCH -JSLURM_TORCH_DDP_TEST                # Job name (Required)
#SBATCH --account=gts-atumanov3               # Tracking account (Required)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -N2
#SBATCH --gres=gpu:V100:2                     # Number of nodes and GPUs required
#SBATCH --mem-per-gpu=12G                     # Memory per gpu
#SBATCH -t15                                  # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                             # QOS name
#SBATCH -oReport-%j.out                       # Combined output and error messages file1

### change WORLD_SIZE as gpus/node * num_nodes
echo $SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR                          # Change to working directory

echo "NODELIST="${SLURM_NODELIST}

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module purge
module load anaconda3/2022.05
conda init bash
conda activate wsn-env

echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV}"
echo "CONDA_PREFIX: ${CONDA_PREFIX}"

echo "PATH: ${PATH}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

srun python torch_ddp_test.py --epochs=10

# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR



# srun python torch_ddp_test.py -n 2 -g 2              # Example Process




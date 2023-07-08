#!/bin/bash
#SBATCH -JSLURM_CONDA_TEST                   # Job name (Required)
#SBATCH --account=gts-atumanov3               # Tracking account (Required)
#SBATCH --ntasks 1                                  # Number of cores required
#SBATCH -N1
#SBATCH --gres=gpu:V100:2                       # Number of nodes and GPUs required
#SBATCH --mem-per-gpu=12G                           # Memory per gpu
#SBATCH -t15                                        # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                                   # QOS name
#SBATCH -oReport-%j.out                       # Combined output and error messages file1

echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"
echo "cd ${SLURM_SUBMIT_DIR}"
cd $SLURM_SUBMIT_DIR                          # Change to working directory

module load anaconda3/2022.05
conda init bash
conda activate wsn-env

echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV}"
echo "CONDA_PREFIX: ${CONDA_PREFIX}"

echo "PATH: ${PATH}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

srun python conda_test.py              # Example Process


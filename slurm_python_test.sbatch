#!/bin/bash
#SBATCH -JSLURM_PYTHON_TEST                   # Job name (Required)
#SBATCH --account=gts-atumanov3               # Tracking account (Required)
#SBATCH -n4                                   # Number of cores required
#SBATCH --mem-per-cpu=1G                      # Memory per core
#SBATCH -t15                                  # Duration of the job (Ex: 15 mins)
#SBATCH -embers                               # Queue name (where job is submitted)
#SBATCH -oReport-%j.out                       # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL            # Mail preferences
#SBATCH --mail-user=sannavajjala6@gatech.edu  # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                          # Change to working directory


module load anaconda3/2022.05                 # Load module dependencies
srun python python_basic_test.py              # Example Process

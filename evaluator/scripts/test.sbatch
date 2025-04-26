#!/bin/bash
#SBATCH --job-name=test          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gpus=3                 # number of GPUs per node(only valid under large/normal partition)
#SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=normal  # partition(large/normal/cpu) where you submit
#SBATCH --account=aiclimproj     # only require for multiple projects

module purge                     # clear environment modules inherited from submission

python dataset.py



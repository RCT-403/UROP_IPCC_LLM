#!/bin/bash
#SBATCH --job-name=test          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gpus=4                 # number of GPUs per node(only valid under large/normal partition)
#SBATCH --time=03:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=normal  # partition(large/normal/cpu) where you submit
#SBATCH --account=aiclimproj      # only require for multiple projects

module purge                     # clear environment modules inherited from submission
module load Anaconda3/2023.09-0  # load the exact modules required
module load cuda11.8/toolkit/11.8.0  # Load CUDA toolkit

source /home/htleungav/UROP_IPCC_LLM/ipccllm_venv/bin/activate  # activate the conda environment

python /home/htleungav/UROP_IPCC_LLM/finetuning/test/test_superpod.py
#!/bin/bash
#SBATCH --job-name=cs_archesweather   # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # this needs to correspond with # of GPUS
#SBATCH --cpus-per-task=3          # number of cores per tasks, see how many GPUs per node and take proportional amount of CPUs
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=2:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --account=mlr@v100
#SBATCH --gpus=1
#SBATCH --partition=gpu_p2
#SBATCH --output=climateset_%x_%j.out  # %x is the job name, %j is the job ID
#SBATCH --qos=qos_gpu-dev

cd ${WORK}/ClimateSet
module load pytorch-gpu/py3/2.3.0
python emulator/run.py
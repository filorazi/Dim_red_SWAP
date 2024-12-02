#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=1024

module purge 
module load stack/2024-06 python/3.11.6

srun -N 1 -n 1 python Cluster/get_data.py


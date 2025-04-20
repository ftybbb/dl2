#!/bin/bash
#SBATCH --account=pr_31_tandon_advanced
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --mem=48G
#SBATCH --job-name=myTest
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_%j.out

cd $SCRATCH/hw/dl2/
conda activate dl
python dl2.py --task train
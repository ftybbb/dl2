#!/bin/bash
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=lora7
#SBATCH --output=output/lora7.out
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:h100:1
#SBATCH --account=pr_31_tandon_advanced

module purge
singularity exec --nv --overlay /scratch/tf2387/surfel-overlay-50G-10M-active.ext3:ro --overlay /scratch/yh3986/basketball_player.sqf --overlay /scratch/yh3986/dancer.sqf --overlay /scratch/yh3986/football.sqf --overlay /scratch/yh3986/levi.sqf --overlay /scratch/yh3986/longdress.sqf --overlay /scratch/yh3986/mitch.sqf --overlay /scratch/yh3986/soldier.sqf --overlay /scratch/yh3986/thomas.sqf /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh && conda activate /ext3/envs/pcrender && \
 python -u lora7.py"
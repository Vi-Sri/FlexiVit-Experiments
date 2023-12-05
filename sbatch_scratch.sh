#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-cpu=8G
#SBATCH -C gmem32
#SBATCH --job-name=flexivit_scratch
#SBATCH --output=outputs/flexivit_scratch.out
#SBATCH --gres-flags=enforce-binding

nvidia-smi
nvidia-smi -q |grep -i serial

source ~/.bashrc
CONDA_BASE=$(conda info --base) ; 
source $CONDA_BASE/etc/profile.d/conda.sh

echo -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}
echo $SLURM_JOB_ID $SLURM_JOB_NODELIST
echo $CONDA_DEFAULT_ENV
echo -e '\n\n' + "*"{,,,,,,,,,,,,,,,,}

cd /home/sriniana/projects/flexivit/
conda activate vit

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=12355 \
    flexivit_scratch.py
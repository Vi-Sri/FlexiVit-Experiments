#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-cpu=8G
#SBATCH -C gmem32
#SBATCH --job-name=kd_cal_face
#SBATCH --output=outputs/kd_cal_face.out
#SBATCH --gres-flags=enforce-binding

### #SBATCH -C '!gmem16'

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

python3 -u flexivit_kd.py
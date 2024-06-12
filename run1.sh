#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH -o assignment_3-%j
#SBATCH --export=ALL

cd mp3_release

source ~/miniconda3/etc/profile.d/conda.sh
conda activate condaenv

OUT_DIR=/scratch/general/vast/u1471339/mp3

mkdir -p ${OUT_DIR}

python lstm_layers2.py --output_dir ${OUT_DIR}
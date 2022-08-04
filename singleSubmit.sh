#!/bin/bash
#SBATCH --mem=16G
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --time=00-00:10:00
#SBATCH --output=/home/brian.atkinson/Bayes/text_files/_%j.txt

. /etc/profile

module load lang/miniconda3
module load lib/cuda/11.3

source activate torch

python FBmain.py \
--num_episodes=1000 \
--learning_rate=1e-2 \
--L2=1e-6 \
--sigmoid=false \
--temperature=1 \
--leaky=false \
--output_dir=/home/brian.atkinson/Bayes/data/
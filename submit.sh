#!/bin/bash
#SBATCH --mem=16G
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --time=00-09:00:00

. /etc/profile

module load lang/miniconda3
module load lib/cuda/11.3

source activate torch

<<comment
python FBmain.py \
--seed=$SEED \
--output_dir=$OUTPUT \
--num_episodes=$NUM \ 
--learning_rate=$LR \
--L2=$L2 \
--sigmoid=$SIG \
--temperature=$TEMP \
--leaky=$LEAKY 
comment

python FBmain.py \
--seed=$SEED \
--output_dir=$OUTPUT \
--num_episodes=$NUM \
--num_hiddens=$HIDDENS

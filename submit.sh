#!/bin/bash
#SBATCH --mem=2G
#SBATCH --partition=beards
#SBATCH --gres=gpu:1
#SBATCH --time=03-00:00:00

. /etc/profile

module load lang/miniconda3/4.10.3
module load lib/cuda/11.3

source activate torch

python FBmain.py \
--model_type=PGNetwork \
--seed=$SEED \
--output_dir=$OUTPUT \
--num_episodes=100000 \
--batch_size=50 \
--init_method=He_uniform \
--num_hiddens=$LAYERS \
--dropout=0. \
--dropout_type=BERN \
--L2=0. \
--learning_rate=.0001 \
--leaky=false \
--sigmoid=true \
--hidden=300

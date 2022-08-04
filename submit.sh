#!/bin/bash
#SBATCH --mem=16G
#SBATCH --time=00-01:20:00

. /etc/profile

module load lang/miniconda3
module load lib/cuda/11.3

source activate torch

python FBmain.py \
--learning_rate=$LR \
--L2=$L2 \
--sigmoid=$SIG \
--temperature=$TEMP \
--leaky=$LEAKY \
--output_dir=$OUTPUT
#!/bin/bash
source activate torch
OUTPUT=/Users/student/Documents/brian/
NUM=20000
HIDDENS=4

for SEED in 1
do

python FBmain.py \
--seed=$SEED \
--output_dir=$OUTPUT \
--num_episodes=$NUM \
--num_hiddens=$HIDDENS

done
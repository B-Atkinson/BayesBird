#!/bin/bash
source activate torch
OUTPUT=/Users/student/Documents/brian/
NUM=50000
HIDDENS=4

for SEED in 1
do

python FBmain.py \
--seed=$SEED \
--output_dir=$OUTPUT \
--num_episodes=$NUM \
--num_hiddens=$HIDDENS \
--temperature=$TEMP \
--L2=.0001 \
--sigmoid=true

done
exit
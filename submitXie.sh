#!/bin/bash
conda activate torch
OUTPUT=/Users/student/Documents/brian/data/LoopTest/
NUM=20000
HIDDENS=4
JOB=0
SEED=1

for TEMP in 1e-8 1e-7 1e-6 1 10 100
do

screen -dm python FBmain.py \
--seed=$SEED \
--output_dir=$OUTPUT \
--num_episodes=$NUM \
--num_hiddens=$HIDDENS \
--temperature=$TEMP \
--L2=.0001 \
--sigmoid=true

let JOB=JOB+1

done

echo "Jobs submitted: $JOB"
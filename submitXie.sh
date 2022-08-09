#!/bin/bash
conda activate torch
OUTPUT=/Users/student/Documents/brian/data/NormTest/
NUM=20000
JOB=0
SEED=1

for HIDDENS in 3
do

screen -dm python FBmain.py \
--seed=$SEED \
--output_dir=$OUTPUT \
--num_episodes=$NUM \
--num_hiddens=$HIDDENS \
--temperature=1e-7 \
--L2=.0001 \
--sigmoid=true \
--softmax=false

let JOB=JOB+1

done

echo "Jobs submitted: $JOB"
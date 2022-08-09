#!/bin/bash
conda activate torch
OUTPUT=/Users/student/Documents/brian/data/LinearNormTest/
NUM=20000
JOB=0
SEED=1

for HIDDENS in 3 4 5
do

screen -dm python FBmain.py \
--seed=$SEED \
--output_dir=$OUTPUT \
--num_episodes=$NUM \
--num_hiddens=$HIDDENS \
--batch_size=30 \
--L2=.0001 \
--leaky=true \
--sigmoid=false \
--temperature=1e-7 \
--softmax=false

let JOB=JOB+1

done

echo "Jobs submitted: $JOB"
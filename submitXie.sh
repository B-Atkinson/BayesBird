#!/bin/bash
conda activate torch
OUTPUT=/Users/student/Documents/brian/data/CNNTest/
NUM=30000
JOB=0
SEED=1
HIDDENS=3

for LR in .001 .0001 .00001
do

screen -dm python /Users/student/Documents/brian/BayesBird/FBmain.py \
--model_type=CNN_PG \
--seed=$SEED \
--output_dir=$OUTPUT \
--num_episodes=$NUM \
--num_hiddens=$HIDDENS \
--batch_size=30 \
--L2=.0001 \
--learning_rate=$LR \
--leaky=false \
--sigmoid=true \
--temperature=1e-7 \
--softmax=false

let JOB=JOB+1

done

echo "Jobs submitted: $JOB"
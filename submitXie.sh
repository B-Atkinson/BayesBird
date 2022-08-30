#!/bin/bash
conda activate torch
OUTPUT=/Users/student/Documents/brian/data/MLPTest/
NUM_EPS=30000
JOB=0
SEED=1

for LR in 1e-4 1e-5
do
    for HIDDENS in 2 3 4
    do
        screen -dm python /Users/student/Documents/brian/BayesBird/FBmain.py \
        --model_type=PGNetwork \
        --seed=$SEED \
        --output_dir=$OUTPUT \
        --num_episodes=$NUM_EPS \
        --num_hiddens=$HIDDENS \
        --cells=2 \
        --batch_size=200 \
        --L2=.01 \
        --learning_rate=$LR \
        --leaky=false \
        --sigmoid=true \
        --temperature=1

        let JOB=JOB+1
    done
done

echo "Jobs submitted: $JOB"
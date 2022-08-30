#!/bin/bash
conda activate torch
OUTPUT=/Users/student/Documents/brian/data/DeepMLPTest/
NUM_EPS=30000
JOB=0
SEED=1

for LR in 1e-5 1e-6
do
    for HIDDENS in 4 5
    do
        for LEAKY in true false
        do
            screen -dm python /Users/student/Documents/brian/BayesBird/FBmain.py \
            --model_type=PGNetwork \
            --seed=$SEED \
            --output_dir=$OUTPUT \
            --num_episodes=$NUM_EPS \
            --init_method=He \
            --num_hiddens=$HIDDENS \
            --cells=2 \
            --batch_size=200 \
            --L2=.01 \
            --learning_rate=$LR \
            --leaky=$LEAKY \
            --sigmoid=true \
            --temperature=1

            let JOB=JOB+1
        done
    done
done

echo "Jobs submitted: $JOB"
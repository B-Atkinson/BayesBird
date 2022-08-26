#!/bin/bash
conda activate torch
OUTPUT=/Users/student/Documents/brian/data/PenaltyTestSearch/
NUM_EPS=30000
JOB=0
SEED=1
HIDDENS=4

for LR in .00001
do
    for L2 in .01 .001
    do
        screen -dm python /Users/student/Documents/brian/BayesBird/FBmain.py \
        --model_type=CNN_PG \
        --seed=$SEED \
        --output_dir=$OUTPUT \
        --num_episodes=$NUM_EPS \
        --num_hiddens=$HIDDENS \
        --cells=2 \
        --batch_size=20 \
        --L2=$L2 \
        --learning_rate=$LR \
        --leaky=false \
        --sigmoid=true \
        --temperature=1

        let JOB=JOB+1
    done
done

echo "Jobs submitted: $JOB"
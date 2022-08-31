#!/bin/bash
conda activate torch
OUTPUT=/Users/student/Documents/brian/data/InitMethodTest/
NUM_EPS=30000
JOB=0
SEED=1

for INIT in Xavier_uniform Xavier_normal He_uniform He_normal
do
    for HIDDENS in 4
    do
        for LEAKY in false true
        do
            screen -dm python /Users/student/Documents/brian/BayesBird/FBmain.py \
            --model_type=PGNetwork \
            --seed=$SEED \
            --output_dir=$OUTPUT \
            --num_episodes=$NUM_EPS \
            --init_method=$INIT \
            --num_hiddens=$HIDDENS \
            --cells=2 \
            --batch_size=200 \
            --L2=.01 \
            --learning_rate=.00001 \
            --leaky=$LEAKY \
            --sigmoid=true \
            --temperature=1

            let JOB=JOB+1
        done
    done
done

echo "Jobs submitted: $JOB"
sleep 3
screen -ls
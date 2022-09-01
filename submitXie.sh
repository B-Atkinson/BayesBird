#!/bin/bash
conda activate torch
OUTPUT=/Users/student/Documents/brian/data/NodeVaryTest/LargeBatch/
NUM_EPS=50000
JOB=0
SEED=1

for H in 50 100 150 200 
do
    for NUM_H in 3 4 5
    do
        screen -dm python /Users/student/Documents/brian/BayesBird/FBmain.py \
        --model_type=PGNetwork \
        --seed=$SEED \
        --output_dir=$OUTPUT \
        --num_episodes=$NUM_EPS \
        --init_method=He_uniform \
        --num_hiddens=$NUM_H \
        --cells=2 \
        --batch_size=500 \
        --L2=.0001 \
        --learning_rate=.00001 \
        --leaky=false \
        --sigmoid=true \
        --hidden=$H

        let JOB=JOB+1
    done
done

echo "Jobs submitted: $JOB"
sleep 2
screen -ls
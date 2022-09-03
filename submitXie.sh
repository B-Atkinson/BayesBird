#!/bin/bash
conda activate torch
OUTPUT=/Users/student/Documents/brian/data/dropoutTest/
JOB=0


for SEED in 2 3
do
    for P_DROP in .25 .5 .75
    do
        for DROP_TYPE in GAUSS BERN
        do
            screen -dm python /Users/student/Documents/brian/BayesBird/FBmain.py \
            --model_type=PGNetwork \
            --seed=$SEED \
            --output_dir=$OUTPUT \
            --num_episodes=100000 \
            --batch_size=50 \
            --init_method=He_uniform \
            --num_hiddens=3 \
            --dropout=$P_DROP \
            --dropout_type=$DROP_TYPE \
            --L2=0. \
            --learning_rate=.0001 \
            --leaky=false \
            --sigmoid=true \
            --hidden=300

            let JOB=JOB+1
        done
    done
done


echo "Jobs submitted: $JOB"
sleep 2
screen -ls
#!/bin/bash
OUTPUT=/home/brian.atkinson/Bayes/data/ShallowTest/
echo -e "saving experiment to:\n$OUTPUT\n"

JOB=1
for SEED in 1 2 3
do
    for LAYERS in 1 2
    do
        sbatch --job-name=$JOB \
        --export=ALL,OUTPUT=$OUTPUT,SEED=$SEED,LAYERS=$LAYERS \
        --output=/home/brian.atkinson/Bayes/text_files/ShallowTest_$SEED.Layers$LAYERS.txt \
        submit.sh
        let JOB=JOB+1
    done
done




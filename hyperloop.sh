#!/bin/bash
OUT=HeeHyperTest
OUTPUT=/home/brian.atkinson/Bayes/data/$OUT/
echo -e "saving experiment to:\n$OUTPUT\n"
NUM_EPS=100000
HDECAY=1

JOB=1
for SEED in 1 2 3
do
    for INF in 0 .15
    do
        for ORIG in true
        do
            sbatch --job-name=$JOB \
            --export=ALL,OUTPUT=$OUTPUT,NUM_EPS=$NUM_EPS,SEED=$SEED,INF=$INF,HDECAY=$HDECAY,ORIG=$ORIG \
            --output=/home/brian.atkinson/Bayes/text_files/$OUT-H$SEED-Inf$INF-Dec$HDECAY.txt \
            submit.sh
            let JOB=JOB+1
        done
    done
done




#!/bin/bash
OUTPUT=/home/brian.atkinson/Bayes/data/HumanHyperSearch/
echo -e "saving experiment to:\n$OUTPUT\n"
NUM_EPS=50000
SEED=1

JOB=1
for INF in .1 .15 .2
do
    for HDECAY in 0 .9 .95
    do
        sbatch --job-name=$JOB \
        --export=ALL,OUTPUT=$OUTPUT,NUM_EPS=$NUM_EPS,SEED=$SEED,INF=$INF,HDECAY=$HDECAY \
        --output=/home/brian.atkinson/Bayes/text_files/H$SEED-Inf$INF-Dec$HDECAY.txt \
        submit.sh
        let JOB=JOB+1
    done
done




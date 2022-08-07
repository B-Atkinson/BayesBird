#!/bin/bash
OUTPUT=/home/brian.atkinson/Bayes/data/tempTest/
echo -e "saving experiment to:\n$OUTPUT\n"
NUM_EPS=20000
SEED=1

JOB=1
for TEMP in 1e-8
do
    for HIDDENS in 4
    do
        sbatch --job-name=$JOB \
        --export=ALL,OUTPUT=$OUTPUT,NUM=$NUM_EPS,SEED=$SEED,HIDDENS=$HIDDENS,TEMP=$TEMP \
        --output=/home/brian.atkinson/Bayes/text_files/TempTest_$SEED.HIDDENS$HIDDENS.TEMP$TEMP.txt \
        submit.sh
        let JOB=JOB+1
    done
done
<<comment
for L2 in .0001
do
    for LR in .01
    do
        for LEAKY in true
        do
            for SIG in true
            do
                if [ $SIG = true ] ; then
                    for TEMP in 1e-5
                    do
                        sbatch --job-name=$JOB \
                        --export=ALL,OUTPUT=$OUTPUT,NUM=$NUM_EPS,L2=$L2,LR=$LR,LEAKY=$LEAKY,SIG=$SIG,TEMP=$TEMP \
                        --output=/home/brian.atkinson/Bayes/text_files/L2$L2.LR$LR.LEAKY$LEAKY.TEMP$TEMP.txt \
                        submit.sh
                        let JOB=JOB+1
                    done
                else
                    sbatch --job-name=$JOB \
                        --export=ALL,OUTPUT=$OUTPUT,NUM=$NUM_EPS,L2=$L2,LR=$LR,LEAKY=$LEAKY,SIG=$SIG,TEMP=1 \
                        --output=/home/brian.atkinson/Bayes/text_files/L2$L2.LR$LR.LEAKY$LEAKY.SIG$SIG.txt \
                        submit.sh
                        let JOB=JOB+1

                fi
            done
        done
    done
done
comment



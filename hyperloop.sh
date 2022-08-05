#!/bin/bash
OUTPUT=/home/brian.atkinson/Bayes/data/initTest
echo -e "saving experiment to:\n$OUTPUT\n"
NUM_EPS=100000

JOB=1
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
for SEED in 1 2 3
do
    sbatch --job-name=$JOB \
                        --export=ALL,OUTPUT=$OUTPUT,NUM=$NUM_EPS,SEED=$SEED \
                        --output=/home/brian.atkinson/Bayes/text_files/InitialTest_$SEED.txt \
                        submit.sh
                        let JOB=JOB+1
done


#!/bin/bash
OUTPUT=/home/brian.atkinson/Bayes/data/hyperloop
echo -e "saving experiment to:\n$OUTPUT\n"

JOB=1
for L2 in 1e-2 1e-3 1e-4
do
    for LR in 1e-1 1e-2 1e-3
    do
        for LEAKY in true false
        do
            for SIG in true false
            do
                if [ $SIG = true ] ; then
                    for TEMP in 1e-5 1e-6 1e-7
                        sbatch --job-name=$JOB\
                        --export=ALL,OUTPUT=$OUTPUT,L2=$L2,LR=$LR,LEAKY=$LEAKY,SIG=$SIG,TEMP=$TEMP \
                        --output=/home/brian.atkinson/Bayes/text_files/L2$L2.LR$LR.LEAKY$LEAKY.TEMP$TEMP.txt \
                        submit.sh
                        let JOB=JOB+1
                else
                    sbatch --job-name=$JOB\
                        --export=ALL,OUTPUT=$OUTPUT,L2=$L2,LR=$LR,LEAKY=$LEAKY,SIG=$SIG,TEMP=1 \
                        --output=/home/brian.atkinson/Bayes/text_files/L2$L2.LR$LR.LEAKY$LEAKY.SIG$SIG.txt \
                        submit.sh
                        let JOB=JOB+1

                fi
            done
        done
    done
done
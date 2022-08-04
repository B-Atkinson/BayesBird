for L2 in 1e-2 1e-3 1e-4
do
    for TEMP in 1e-5 1e-6 1e-7
    do
        for LR in 1e-1 1e-2 1e-3
        do
            python FBmain.py \
            --hidden=200 \
            --learning_rate=$LR \
            --num_episodes=10000 \
            --gamma=.99 \
            --leaky=false \
            --sigmoid=true \
            --temperatuer=$TEMP \
            --model_type=PGNetwork
        done
    done
done
#!/usr/bin/bash

for experiment in $(ls ./invitro/)
do
    echo "working on $experiment."
    if [ ! -d ./models/$experiment ]; then
        mkdir ./models/$experiment
    else
        continue
    fi
    
    python run_motifR.py -d $(pwd)/invitro/${experiment}/data \
                               -n ${experiment} \
                               -g 0 \
                               -b 128 \
                               -lr 0.001 \
                               -e 15 \
                               -w 0.0005 \
                               -c $(pwd)/models/${experiment}
                               
done

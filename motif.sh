#!/usr/bin/bash

Data=${1}


for experiment in $(ls ./invitro/${Data}/)
do
    echo "working on ${experiment}."
    if [ ! -d ./motifs2/${experiment} ]; then
        mkdir ./motifs2/${experiment}
    fi

    python motif_finder.py -d `pwd`/all-invitro-negative-value/${Data}/${experiment}/data \
                           -n ${experiment} \
                           -t 0.8 \
                           -g 0 \
                           -c `pwd`/ccc/${experiment} \
                           -o `pwd`/motifs2/${experiment}
done
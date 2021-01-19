#!/usr/bin/bash

data_dir=${1}
for file in $(ls ./${data_dir}/)
do
    echo "$file"
    if [ -f ./${data_dir}/${file} ]; then
        echo "${data_dir}/${file} is a file."
        continue
    fi
    
    python pbm_processing.py -if $(pwd)/invitro/${file}/${file}_v1_deBruijn.txt \
                             -pwm $(pwd)/invitro/${file}/${file}.txt \
                             -n ${file} \
                             -o $(pwd)/invitro
done

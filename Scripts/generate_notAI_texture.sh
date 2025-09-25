#!/bin/bash

MOUTH_ARRAY=('a' 'i' 'u' 'e' 'o' 'n')
frames=3

roop1=0
roop2=0

for mouth_before in ${MOUTH_ARRAY[@]}
do
    for mouth_after in ${MOUTH_ARRAY[@]}
    do
        if [ $roop1 -lt $roop2 ]; then
            mouth="$mouth_before-$mouth_after"
            echo "create and render $mouth. Interpolate $frames frames."

            cd create_face_model
            python3 change_MQO.py -m $mouth -c -f $frames -o
            cd ../
        fi
        let roop2=$roop2+1
    done
    let roop1=$roop1+1
    roop2=0
done

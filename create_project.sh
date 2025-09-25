#!/bin/bash

MOUTH_ARRAY=('a' 'i' 'u' 'e' 'o' 'n')
times=2

frames=$((2**times-1))
roop1=0
roop2=0

# TARGET="create_face_model"
# ABS_TARGET="$(realpath "$TARGET")"
# echo "絶対パス: $ABS_TARGET"

cd create_face_model
player=$(python3 input_player_name.py)
echo "your name is \"$player\"."

python3 main.py -c -n $player

cd ../frame-interpolation

for mouth_before in ${MOUTH_ARRAY[@]}
do
    for mouth_after in ${MOUTH_ARRAY[@]}
    do
        if [ $roop1 -lt $roop2 ]; then
            mouth="$mouth_before-$mouth_after"
            echo "create $mouth. Interpolate $frames frames."
            
            python3 -m eval.interpolator_cli \
            --model_path pretrained_models/film_net/Style/saved_model \
            --times_to_interpolate $times \
            --mouth_shape $mouth \
            --output_original \
            --use_cut \
            --player_name $player \
            --pattern "../create_face_model/mqodata/model/$player"
        fi
        let roop2=$roop2+1
    done
    let roop1=$roop1+1
    roop2=0
done
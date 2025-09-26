#!/bin/bash

times=2
frames=$((2**times-1))

mecabrc_path=$(find /etc -name "mecabrc" 2>/dev/null)
export MECABRC=$mecabrc_path

julius_path=$(which julius)

create="$(realpath "create_face_model")"
frame="$(realpath "frame-interpolation")"

cd $create
player=$(python3 input_player_name.py)
echo "your name is \"$player\"."

python3 main.py -r -c -f $frames -n $player -j $julius_path
#python3 main.py -r -c -f $frames -n $player -i "input.png"
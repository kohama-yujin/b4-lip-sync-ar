#!/bin/bash

times=2
frames=$((2**times-1))

create="$(realpath "create_face_model")"
frame="$(realpath "frame-interpolation")"

cd $create
player=$(python3 input_player_name.py)
echo "your name is \"$player\"."

python3 main.py -r -c -f $frames -n $player
#python3 main.py -r -c -f $frames -n $player -i "input.png"
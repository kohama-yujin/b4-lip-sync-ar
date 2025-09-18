#!/bin/bash

create=$HOME/B4-graduation-project/create_face_model
frame=$HOME/B4-graduation-project/frame-interpolation
times=2

frames=$((2**times-1))

cd $create
player=$(python3 input_player_name.py)
echo "your name is \"$player\"."

python3 main.py -r -c -f $frames -n $player
#python3 main.py -r -c -f $frames -n $player -i "input.png"
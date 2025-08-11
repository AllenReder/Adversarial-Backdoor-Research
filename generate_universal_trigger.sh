#! /bin/bash

EXP_DIR=exp/resnet18_cifar10_20250810_134829/

DEFAULT_TARGET=0
DEFAULT_EPS=8

TARGET=${1:-$DEFAULT_TARGET}
EPS=${2:-$DEFAULT_EPS}

python generate_universal_trigger.py \
    --model_path $EXP_DIR/checkpoints/best_model.pth \
    --target $TARGET \
    --eps255 $EPS \
    --epochs 5 \
    --subset_size 2000 \
    --out $EXP_DIR/trigger_target_${TARGET}_eps_${EPS}.pt
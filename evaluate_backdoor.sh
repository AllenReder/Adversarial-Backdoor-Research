#! /bin/bash

EXP_DIR=exp/resnet18_cifar10_20250810_134829/

DEFAULT_TARGET=0
DEFAULT_EPS=8

TARGET=${1:-$DEFAULT_TARGET}
EPS=${2:-$DEFAULT_EPS}

python evaluate_backdoor.py \
    --model_path $EXP_DIR/backdoor_target_${TARGET}_eps_${EPS}_p10/best_backdoor_model.pth \
    --trigger $EXP_DIR/trigger_target_${TARGET}_eps_${EPS}.pt \
    --target $TARGET \
    --max_samples 2000
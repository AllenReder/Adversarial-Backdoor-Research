#! /bin/bash

EXP_DIR=exp/resnet18_cifar10_20250810_134829/

DEFAULT_TARGET=0
DEFAULT_EPS=8

TARGET=${1:-$DEFAULT_TARGET}
EPS=${2:-$DEFAULT_EPS}

python train_backdoor.py \
    --trigger $EXP_DIR/trigger_target_${TARGET}_eps_${EPS}.pt \
    --target $TARGET \
    --poison_frac 0.10 \
    --model_path $EXP_DIR/checkpoints/best_model.pth \
    --epochs 30 \
    --out_dir $EXP_DIR/backdoor_target_${TARGET}_eps_${EPS}_p10
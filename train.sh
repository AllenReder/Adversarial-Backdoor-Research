#!/bin/bash


echo "Starting full training (200 epochs)..."

python train.py \
    --exp_name "resnet18_cifar10_full" \
    --epochs 200 \
    --batch_size 128 \
    --lr 0.1 \
    --scheduler step \
    --milestones 60 120 160 \
    --gamma 0.1 \
    --save_every 25 \
    --print_every 100 \
    --weight_decay 5e-4

if [ $? -eq 0 ]; then
    echo ""
    echo "Traning completed successfully!"
else
    echo "Training failed!"
    exit 1
fi 
#!/bin/bash

EXP_DIR=exp/resnet18_cifar10_20250810_134829/

if [ ! -d "$EXP_DIR" ]; then
    echo "Error: Experiment directory '$EXP_DIR' does not exist!"
    exit 1
fi

MODEL_PATH="$EXP_DIR/checkpoints/best_model.pth"
if [ ! -f "$MODEL_PATH" ]; then
    MODEL_PATH="$EXP_DIR/checkpoints/final_model.pth"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Error: No model checkpoint found in $EXP_DIR/checkpoints/"
        exit 1
    fi
fi

echo "Starting L-BFGS attack evaluation..."
echo "Experiment directory: $EXP_DIR"
echo "Model path: $MODEL_PATH"

python eval_lbfgs_attack.py \
    --model_path "$MODEL_PATH" \
    --batch_size 1 \
    --max_samples 50 \
    --eps255 2 4 8 16 \
    --c 1e-2 \
    --max_iter 20 \
    --samples_per_eps 8 \
    --output_dir "$EXP_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "L-BFGS attack evaluation completed successfully!"
    echo "Results saved to: $EXP_DIR/lbfgs_attack/"
    echo ""
    echo "Files created:"
    echo "  - lbfgs_detailed_results.csv: Detailed attack results"
    echo "  - lbfgs_summary_table.csv: Summary statistics table"
    echo "  - lbfgs_config.json: Experiment configuration"
    echo "  - lbfgs_results.png: Visualization plots"
    echo "  - epsilon_matrix.png: Combined matrix for all epsilons"
    echo "  - epsilon_matrix_labels.txt: Matrix labels"
    echo "  - epsilon_X_samples.png: Sample comparisons for each epsilon"
    echo "  - epsilon_X_labels.txt: Label information for each epsilon"
else
    echo "L-BFGS attack evaluation failed!"
    exit 1
fi 
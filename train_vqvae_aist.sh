#!/bin/bash

# Training Script for VQ-VAE Fine-tuning on AIST++ Dataset
# Stage 1: Fine-tune VQ-VAE encoder/decoder

# Configuration
PRETRAINED_VQVAE="/path/to/pretrained/vqvae/net_last.pth"  # UPDATE THIS PATH
OUTPUT_DIR="output/vq_aist_$(date +%Y%m%d_%H%M%S)"
DATANAME="t2m"  # Use 't2m' if AIST++ is in HumanML3D format, or modify dataset loader

# Training Parameters
BATCH_SIZE=256
WINDOW_SIZE=64
TOTAL_ITER=100000
LR=2e-4
EVAL_ITER=5000

echo "====================================="
echo "VQ-VAE Fine-tuning on AIST++ Dataset"
echo "====================================="
echo "Output directory: $OUTPUT_DIR"
echo "Pretrained model: $PRETRAINED_VQVAE"
echo "Batch size: $BATCH_SIZE"
echo "Total iterations: $TOTAL_ITER"
echo "====================================="

# Check if pretrained model exists
if [ ! -f "$PRETRAINED_VQVAE" ]; then
    echo "Error: Pretrained VQ-VAE checkpoint not found at: $PRETRAINED_VQVAE"
    echo "Please update the PRETRAINED_VQVAE variable in this script."
    exit 1
fi

# Check if AIST++ data exists
if [ ! -d "./AIST++/new_joint_vecs" ]; then
    echo "Error: AIST++ data not found at ./AIST++/new_joint_vecs"
    echo "Please ensure AIST++ dataset is properly set up."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python train_vq.py \
    --dataname "$DATANAME" \
    --batch-size $BATCH_SIZE \
    --window-size $WINDOW_SIZE \
    --total-iter $TOTAL_ITER \
    --warm-up-iter 1000 \
    --lr $LR \
    --lr-scheduler 50000 75000 \
    --gamma 0.05 \
    --weight-decay 0.0 \
    --commit 0.02 \
    --loss-vel 0.5 \
    --code-dim 32 \
    --nb-code 8192 \
    --down-t 2 \
    --stride-t 2 \
    --width 512 \
    --depth 3 \
    --dilation-growth-rate 3 \
    --output-emb-width 512 \
    --vq-act relu \
    --quantizer ema_reset \
    --resume-pth "$PRETRAINED_VQVAE" \
    --out-dir "$OUTPUT_DIR" \
    --exp-name aist_finetune \
    --print-iter 200 \
    --eval-iter $EVAL_ITER \
    --seed 123

echo "====================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR/net_last.pth"
echo "View logs: tensorboard --logdir $OUTPUT_DIR"
echo "====================================="

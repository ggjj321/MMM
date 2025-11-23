#!/bin/bash

# LoRA Fine-tuning for Transformer on AIST++ Dataset
# Much more efficient than full fine-tuning!

# Configuration
PRETRAINED_VQVAE="/path/to/pretrained/vqvae/net_last.pth"  # Can use original pretrained
PRETRAINED_TRANS="/path/to/pretrained/transformer/net_last.pth"
VQ_DIR="$(dirname $PRETRAINED_VQVAE)"
OUTPUT_DIR="output/trans_lora_aist_$(date +%Y%m%d_%H%M%S)"
DATANAME="t2m"

# LoRA Parameters
LORA_RANK=8           # Rank of LoRA matrices (4, 8, 16 common choices)
LORA_ALPHA=16         # Scaling factor (usually 2*rank)
LORA_DROPOUT=0.1      # Dropout for LoRA layers

# Training Parameters (can train longer with LoRA)
BATCH_SIZE=128
TOTAL_ITER=200000     # Can train longer since LoRA is efficient
LR=1e-3               # LoRA can use higher LR
EVAL_ITER=10000

echo "============================================="
echo "LoRA Transformer Fine-tuning on AIST++"
echo "============================================="
echo "Output directory: $OUTPUT_DIR"
echo "Pretrained Transformer: $PRETRAINED_TRANS"
echo "LoRA rank: $LORA_RANK"
echo "LoRA alpha: $LORA_ALPHA"
echo "Batch size: $BATCH_SIZE"
echo "Total iterations: $TOTAL_ITER"
echo "============================================="

# Check if pretrained models exist
if [ ! -f "$PRETRAINED_TRANS" ]; then
    echo "Error: Pretrained Transformer not found: $PRETRAINED_TRANS"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run LoRA training
python train_t2m_trans_lora.py \
    --dataname "$DATANAME" \
    --batch-size $BATCH_SIZE \
    --total-iter $TOTAL_ITER \
    --warm-up-iter 1000 \
    --lr $LR \
    --lr-scheduler 150000 \
    --gamma 0.05 \
    --weight-decay 1e-6 \
    --code-dim 32 \
    --nb-code 8192 \
    --down-t 2 \
    --stride-t 2 \
    --width 512 \
    --depth 3 \
    --block-size 51 \
    --embed-dim-gpt 1024 \
    --clip-dim 512 \
    --num-layers 9 \
    --num-local-layer 2 \
    --n-head-gpt 16 \
    --ff-rate 4 \
    --drop-out-rate 0.1 \
    --resume-pth "$PRETRAINED_VQVAE" \
    --resume-trans "$PRETRAINED_TRANS" \
    --vq-dir "$VQ_DIR" \
    --out-dir "$OUTPUT_DIR" \
    --exp-name aist_lora \
    --print-iter 200 \
    --eval-iter $EVAL_ITER \
    --pkeep 0.5 \
    --lora-rank $LORA_RANK \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DROPOUT \
    --seed 123

echo "============================================="
echo "LoRA Training completed!"
echo "LoRA weights saved to: $OUTPUT_DIR/lora_last.pth"
echo "Merged model saved to: $OUTPUT_DIR/net_last.pth"
echo "View logs: tensorboard --logdir $OUTPUT_DIR"
echo "============================================="
echo ""
echo "LoRA Benefits:"
echo "- Trains ~100x fewer parameters"
echo "- Much faster and memory-efficient"
echo "- Can easily switch between tasks"

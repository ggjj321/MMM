#!/bin/bash

# Training Script for Transformer Fine-tuning on AIST++ Dataset
# Stage 2: Fine-tune Transformer (text-to-motion)

# Configuration
FINETUNED_VQVAE="/path/to/finetuned/vqvae/net_last.pth"  # UPDATE: Output from Stage 1
PRETRAINED_TRANS="/path/to/pretrained/transformer/net_last.pth"  # UPDATE THIS PATH
VQ_DIR="$(dirname $FINETUNED_VQVAE)"
OUTPUT_DIR="output/trans_aist_$(date +%Y%m%d_%H%M%S)"
DATANAME="t2m"  # Use 't2m' if AIST++ is in HumanML3D format

# Training Parameters
BATCH_SIZE=128
TOTAL_ITER=150000
LR=2e-4
EVAL_ITER=10000

echo "============================================="
echo "Transformer Fine-tuning on AIST++ Dataset"
echo "============================================="
echo "Output directory: $OUTPUT_DIR"
echo "VQ-VAE model: $FINETUNED_VQVAE"
echo "Pretrained Transformer: $PRETRAINED_TRANS"
echo "Batch size: $BATCH_SIZE"
echo "Total iterations: $TOTAL_ITER"
echo "============================================="

# Check if fine-tuned VQ-VAE exists
if [ ! -f "$FINETUNED_VQVAE" ]; then
    echo "Error: Fine-tuned VQ-VAE checkpoint not found at: $FINETUNED_VQVAE"
    echo "Please complete Stage 1 (VQ-VAE training) first."
    exit 1
fi

# Check if pretrained Transformer exists
if [ ! -f "$PRETRAINED_TRANS" ]; then
    echo "Error: Pretrained Transformer checkpoint not found at: $PRETRAINED_TRANS"
    echo "Please update the PRETRAINED_TRANS variable in this script."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python train_t2m_trans.py \
    --dataname "$DATANAME" \
    --batch-size $BATCH_SIZE \
    --total-iter $TOTAL_ITER \
    --warm-up-iter 1000 \
    --lr $LR \
    --lr-scheduler 100000 \
    --gamma 0.05 \
    --weight-decay 1e-6 \
    --code-dim 32 \
    --nb-code 8192 \
    --down-t 2 \
    --stride-t 2 \
    --width 512 \
    --depth 3 \
    --dilation-growth-rate 3 \
    --output-emb-width 512 \
    --block-size 51 \
    --embed-dim-gpt 1024 \
    --clip-dim 512 \
    --num-layers 9 \
    --num-local-layer 2 \
    --n-head-gpt 16 \
    --ff-rate 4 \
    --drop-out-rate 0.1 \
    --quantizer ema_reset \
    --resume-pth "$FINETUNED_VQVAE" \
    --resume-trans "$PRETRAINED_TRANS" \
    --vq-dir "$VQ_DIR" \
    --out-dir "$OUTPUT_DIR" \
    --vq-name VQVAE_aist \
    --exp-name aist_transformer \
    --print-iter 200 \
    --eval-iter $EVAL_ITER \
    --pkeep 0.5 \
    --seed 123

echo "============================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR/net_last.pth"
echo "View logs: tensorboard --logdir $OUTPUT_DIR"
echo "============================================="
echo ""
echo "Next steps:"
echo "1. Test with: python run_inbetweening.py --motion-id <id> \\"
echo "              --resume-pth $FINETUNED_VQVAE \\"
echo "              --resume-trans $OUTPUT_DIR/net_last.pth"

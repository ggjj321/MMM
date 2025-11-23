# AIST++ Fine-tuning Quick Start

快速開始使用 pretrained HumanML3D 模型在 AIST++ 數據上進行 fine-tuning。

## 前置準備

1. **確認數據格式**
   ```bash
   ls AIST++/
   # 應該看到: Mean.npy, Std.npy, new_joint_vecs/, texts/, train.txt, val.txt, test.txt
   ```

2. **準備 Pretrained Models**
   - VQ-VAE checkpoint: `checkpoints/vqvae/net_last.pth`
   - Transformer checkpoint: `checkpoints/transformer/net_last.pth`

## 訓練流程

### Step 1: Fine-tune VQ-VAE

1. 編輯 `train_vqvae_aist.sh`，更新 pretrained model 路徑：
   ```bash
   PRETRAINED_VQVAE="/path/to/pretrained/vqvae/net_last.pth"
   ```

2. 運行訓練：
   ```bash
   ./train_vqvae_aist.sh
   ```

3. 監控訓練（另一個terminal）：
   ```bash
   tensorboard --logdir output/vq_aist_*
   ```

4. 訓練完成後，checkpoint 保存在：
   ```
   output/vq_aist_<timestamp>/net_last.pth
   ```

### Step 2: Fine-tune Transformer

1. 編輯 `train_transformer_aist.sh`，更新路徑：
   ```bash
   FINETUNED_VQVAE="output/vq_aist_<timestamp>/net_last.pth"  # Stage 1 輸出
   PRETRAINED_TRANS="/path/to/pretrained/transformer/net_last.pth"
   ```

2. 運行訓練：
   ```bash
   ./train_transformer_aist.sh
   ```

3. 監控訓練：
   ```bash
   tensorboard --logdir output/trans_aist_*
   ```

### Step 3: 測試 In-betweening

```bash
python run_inbetweening.py \
    --motion-id gJB_sBM_cAll_d08_mJB5_ch02 \
    --resume-pth output/vq_aist_<timestamp>/net_last.pth \
    --resume-trans output/trans_aist_<timestamp>/net_last.pth \
    --inbetween-text "a person dances energetically"
```

輸出會保存在 `./output_inbetween/`

## 關鍵參數調整

| 參數 | 默認值 | 建議範圍 | 說明 |
|------|--------|----------|------|
| Batch Size (VQ-VAE) | 256 | 128-512 | 根據 GPU 記憶體調整 |
| Batch Size (Trans) | 128 | 64-256 | 同上 |
| Learning Rate | 2e-4 | 1e-4 ~ 5e-4 | Fine-tuning 可用較小 LR |
| Total Iterations | 100K/150K | 50K-200K | 根據數據量調整 |

## 疑難排解

**CUDA Out of Memory**
```bash
# 降低 batch size
# 在 train_vqvae_aist.sh 中修改
BATCH_SIZE=128  # 原本 256
```

**找不到數據**
```bash
# 檢查 dataset loader
# 方案1: 修改 dataset/dataset_VQ.py 添加 'aist' 支持
# 方案2: 使用 --dataname t2m (如果完全兼容 HumanML3D 格式)
```

**Training 不收斂**
- 降低 learning rate
- 確認數據正確加載（檢查 shapes）
- 增加 warmup iterations

## 完整文檔

詳細說明請參考：[training_guide.md](file:///.gemini/antigravity/brain/e158f881-0c91-4603-89e5-bfbd65ac3e4a/training_guide.md)

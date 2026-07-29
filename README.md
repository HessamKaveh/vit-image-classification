# Vision Transformer (ViT) for Image Classification

A Vision Transformer implemented from scratch (patch embedding, multi-head
self-attention, transformer encoder blocks) and trained on CIFAR-10.

## Pipeline
1. Split images into fixed-size patches (4x4), linearly embed each patch
2. Prepend a learnable [CLS] token, add learnable positional embeddings
3. Pass through 6 transformer encoder blocks (multi-head self-attention + MLP)
4. Classify using the [CLS] token's final representation
5. Train with AdamW + cosine learning rate schedule

## Architecture
- Patch size: 4x4, embedding dim: 192
- Depth: 6 transformer blocks, 6 attention heads
- No pretraining — trained from scratch on CIFAR-10

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/train.py
python src/evaluate.py
```

## Results
See `results/training_curves.png` and `results/confusion_matrix.png`

## Author
Hessam Kaveh — Research Fellow, Italian Institute of Technology

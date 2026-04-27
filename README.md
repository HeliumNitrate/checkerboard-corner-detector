# Checkerboard Corner Detector

EfficientNetV2-S encoder + U-Net decoder for checkerboard corner detection.
Used as the first stage of the [OpticalDistortionLab](https://opticaldistortionlab.com) calibration pipeline.

## Quick Start (Colab)

Open `train_colab.ipynb` in Google Colab and run all cells.

## Architecture

- **Encoder**: EfficientNetV2-S (ImageNet pretrained), feature maps at strides [2,4,8,16,32]
- **Decoder**: U-Net with skip connections → full-resolution heatmap
- **Output**: (B, 1, H, W) logit map — sigmoid gives corner probability per pixel
- **Loss**: Focal BCE (gamma=2, alpha=0.5)

## Training

```bash
pip install torch timm scipy Pillow
cd <repo>
python train.py --epochs 50 --batch 8 --n_train 8000 --n_val 1000
```

## Inference

```python
from infer import load_model, estimate_distortion

model = load_model('checkpoints/best.pt')
constants, info = estimate_distortion(
    model, img,
    n_inner_rows=15, n_inner_cols=15,
    image_center=(cy, cx),
    pix_size_norm=pix_size / focal_length,
    pol_degree=[3, 5, 7],
)
```

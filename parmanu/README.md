# KLA Problem Statement – AI-Based Restoration of Degraded Images

###  Team Name: `parmanu`

---

## Solution Architecture: AdaIR (Adaptive All-in-One Image Restoration)

This directory contains the official competition submission package for **Team parmanu**. Our solution employs **AdaIR** (*ICLR 2024*), a dual-domain deep learning architecture leveraging frequency-domain subband mining and spatial self-attention for joint noise removal and 2x super-resolution restoration of degraded semiconductor images.

---

##  Directory Layout

```text
parmanu/
├── run.py                 # Primary entry script: python run.py <input-dir> <output-dir>
├── requirements.txt       # Dependencies with exact version specifications
├── README.md              # Technical and execution documentation
└── models/
    ├── arch.py            # Model architecture definition
    └── ADAHYPER.pth       # Trained PyTorch model weights checkpoint (72.7MB)
```

---

##  Technical Specifications & Compliance

- **Input Format**: Grayscale `.npy` files (`128x128` low-resolution noisy images).
- **Output Format**: Grayscale 2x super-resolved restored `.npy` files (`256x256`, `np.float32`).
- **Output Post-Processing**:
  - Normalized range strictly in `[0.0, 1.0]` via `.clip(0.0, 1.0)`.
  - Guaranteed zero `NaN` or `Inf` values via `np.nan_to_num()`.
  - Shape: 2D array `(H, W)` matching input filename.
- **Test-Time Augmentation (TTA)**: 8-fold flip and rotation ensemble during inference to maximize PSNR and SSIM fidelity.

---

##  Execution Guide

### Command Format
```bash
python run.py <input-dir> <output-dir>
```

### Example
```bash
python run.py /content/train/NoisyLR /content/train/RestoredOutput
```

---

##  Offline Execution
This package is 100% self-contained and offline-ready. It requires zero internet access, no API keys, no additional weight downloads, and runs automatically on CUDA GPUs or CPU fallbacks.

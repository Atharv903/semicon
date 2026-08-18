#  KLA Hackathon - AI-Based Restoration of Degraded Images

###  Team Name: `parmanu`

---

##  Executive Summary & Key Results

Our solution employs **AdaIR** (*ICLR 2024*), an adaptive dual-domain transformer architecture engineered for joint noise removal and 2x super-resolution restoration of degraded semiconductor images.

| Metric | Single Pass | 8-Fold TTA (Test-Time Augmentation) |
| :--- | :--- | :--- |
| **Peak Validation PSNR** | **33.49 dB** | **33.50 dB** |
| **Peak Validation SSIM** | **0.9040** | **0.9048** |
| **Average Validation PSNR** | **26.61 dB** | **26.66 dB** |
| **Average Validation SSIM** | **0.7408** | **0.7433** |
| **GPU Inference Latency** | **~10 ms / image** | **~50 ms / image** |
| **Inference Throughput** | **~100 imgs / sec** | **~15–20 imgs / sec** |

---

##  Repository Structure & Required Deliverables Checklist

```text
semicon/
├── parmanu/                     #  Primary Official Submission Folder
│   ├── run.py                   # Standalone Evaluation Script: python run.py <input-dir> <output-dir>
│   ├── train.py                 # Standalone Training Script: python train.py --lr_dir <path> --gt_dir <path>
│   ├── requirements.txt         # Dependencies with exact version specifications
│   ├── README.md                # Submission guide for Team Parmanu
│   └── models/
│       ├── arch.py              # Self-contained AdaIR model architecture definition
│       └── ADAHYPER.pth         # Final trained PyTorch model weights checkpoint (72.7MB)
├── restored_test_outputs/       #  Restored Output Files for all 400 test images (000000.npy - 000399.npy)
├── notebooks/                   #  Self-contained Training Notebooks for Google Colab / Kaggle
│   ├── Colab_HyperFast_BicubicAdaIR.ipynb
│   ├── Colab_AdaIR_Optimized.ipynb
│   └── Kaggle_AdaIR_Optimized.ipynb
├── app/                         #  Interactive Web Application for real-time visual inspection & benchmarking
│   ├── app.py
│   └── templates/
├── run.py                       # Top-level runner alias calling parmanu/run.py
├── requirements.txt             # Environment requirements specification
└── README.md                    # Project documentation & complete evaluation guide
```

###  Deliverables Verification:
1. **README.md**: Complete instructions to clone and run inference without manual edits or contacting the team.
2. **Evaluation Script**: `parmanu/run.py` & `run.py` accepting `<input-dir>` and `<output-dir>`.
3. **Training Script**: `parmanu/train.py` & `notebooks/Colab_HyperFast_BicubicAdaIR.ipynb`.
4. **Trained Model Weights**: `parmanu/models/ADAHYPER.pth` (72.7MB, direct repository commit).
5. **Restored Test Outputs**: `restored_test_outputs/` (Contains all 400 `.npy` restored test outputs).
6. **requirements.txt**: Environment requirements file for exact reproduction.

---

##  Quickstart: Setup & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Atharv903/semicon.git
cd semicon
pip install -r requirements.txt
```

---

##  Running Inference (Evaluation Script)

To run inference on any folder of degraded `.npy` files:

```bash
python run.py <input-dir> <output-dir>
```

#### Example Command:
```bash
python run.py ./Dataset/NoisyLR ./Restored_Outputs
```

- **Inputs**: Directory containing degraded `.npy` files (`128x128`).
- **Outputs**: Directory containing restored `.npy` files (`256x256`, float32, range `[0.0, 1.0]`, zero NaNs/Infs).

---

##  Reproducing Training (Training Script)

To train the AdaIR model from scratch:

```bash
python parmanu/train.py --lr_dir /path/to/NoisyLR --gt_dir /path/to/GT --epochs 30 --batch_size 8
```

Alternatively, open `notebooks/Colab_HyperFast_BicubicAdaIR.ipynb` directly in Google Colab (T4 GPU) to reproduce training in ~6 minutes.

---

##  Interactive Web Application

We built an interactive Flask web interface to visually inspect restorations side-by-side with ground truth images, benchmark PSNR/SSIM, and toggle TTA:

```bash
python app/app.py
```
Open **`http://127.0.0.1:5000`** in your web browser.

---

##  Model Architecture Overview: AdaIR

Our solution utilizes **AdaIR** (*ICLR 2024*), featuring:
1. **Adaptive Frequency Learning Block (AFLB / FreModule)**: Subband frequency decomposition via Fast Fourier Transforms (FFT) to isolate high-frequency semiconductor edges from noise.
2. **Channel-Wise Cross-Attention (CA)**: Global contextual feature modulation without quadratic spatial overhead $O(N \cdot C^2)$.
3. **Gated-Dconv Feed-Forward Network (GDFN)**: Non-linear gating to enhance structural sharpness.
4. **Native 2x PixelShuffle Upsampler**: Integrated sub-pixel convolution upsampler.

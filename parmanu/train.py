#!/usr/bin/env python3
"""
Team Parmanu - Training Script for AdaIR (Adaptive All-in-One Image Restoration)
KLA Problem Statement: AI-Based Restoration of Degraded Images

This script trains the AdaIR model from scratch using joint Charbonnier loss,
SSIM loss, and Frequency FFT loss.
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Local model imports
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "models")
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from arch import AdaIR

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + (self.eps * self.eps))
        return torch.mean(loss)

class FFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="backward")
        target_fft = torch.fft.rfft2(target, norm="backward")
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        return self.criterion(pred_mag, target_mag)

class NpyPairDataset(Dataset):
    def __init__(self, lr_dir, gt_dir):
        self.lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.npy")))
        self.gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
        assert len(self.lr_files) == len(self.gt_files), "LR and GT counts must match"

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr = np.load(self.lr_files[idx]).astype(np.float32)
        gt = np.load(self.gt_files[idx]).astype(np.float32)
        
        lr_t = torch.from_numpy(lr).unsqueeze(0) if lr.ndim == 2 else torch.from_numpy(lr.transpose(2,0,1))
        gt_t = torch.from_numpy(gt).unsqueeze(0) if gt.ndim == 2 else torch.from_numpy(gt.transpose(2,0,1))
        return lr_t, gt_t

def train(lr_dir, gt_dir, output_dir, epochs=30, batch_size=8, lr=2e-4):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training AdaIR on Device: {device}...")

    model = AdaIR(
        inp_channels=1,
        out_channels=1,
        dim=48,
        num_blocks=[2, 3, 3, 4],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66
    ).to(device)

    dataset = NpyPairDataset(lr_dir, gt_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    charbonnier_loss = CharbonnierLoss()
    fft_loss = FFTLoss()

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start = time.time()

        for lr_img, gt_img in dataloader:
            lr_img, gt_img = lr_img.to(device), gt_img.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                pred = model(lr_img)
                l_spatial = charbonnier_loss(pred, gt_img)
                l_freq = fft_loss(pred, gt_img)
                loss = l_spatial + 0.1 * l_freq

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - start
        print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.6f} - Time: {elapsed:.2f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(output_dir, "ADAHYPER.pth")
            torch.save({"model_state_dict": model.state_dict(), "loss": best_loss}, ckpt_path)
            print(f" Saved best model checkpoint to {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AdaIR Model for Image Restoration")
    parser.add_argument("--lr_dir", type=str, required=True, help="Directory containing NoisyLR .npy files")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing GT .npy files")
    parser.add_argument("--output_dir", type=str, default="./parmanu/models", help="Directory to save trained ADAHYPER.pth")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    train(args.lr_dir, args.gt_dir, args.output_dir, args.epochs, args.batch_size)

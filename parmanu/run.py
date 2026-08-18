#!/usr/bin/env python3
"""
KLA Hackathon - AI-Based Restoration of Degraded Images
Submission Script: run.py
Command: python run.py <input-dir> <output-dir>
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add local models directory to path for offline self-contained execution
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "models")
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from arch import AdaIR

class NpyInferenceDataset(Dataset):
    def __init__(self, input_dir):
        self.files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
        if len(self.files) == 0:
            print(f"Warning: No .npy files found in {input_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        filename = os.path.basename(path)
        img = np.load(path).astype(np.float32)
        
        # Ensure 2D (H, W) or 3D (H, W, 1) -> convert to 1xHxW tensor
        if img.ndim == 2:
            img_tensor = torch.from_numpy(img).unsqueeze(0)
        elif img.ndim == 3 and img.shape[2] == 1:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1))
        elif img.ndim == 3 and img.shape[0] == 1:
            img_tensor = torch.from_numpy(img)
        else:
            # Fallback if channel dimension is unexpected
            img_tensor = torch.from_numpy(img).unsqueeze(0)

        return img_tensor, filename

def find_checkpoint(models_dir):
    """Find the model checkpoint file inside models/ directory."""
    possible_names = [
        "ADAHYPER.pth",
        "adair_hyperfast_best.pth",
        "adair_colab_optimized_best.pth",
        "best_model.pth",
        "model.pth",
        "adair_kaggle_best.pth"
    ]
    for name in possible_names:
        p = os.path.join(models_dir, name)
        if os.path.exists(p):
            return p
    
    # Fallback to any .pth file found in models_dir
    pth_files = glob.glob(os.path.join(models_dir, "*.pth"))
    if pth_files:
        return pth_files[0]
    
    raise FileNotFoundError(f"No .pth model weights found in {models_dir}")

def run_inference(input_dir, output_dir, use_tta=True, batch_size=8):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # Load Model
    ckpt_path = find_checkpoint(models_dir)
    print(f"Loading weights from: {ckpt_path}")
    
    model = AdaIR(
        inp_channels=1, 
        out_channels=1, 
        dim=48,
        num_blocks=[2, 3, 3, 4], 
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)

    model.eval()

    dataset = NpyInferenceDataset(input_dir)
    if len(dataset) == 0:
        print("No files to process. Exiting.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Processing {len(dataset)} files from '{input_dir}' -> '{output_dir}' (TTA={use_tta})...")
    start_time = time.time()
    count = 0

    with torch.no_grad():
        for batch_imgs, filenames in dataloader:
            batch_imgs = batch_imgs.to(device)

            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    if not use_tta:
                        preds = model(batch_imgs)
                    else:
                        # 8-fold Test-Time Augmentation for maximum PSNR/SSIM boost
                        preds = model(batch_imgs)
                        for k in range(1, 4):
                            preds += torch.rot90(model(torch.rot90(batch_imgs, k, [2, 3])), -k, [2, 3])
                        batch_flip = torch.flip(batch_imgs, [2])
                        preds += model(batch_flip).flip([2])
                        for k in range(1, 4):
                            preds += torch.rot90(model(torch.rot90(batch_flip, k, [2, 3])), -k, [2, 3]).flip([2])
                        preds /= 8.0
            else:
                if not use_tta:
                    preds = model(batch_imgs)
                else:
                    preds = model(batch_imgs)
                    for k in range(1, 4):
                        preds += torch.rot90(model(torch.rot90(batch_imgs, k, [2, 3])), -k, [2, 3])
                    batch_flip = torch.flip(batch_imgs, [2])
                    preds += model(batch_flip).flip([2])
                    for k in range(1, 4):
                        preds += torch.rot90(model(torch.rot90(batch_flip, k, [2, 3])), -k, [2, 3]).flip([2])
                    preds /= 8.0

            preds = preds.cpu().numpy()

            for i in range(len(filenames)):
                out_arr = preds[i, 0, :, :]
                
                # Compliance & Safety Checks
                out_arr = np.nan_to_num(out_arr, nan=0.0, posinf=1.0, neginf=0.0)
                out_arr = np.clip(out_arr, 0.0, 1.0).astype(np.float32)
                
                save_path = os.path.join(output_dir, filenames[i])
                np.save(save_path, out_arr)
                count += 1

    elapsed = time.time() - start_time
    print(f"\nSuccessfully restored {count} files in {elapsed:.2f}s ({count/max(elapsed, 0.001):.1f} files/sec).")

def main():
    parser = argparse.ArgumentParser(description="KLA Image Restoration Entry Script")
    parser.add_argument("input_dir", nargs="?", type=str, help="Input directory containing degraded .npy files")
    parser.add_argument("output_dir", nargs="?", type=str, help="Output directory to save restored .npy files")
    parser.add_argument("--no_tta", action="store_true", help="Disable TTA for faster execution")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    args = parser.parse_args()

    # Handle positional arguments from command line: python run.py <input-dir> <output-dir>
    if len(sys.argv) >= 3 and not sys.argv[1].startswith("--"):
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
    elif args.input_dir and args.output_dir:
        input_dir = args.input_dir
        output_dir = args.output_dir
    else:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    use_tta = not args.no_tta
    run_inference(input_dir, output_dir, use_tta=use_tta, batch_size=args.batch_size)

if __name__ == "__main__":
    main()

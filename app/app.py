import os
import sys
import time
import base64
import io
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template, request, jsonify

# Include model paths
sys.path.append(r"d:\hackthon\SemiHack\models")
sys.path.append(r"d:\hackthon\SemiHack\models\nafnet_w32")
sys.path.append(r"d:\hackthon\SemiHack\models\ada_naf")
sys.path.append(r"d:\hackthon\SemiHack\models\adair")

from arch import NAFNetW64
from safmn_arch import SAFMNet
from metrics import calculate_psnr, calculate_ssim_tensor

app = Flask(__name__)

# Dataset directory
GT_DIR = r"d:\hackthon\SemiHack\Dataset\Data-public-20260816T151605Z-1-001\Data-public\train\train\GT"
LR_DIR = r"d:\hackthon\SemiHack\Dataset\Data-public-20260816T151605Z-1-001\Data-public\train\train\NoisyLR"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing Flask Web App backend on Device: {device}...")

# Preload Models
models_dict = {}

net64_path = r"d:\hackthon\SemiHack\models\nafnet_w32\breathrough.pth"
if os.path.exists(net64_path):
    print("Loading NAFNet-w64 (breathrough.pth)...")
    m64 = NAFNetW64(width=64).to(device)
    ckpt = torch.load(net64_path, map_location=device)
    m64.load_state_dict(ckpt.get("model_state_dict", ckpt))
    m64.eval()
    models_dict["nafnet_w64"] = m64

from ada_naf.arch import AdaFreqNAFNet
from adair.arch import AdaIR

iter2_path = r"d:\hackthon\SemiHack\models\ada_naf\iter2.pth"
if os.path.exists(iter2_path):
    print("Loading AdaFreq-NAFNet Epoch 30 Model (iter2.pth - 28.75 dB)...")
    m_ada = AdaFreqNAFNet(width=64).to(device)
    ckpt = torch.load(iter2_path, map_location=device)
    m_ada.load_state_dict(ckpt.get("model_state_dict", ckpt))
    m_ada.eval()
    models_dict["ada_naf_ep30"] = m_ada

adair_path = r"d:\hackthon\SemiHack\models\adair\ADAHYPER.pth"
if not os.path.exists(adair_path):
    adair_path = r"d:\hackthon\SemiHack\models\adair\checkpoints\adair_best.pth"

if os.path.exists(adair_path):
    print(f"Loading Hyper-Fast AdaIR Model ({os.path.basename(adair_path)})...")
    m_adair = AdaIR(
        inp_channels=1, 
        out_channels=1, 
        dim=48,
        num_blocks=[2, 3, 3, 4], 
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66
    ).to(device)
    ckpt = torch.load(adair_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        m_adair.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        m_adair.load_state_dict(ckpt["state_dict"])
    else:
        m_adair.load_state_dict(ckpt)
    m_adair.eval()
    models_dict["adair"] = m_adair

def predict_8fold_tta(model, x):
    transforms = [
        (lambda img: img, lambda img: img),
        (lambda img: torch.rot90(img, 1, [2, 3]), lambda img: torch.rot90(img, -1, [2, 3])),
        (lambda img: torch.rot90(img, 2, [2, 3]), lambda img: torch.rot90(img, -2, [2, 3])),
        (lambda img: torch.rot90(img, 3, [2, 3]), lambda img: torch.rot90(img, -3, [2, 3])),
        (lambda img: torch.flip(img, [3]), lambda img: torch.flip(img, [3])),
        (lambda img: torch.flip(torch.rot90(img, 1, [2, 3]), [3]), lambda img: torch.rot90(torch.flip(img, [3]), -1, [2, 3])),
        (lambda img: torch.flip(torch.rot90(img, 2, [2, 3]), [3]), lambda img: torch.rot90(torch.flip(img, [3]), -2, [2, 3])),
        (lambda img: torch.flip(torch.rot90(img, 3, [2, 3]), [3]), lambda img: torch.rot90(torch.flip(img, [3]), -3, [2, 3])),
    ]
    preds = []
    for forward_tf, backward_tf in transforms:
        augmented_input = forward_tf(x)
        pred = model(augmented_input)
        restored_pred = backward_tf(pred)
        preds.append(restored_pred)
    return torch.mean(torch.stack(preds, dim=0), dim=0)

def tensor_to_base64(t):
    arr = (t.squeeze().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.route("/")
def index():
    files = sorted([f for f in os.listdir(GT_DIR) if f.endswith(".npy")])
    return render_template("index.html", sample_files=files[:50])

def laplacian_kernel():
    return torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

def adaptive_sharpen(img_tensor, alpha=0.25):
    if alpha <= 0.0:
        return img_tensor
    kernel = laplacian_kernel()
    edges = F.conv2d(img_tensor, kernel, padding=1)
    return torch.clamp(img_tensor + alpha * edges, 0.0, 1.0)

@app.route("/api/infer", methods=["POST"])
def api_infer():
    data = request.json
    filename = data.get("filename", "000000.npy")
    selected_model = data.get("model", "adair")
    use_tta = data.get("use_tta", True)
    sharpness = float(data.get("sharpness", 0.20))

    gt_path = os.path.join(GT_DIR, filename)
    lr_path = os.path.join(LR_DIR, filename)

    if not os.path.exists(gt_path) or not os.path.exists(lr_path):
        return jsonify({"error": "File not found"}), 404

    gt_np = np.load(gt_path).astype(np.float32)
    lr_np = np.load(lr_path).astype(np.float32)

    lr_t = torch.from_numpy(lr_np).unsqueeze(0).unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).to(device)

    # Bicubic baseline
    bicubic_t = F.interpolate(lr_t, scale_factor=2, mode="bicubic", align_corners=False)
    psnr_bicubic = calculate_psnr(bicubic_t, gt_t)

    t0 = time.time()
    with torch.no_grad():
        if selected_model == "adair" and "adair" in models_dict:
            pred_t = predict_8fold_tta(models_dict["adair"], lr_t) if use_tta else models_dict["adair"](lr_t)
        elif selected_model == "ada_naf_ep30" and "ada_naf_ep30" in models_dict:
            pred_t = predict_8fold_tta(models_dict["ada_naf_ep30"], lr_t) if use_tta else models_dict["ada_naf_ep30"](lr_t)
        elif selected_model == "nafnet_w64" and "nafnet_w64" in models_dict:
            pred_t = predict_8fold_tta(models_dict["nafnet_w64"], lr_t) if use_tta else models_dict["nafnet_w64"](lr_t)
        elif selected_model == "ensemble":
            # Multi-Model Ensemble
            preds_list = []
            if "ada_naf_ep30" in models_dict:
                preds_list.append(predict_8fold_tta(models_dict["ada_naf_ep30"], lr_t) if use_tta else models_dict["ada_naf_ep30"](lr_t))
            if "nafnet_w64" in models_dict:
                preds_list.append(predict_8fold_tta(models_dict["nafnet_w64"], lr_t) if use_tta else models_dict["nafnet_w64"](lr_t))
            
            if len(preds_list) > 0:
                pred_t = torch.mean(torch.stack(preds_list, dim=0), dim=0)
            else:
                pred_t = bicubic_t
        else:
            pred_t = bicubic_t

        # Apply High-Frequency Detail Sharpening
        if sharpness > 0.0 and selected_model != "bicubic":
            pred_t = adaptive_sharpen(pred_t, alpha=sharpness)

    latency_ms = (time.time() - t0) * 1000.0
    psnr_val = calculate_psnr(pred_t, gt_t)
    ssim_val = calculate_ssim_tensor(pred_t, gt_t)

    return jsonify({
        "filename": filename,
        "model": selected_model,
        "use_tta": use_tta,
        "psnr": round(psnr_val, 2),
        "ssim": round(ssim_val, 4),
        "psnr_bicubic": round(psnr_bicubic, 2),
        "psnr_boost": round(psnr_val - psnr_bicubic, 2),
        "latency_ms": round(latency_ms, 1),
        "img_noisy": tensor_to_base64(bicubic_t),
        "img_denoised": tensor_to_base64(pred_t),
        "img_gt": tensor_to_base64(gt_t),
    })

if __name__ == "__main__":
    os.makedirs(r"d:\hackthon\SemiHack\app\templates", exist_ok=True)
    os.makedirs(r"d:\hackthon\SemiHack\app\static\css", exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False)

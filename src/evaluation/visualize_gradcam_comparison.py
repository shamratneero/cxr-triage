"""
Visual model comparison: for a set of sample X-rays, shows the original image
(with ground-truth bounding box annotated in red, if one exists for that finding)
next to each model's GradCAM heatmap overlay + predicted confidence.

Saves one PNG grid per case into OUTPUT_DIR. Run this AFTER full_benchmark.py's
CONVNEXT_CHECKPOINT path is filled in and ConvNeXt has finished training.

Layout per case:  [ Original + GT box ] [ DenseNet_BCE ] [ DenseNet_Focal ] [ DenseNet_CLAHE_Focal ] [ ConvNeXt_Focal ]
"""

import torch
import numpy as np
import pandas as pd
import cv2
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.append('D:/cxr-triage')

from src.models.densenet import DenseNetModel
from src.models.convnext import ConvNeXtModel
from src.inference.gradcam import GradCAM
from src.data.transforms import get_val_transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_ROOT = "F:/X ray dataset/Second Version"
OUTPUT_DIR = "D:/cxr-triage/reports/gradcam_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = [
    'Atelectasis', 'Consolidation', 'Infiltration',
    'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
    'Effusion', 'Pneumonia', 'Pleural_Thickening',
    'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}

BBOX_TO_LABEL = {
    'Atelectasis': 'Atelectasis',
    'Cardiomegaly': 'Cardiomegaly',
    'Effusion': 'Effusion',
    'Infiltrate': 'Infiltration',
    'Mass': 'Mass',
    'Nodule': 'Nodule',
    'Pneumonia': 'Pneumonia',
    'Pneumothorax': 'Pneumothorax'
}

# ─── EDIT ONCE CONVNEXT FINISHES TRAINING ──────────────────────────────
CONVNEXT_CHECKPOINT = 'D:/cxr-triage/checkpoints/convnext_focal_fixed/best_model.pth'
CONVNEXT_IMAGE_SIZE = 224
CONVNEXT_USE_CLAHE = False
# ─────────────────────────────────────────────────────────────────────

MODELS = {
    'DenseNet_BCE':         {'ckpt': 'D:/cxr-triage/checkpoints/densenet_bce_fixed/best_model.pth',
                              'image_size': 224, 'use_clahe': False, 'arch': 'densenet'},
    'DenseNet_Focal':       {'ckpt': 'D:/cxr-triage/checkpoints/densenet_focal_fixed/best_model.pth',
                              'image_size': 224, 'use_clahe': False, 'arch': 'densenet'},
    'DenseNet_CLAHE_Focal': {'ckpt': 'D:/cxr-triage/checkpoints/clahe_320_logits_fix/best_model.pth',
                              'image_size': 320, 'use_clahe': True, 'arch': 'densenet'},
    'ConvNeXt_Focal':       {'ckpt': CONVNEXT_CHECKPOINT,
                              'image_size': CONVNEXT_IMAGE_SIZE, 'use_clahe': CONVNEXT_USE_CLAHE, 'arch': 'convnext'},
}

DISPLAY_SIZE = 320       # every image/heatmap gets resized to this for a consistent grid
ORIGINAL_SIZE = 1024
PERCENTILE = 90          # for the pred_mask, kept consistent with the benchmark script
N_CASES_PER_BBOX_CLASS = 2   # how many sample images to pull per bbox-annotated finding
N_CASES_NO_BBOX = 6          # extra cases for findings without bbox ground truth


def find_image(image_name, base_root):
    for folder in os.listdir(base_root):
        if folder.startswith('images_'):
            path = os.path.join(base_root, folder, 'images', image_name)
            if os.path.exists(path):
                return path
    return None


def load_model(checkpoint_path, arch):
    if arch == 'densenet':
        model = DenseNetModel(num_classes=14, pretrained=False).to(DEVICE)
    elif arch == 'convnext':
        model = ConvNeXtModel(num_classes=14, pretrained=False).to(DEVICE)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def overlay_heatmap(img_rgb_uint8, heatmap):
    """img_rgb_uint8: HxWx3 uint8. heatmap: HxW float in [0,1] (already resized to match)."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb_uint8, 0.6, colored, 0.4, 0)
    return overlay


def scale_bbox(gx, gy, gw, gh, from_size, to_size):
    scale = to_size / from_size
    return gx * scale, gy * scale, gw * scale, gh * scale


test_df = pd.read_csv('D:/cxr-triage/data/processed/test.csv')
bbox_df_raw = pd.read_csv('F:/X ray dataset/Second Version/BBox_List_2017.csv')
bbox_df_raw = bbox_df_raw.rename(columns={'Bbox [x': 'x', 'y': 'y', 'w': 'w', 'h]': 'h'})[
    ['Image Index', 'Finding Label', 'x', 'y', 'w', 'h']
]
test_images = set(test_df['Image Index'].values)
bbox_df = bbox_df_raw[bbox_df_raw['Image Index'].isin(test_images)].reset_index(drop=True)


def select_cases():
    """Returns list of dicts: {image_name, finding, bbox (or None)}"""
    cases = []

    # Cases WITH ground-truth bbox, spread across the 8 bbox-annotated findings
    for bbox_label in BBOX_TO_LABEL.keys():
        subset = bbox_df[bbox_df['Finding Label'] == bbox_label]
        picks = subset.sample(n=min(N_CASES_PER_BBOX_CLASS, len(subset)), random_state=42) if len(subset) else subset
        for _, row in picks.iterrows():
            cases.append({
                'image_name': row['Image Index'],
                'finding': BBOX_TO_LABEL[bbox_label],
                'bbox': (row['x'], row['y'], row['w'], row['h']),
            })

    # Extra cases WITHOUT bbox ground truth, for findings not covered by BBox_List_2017
    no_bbox_labels = [l for l in LABELS if l not in BBOX_TO_LABEL.values()]
    for label in no_bbox_labels[:N_CASES_NO_BBOX]:
        matches = test_df[test_df['Finding Labels'].astype(str).str.contains(label, na=False)]
        if len(matches) > 0:
            row = matches.sample(n=1, random_state=42).iloc[0]
            cases.append({
                'image_name': row['Image Index'],
                'finding': label,
                'bbox': None,
            })

    return cases


def build_case_figure(case, models_loaded):
    image_name = case['image_name']
    finding = case['finding']
    bbox = case['bbox']
    class_idx = LABEL_TO_IDX[finding]

    img_path = find_image(image_name, BASE_ROOT)
    if img_path is None:
        print(f"  Image not found: {image_name}, skipping case")
        return

    orig_img = Image.open(img_path).convert('RGB')
    orig_resized = orig_img.resize((DISPLAY_SIZE, DISPLAY_SIZE))
    orig_np = np.array(orig_resized)

    n_panels = 1 + len(models_loaded)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.6))

    # Panel 0: original + GT box
    axes[0].imshow(orig_np)
    axes[0].set_title(f"Original\n{image_name}\nFinding: {finding}", fontsize=9)
    axes[0].axis('off')
    if bbox is not None:
        gx, gy, gw, gh = scale_bbox(*bbox, from_size=ORIGINAL_SIZE, to_size=DISPLAY_SIZE)
        rect = patches.Rectangle((gx, gy), gw, gh, linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(gx, max(gy - 5, 0), 'ground truth', color='red', fontsize=8, weight='bold')
    else:
        axes[0].text(5, DISPLAY_SIZE - 10, 'no bbox annotation available', color='orange', fontsize=8)

    # Panels 1..N: each model's heatmap overlay + predicted confidence
    for i, (model_name, bundle) in enumerate(models_loaded.items(), start=1):
        model = bundle['model']
        transform = bundle['transform']
        image_size = bundle['image_size']
        gradcam = bundle['gradcam']

        img_tensor = transform(orig_img).unsqueeze(0)

        with torch.no_grad():
            logits = model(img_tensor.to(DEVICE))
            conf = torch.sigmoid(logits)[0, class_idx].item()

        heatmap = gradcam.generate(img_tensor.clone(), class_idx=class_idx)
        heatmap_resized = cv2.resize(heatmap.astype(np.float32), (DISPLAY_SIZE, DISPLAY_SIZE))
        hmax = heatmap_resized.max()
        heatmap_norm = heatmap_resized / hmax if hmax > 0 else heatmap_resized

        overlay = overlay_heatmap(orig_np, heatmap_norm)
        axes[i].imshow(overlay)
        axes[i].set_title(f"{model_name}\nconfidence: {conf:.3f}", fontsize=9)
        axes[i].axis('off')

        if bbox is not None:
            gx, gy, gw, gh = scale_bbox(*bbox, from_size=ORIGINAL_SIZE, to_size=DISPLAY_SIZE)
            rect = patches.Rectangle((gx, gy), gw, gh, linewidth=2, edgecolor='red', facecolor='none')
            axes[i].add_patch(rect)

    plt.tight_layout()
    safe_finding = finding.replace(' ', '_')
    out_path = os.path.join(OUTPUT_DIR, f"{safe_finding}_{image_name.replace('.png','')}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print("Loading models...")
    models_loaded = {}
    for name, cfg in MODELS.items():
        if not os.path.exists(cfg['ckpt']):
            print(f"  SKIPPING {name} — checkpoint not found at {cfg['ckpt']}")
            continue
        model = load_model(cfg['ckpt'], cfg['arch'])
        transform = get_val_transforms(image_size=cfg['image_size'], use_clahe=cfg['use_clahe'])
        gradcam = GradCAM(model)
        models_loaded[name] = {
            'model': model, 'transform': transform,
            'image_size': cfg['image_size'], 'gradcam': gradcam
        }
        print(f"  Loaded {name}")

    if not models_loaded:
        print("No models loaded — check checkpoint paths.")
        return

    cases = select_cases()
    print(f"\nSelected {len(cases)} cases. Generating comparison figures...")

    for case in cases:
        print(f"\nCase: {case['image_name']} | {case['finding']} | bbox={'yes' if case['bbox'] else 'no'}")
        build_case_figure(case, models_loaded)

    for bundle in models_loaded.values():
        bundle['gradcam'].remove_hooks()

    print(f"\nDone. All comparison figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

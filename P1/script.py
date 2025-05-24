#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segment_and_evaluate_with_global.py

Comme segment_and_evaluate.py mais en fin de traitement:
  - on crée une image `GlobalResult.png` qui empile **verticalement**
    tous les fichiers `*_viz.png` générés, pour un aperçu global.

Dépendances :
    pip install torch transformers pillow numpy matplotlib scikit-learn
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Forcer backend non interactif
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from sklearn.model_selection import train_test_split

# ────────────────────────────────────────────────────────────────────────────────
# 1) CONFIGURATION DU MODELE
#────────────────────────────────────────────────────────────────────────────────
MODEL_ID = "mattmdjaga/segformer_b2_clothes"
processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
model     = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

id2label = {int(k):v for k,v in model.config.id2label.items()}
PALETTE = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),
           (255,0,255),(0,255,255),(128,0,0),(0,128,0)]
def get_color(lbl:int): return PALETTE[lbl % len(PALETTE)]

# ────────────────────────────────────────────────────────────────────────────────
# 2) CHEMINS & PRÉP
#────────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
IMG_DIR    = os.path.join(BASE_DIR, "top_influenceurs_2024", "IMG")
MASKS_DIR  = os.path.join(BASE_DIR, "masks")
GLOBAL_OUT = os.path.join(MASKS_DIR, "GlobalResult.png")
os.makedirs(MASKS_DIR, exist_ok=True)

if not os.path.isdir(IMG_DIR):
    print(f"[ERROR] Dossier introuvable : {IMG_DIR}")
    sys.exit(1)

EXTS = {".jpg",".jpeg",".png",".bmp",".gif"}
FONT = ImageFont.load_default()
confidences = []
viz_paths = []  # pour stocker tous les *_viz.png

# ────────────────────────────────────────────────────────────────────────────────
# 3) TRAITEMENT & VISUALISATION POUR CHAQUE IMAGE
#────────────────────────────────────────────────────────────────────────────────
for fname in sorted(os.listdir(IMG_DIR)):
    name, ext = os.path.splitext(fname.lower())
    if ext not in EXTS:
        continue

    img_path = os.path.join(IMG_DIR, fname)
    print(f"[INFO] Traitement de {fname}")

    # a) Charger l’image
    image = Image.open(img_path).convert("RGB")

    # b) Pré-traitement et inférence
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    up = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )

    # c) Calcul de la confiance moyenne (mean softmax max)
    probs     = torch.softmax(up, dim=1)[0]
    max_probs = probs.max(dim=0).values.cpu().numpy()
    mean_conf = float(max_probs.mean())
    confidences.append(mean_conf)

    # d) Argmax → carte de labels
    pred = up.argmax(dim=1)[0].cpu().numpy()

    # e) Créer et sauvegarder masque grayscale
    ml = pred.max() or 1
    gray_arr = (pred.astype(np.float32)/ml*255).astype(np.uint8)
    mask_gray = Image.fromarray(gray_arr, mode="L")
    mask_gray.save(os.path.join(MASKS_DIR, f"{name}_mask.png"))

    # f) Créer et sauvegarder masque color + légende
    h,w = pred.shape
    color_arr = np.zeros((h,w,3),dtype=np.uint8)
    for lbl in np.unique(pred):
        color_arr[pred==lbl] = get_color(lbl)
    mask_color = Image.fromarray(color_arr)
    mask_color.save(os.path.join(MASKS_DIR, f"{name}_mask_color.png"))

    # g) Générer la légende en image
    dummy = Image.new("RGB",(1,1))
    draw_dummy = ImageDraw.Draw(dummy)
    _,_,_,lh = draw_dummy.textbbox((0,0),"Hg",font=FONT)
    legend_h = lh*len(id2label)+10
    legend = Image.new("RGB",(200,legend_h),"white")
    draw = ImageDraw.Draw(legend)
    y=5
    for lbl,label in id2label.items():
        draw.rectangle((5,y,5+lh-4,y+lh-4),fill=get_color(lbl))
        draw.text((5+lh,y), label, font=FONT, fill="black")
        y+=lh

    # h) Combiner mask_color + légende
    total_w = mask_color.width + legend.width
    total_h = max(mask_color.height, legend.height)
    combined = Image.new("RGB",(total_w,total_h),"white")
    combined.paste(mask_color,(0,0))
    combined.paste(legend,(mask_color.width,0))
    legend_path = os.path.join(MASKS_DIR, f"{name}_mask_color_legend.png")
    combined.save(legend_path)

    # i) Générer et sauvegarder la figure comparative
    viz_path = os.path.join(MASKS_DIR, f"{name}_viz.png")
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    axes[0].imshow(image); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(mask_gray, cmap="gray"); axes[1].set_title("Grayscale"); axes[1].axis("off")
    axes[2].imshow(combined); axes[2].set_title(f"Color+Legend\nConf={mean_conf:.2f}"); axes[2].axis("off")
    plt.tight_layout()
    fig.savefig(viz_path)
    plt.close(fig)
    print(f"[SAVED] {viz_path}")

    viz_paths.append(viz_path)

# ────────────────────────────────────────────────────────────────────────────────
# 4) Création de GlobalResult.png (stack vertical de tous les viz)
#────────────────────────────────────────────────────────────────────────────────
if viz_paths:
    print("[INFO] Génération de GlobalResult.png…")
    # Charger toutes les viz et mesurer dimensions
    imgs = [Image.open(p) for p in viz_paths]
    widths, heights = zip(*(i.size for i in imgs))
    max_w = max(widths)
    total_h = sum(heights)

    # Nouvel canevas
    global_img = Image.new("RGB", (max_w, total_h), "white")
    y_offset = 0
    for im in imgs:
        global_img.paste(im, (0, y_offset))
        y_offset += im.height

    global_img.save(GLOBAL_OUT)
    print(f"[SAVED] GlobalResult -> {GLOBAL_OUT}")

# ────────────────────────────────────────────────────────────────────────────────
# 5) Rapport de confiance moyen
#────────────────────────────────────────────────────────────────────────────────
avg_conf = np.mean(confidences) if confidences else 0.0
print(f"\n[REPORT] Confiance moyenne sur {len(confidences)} images: {avg_conf:.3f}")

# ────────────────────────────────────────────────────────────────────────────────
# 6) Validation : split + IoU/Dice (identique à avant)
#────────────────────────────────────────────────────────────────────────────────
def compute_iou(pred, gt, num_classes):
    ious=[]
    for c in range(num_classes):
        inter = ((pred==c)&(gt==c)).sum()
        union = ((pred==c)|(gt==c)).sum()
        ious.append(inter/union if union>0 else np.nan)
    return np.array(ious), np.nanmean(ious)

def compute_dice(pred, gt, num_classes):
    dices=[]
    for c in range(num_classes):
        inter = 2*((pred==c)&(gt==c)).sum()
        denom = (pred==c).sum() + (gt==c).sum()
        dices.append(inter/denom if denom>0 else np.nan)
    return np.array(dices), np.nanmean(dices)

def split_dataset(img_dir, test_size=0.2, val_size=0.1, seed=42):
    files=[f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in EXTS]
    tv, te = train_test_split(files, test_size=test_size, random_state=seed)
    tr, va = train_test_split(tv, test_size=val_size/(1-test_size), random_state=seed)
    return tr, va, te

print("\n[METHOD] Pour valider : split_dataset → prédictions + GT → compute_iou / compute_dice")

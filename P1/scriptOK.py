#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segment_all_images_with_legend.py

Segmentation sémantique des vêtements (mattmdjaga/segformer_b2_clothes) pour 
toutes les images dans top_influenceurs_2024/IMG. 
Génère à la fois :
  - un masque grayscale (_mask.png)
  - une image colorée (_mask_color.png) + sa légende embarquée (_mask_color_legend.png)

Dépendances :
    pip install torch transformers pillow numpy
"""

import os
import sys

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

# 1) Initialisation du modèle
MODEL_ID = "mattmdjaga/segformer_b2_clothes"
processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
model     = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Récupération du mapping id → label
id2label = model.config.id2label

# Palette de couleurs
PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0,   0), (0,   128, 0), (0,   0, 128),
    (128, 128, 0), (128, 0, 128), (0,   128, 128),
]
def get_color(label_id: int):
    return PALETTE[label_id % len(PALETTE)]

# 2) Dossiers
BASE_DIR  = os.path.dirname(__file__)
IMG_DIR   = os.path.join(BASE_DIR, "top_influenceurs_2024", "IMG")
MASKS_DIR = os.path.join(BASE_DIR, "masks")
os.makedirs(MASKS_DIR, exist_ok=True)

if not os.path.isdir(IMG_DIR):
    print(f"[ERROR] Dossier introuvable : {IMG_DIR}")
    sys.exit(1)

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# Police par défaut pour la légende
FONT = ImageFont.load_default()

for fname in sorted(os.listdir(IMG_DIR)):
    name, ext = os.path.splitext(fname.lower())
    if ext not in EXTS:
        continue

    img_path = os.path.join(IMG_DIR, fname)
    print(f"[INFO] Traitement de {fname}")

    # a) Charger et pré-traiter
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # b) Inférence
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    up = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    pred = up.argmax(dim=1)[0].cpu().numpy()  # (H, W)

    # c) Masque grayscale
    max_lbl = pred.max() if pred.max() > 0 else 1
    gray = (pred.astype(np.float32) / max_lbl * 255).astype(np.uint8)
    mask_gray = Image.fromarray(gray, mode="L")
    mask_gray.save(os.path.join(MASKS_DIR, f"{name}_mask.png"))

    # d) Masque colorisé
    h, w = pred.shape
    color_arr = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl, label_name in id2label.items():
        lbl = int(lbl)
        color_arr[pred == lbl] = get_color(lbl)
    mask_color = Image.fromarray(color_arr, mode="RGB")
    mask_color.save(os.path.join(MASKS_DIR, f"{name}_mask_color.png"))

    # e) Création de la légende
    entries = list(id2label.items())
    # On crée d'abord un canevas de type draw pour mesurer la taille de texte
    legend_w = 200
    # Mesurer la hauteur d'une ligne ("Hg" est une bonne référence)
    dummy_img = Image.new("RGB", (1,1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0,0), "Hg", font=FONT)
    line_h = (bbox[3] - bbox[1]) + 4  # hauteur du texte + marge
    legend_h = line_h * len(entries) + 10

    legend = Image.new("RGB", (legend_w, legend_h), "white")
    draw = ImageDraw.Draw(legend)

    y = 5
    for lbl, label_name in entries:
        lbl = int(lbl)
        col = get_color(lbl)
        # petit carré de couleur
        draw.rectangle([5, y, 5+line_h-4, y+line_h-4], fill=col)
        # texte
        draw.text((5+line_h, y), label_name, font=FONT, fill="black")
        y += line_h

    # f) Concaténation masque color + légende
    total_w = mask_color.width + legend.width
    total_h = max(mask_color.height, legend.height)
    combined = Image.new("RGB", (total_w, total_h), "white")
    combined.paste(mask_color, (0, 0))
    combined.paste(legend, (mask_color.width, 0))
    combined.save(os.path.join(MASKS_DIR, f"{name}_mask_color_legend.png"))

print("[DONE] Tous les masques et légendes sont dans ‘masks/’")

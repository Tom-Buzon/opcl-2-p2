#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from PIL import Image
import numpy as np
import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

# 1) Chargement du modèle & du processor
MODEL_ID = "mattmdjaga/segformer_b2_clothes"
processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
model     = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID)

# 2) Trouver la 1ʳᵉ image
base_dir = os.path.dirname(__file__)
img_dir  = os.path.join(base_dir, "top_influenceurs_2024", "IMG")
if not os.path.isdir(img_dir):
    print("❌ Dossier introuvable :", img_dir)
    sys.exit(1)

exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
img_path = None
for fn in sorted(os.listdir(img_dir)):
    if os.path.splitext(fn)[1].lower() in exts:
        img_path = os.path.join(img_dir, fn)
        break
if not img_path:
    print("❌ Aucune image valide trouvée.")
    sys.exit(1)

print("🔍 Segmentation en local pour :", img_path)

# 3) Préparation et inférence
image = Image.open(img_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits  # (batch=1, n_labels, H//P, W//P)
# on remonte à la taille d'origine
upsampled = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False
)
pred = upsampled.argmax(dim=1)[0].cpu().numpy()  # (H, W), valeurs [0..n_labels-1]

# 4) Conversion en image (niveaux de gris)
# On normalise pour couvrir 0→255
norm = (pred.astype(np.float32) / (pred.max() or 1) * 255).astype(np.uint8)
mask_img = Image.fromarray(norm, mode="L")

# 5) Sauvegarde
masks_dir = os.path.join(base_dir, "masks")
os.makedirs(masks_dir, exist_ok=True)
out_path = os.path.join(masks_dir, "mask_01.png")
mask_img.save(out_path)

print(f"✅ Masque sauvegardé dans : {out_path}")

"""
Build a reproducible 20k paired subset from TAMPCOCO and flatten outputs.

Usage note:
- Put this script in the TAMPCOCO root directory (the same directory that
  contains `cm_images/`, `cm_masks/`, `sp_images/`, and `sp_masks/`), then run it.
- The script writes outputs to `tp/` and `tp/mask/` in that same directory.
"""

import os
import random
import shutil
from tqdm import tqdm

# -----------------------------
# SETTINGS
# -----------------------------
RANDOM_SEED = 42
TOTAL_IMAGES = 20000

image_folders = {
    "cm": "cm_images",
    "sp": "sp_images"
}

mask_folders = {
    "cm": "cm_masks",
    "sp": "sp_masks"
}

output_image_folder = "tp"
output_mask_folder = os.path.join("tp", "mask")

random.seed(RANDOM_SEED)


def build_match_key(filename):
    """Create a format-agnostic key so image and mask names align reliably."""
    stem = os.path.splitext(filename)[0].lower()
    for ext_token in (".jpg", ".jpeg", ".png"):
        stem = stem.replace(ext_token, "")
    return stem

# -----------------------------
# CREATE OUTPUT FOLDERS
# -----------------------------
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# -----------------------------
# DETERMINE SAMPLE SIZE
# -----------------------------
num_domains = len(image_folders)
samples_per_domain = TOTAL_IMAGES // num_domains

print(f"Sampling {samples_per_domain} images per manipulation type")

selected_pairs = []

# -----------------------------
# PROCESS EACH DOMAIN
# -----------------------------
for domain in image_folders:

    img_folder = image_folders[domain]
    mask_folder = mask_folders[domain]

    print(f"\nProcessing {domain}")

    images = os.listdir(img_folder)
    masks = os.listdir(mask_folder)

    # Build mask lookup using normalized full-name keys.
    mask_lookup = {}

    for m in masks:
        key = build_match_key(m)
        mask_lookup.setdefault(key, []).append(m)

    random.shuffle(images)

    count = 0

    for img in images:

        key = build_match_key(img)

        if key not in mask_lookup:
            continue

        img_path = os.path.join(img_folder, img)

        mask_name = mask_lookup[key][0]
        mask_path = os.path.join(mask_folder, mask_name)

        if os.path.isfile(img_path) and os.path.isfile(mask_path):

            selected_pairs.append((img_path, mask_path))
            count += 1

        if count >= samples_per_domain:
            break

    print(f"Selected {count} pairs")

# -----------------------------
# SHUFFLE FINAL DATASET
# -----------------------------
random.shuffle(selected_pairs)

print(f"\nTotal selected pairs: {len(selected_pairs)}")

# -----------------------------
# COPY FILES
# -----------------------------
for idx, (img_path, mask_path) in enumerate(tqdm(selected_pairs, desc="Copying files")):

    ext = os.path.splitext(img_path)[1]

    img_out = f"{idx+1:06d}{ext}"
    mask_out = f"{idx+1:06d}.png"

    shutil.copy(img_path, os.path.join(output_image_folder, img_out))
    shutil.copy(mask_path, os.path.join(output_mask_folder, mask_out))

print("\nDone!")
print(f"Images saved to: {output_image_folder}")
print(f"Masks saved to: {output_mask_folder}")
print(f"Total pairs: {len(selected_pairs)}")


# STEP 1: Mount Google Drive + BraTS20 FIX
from google.colab import drive
drive.mount('/content/drive')

# CORRECT PATH
BRAts_PATH = '/content/drive/MyDrive/BraTS2020/training_data'  # Keep same path
print(f"🔍 Checking BraTS20 data at: {BRAts_PATH}")

# Install packages
!pip install nibabel tqdm -q

# ✅ FIXED: Look for BraTS20_Training_XXX folders
import os
cases = [d for d in os.listdir(BRAts_PATH) if d.startswith('BraTS20_Training_')]
print(f"✅ Found {len(cases)} BraTS20 training cases: {cases[:3]}...")

# Test first case loads correctly
import nibabel as nib
test_case = os.path.join(BRAts_PATH, cases[0])
print(f"\n🔬 Testing: {cases[0]}")

# Find FLAIR and SEG files (case insensitive)
flair_file = None
seg_file = None
for file in os.listdir(test_case):
    if 'flair' in file.lower():
        flair_file = os.path.join(test_case, file)
    if 'seg' in file.lower():
        seg_file = os.path.join(test_case, file)

print(f"FLAIR: {os.path.basename(flair_file) if flair_file else 'NOT FOUND'}")
print(f"SEG:   {os.path.basename(seg_file) if seg_file else 'NOT FOUND'}")

# Load test slice
flair = nib.load(flair_file).get_fdata()[:,:,78]  # Middle slice
seg = nib.load(seg_file).get_fdata()[:,:,78]
print(f"✅ SUCCESS: FLAIR shape {flair.shape}, SEG shape {seg.shape}")
print(f"🎉 STEP 1 COMPLETE - BraTS20 folders detected! RUN STEP 2!")

# STEP 2: Training U-Net on 30 BraTS20 Cases
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from tqdm import tqdm
import os

# U-Net Model (FLAIR → Tumor)
class FLAIRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.bottleneck = self.conv_block(128, 256)
        self.dec3 = self.conv_block(256+128, 128)
        self.dec2 = self.conv_block(128+64, 64)
        self.dec1 = self.conv_block(64+32, 32)
        self.final = nn.Conv2d(32, 1, 1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([F.interpolate(b, e3.shape[2:]), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, e2.shape[2:]), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, e1.shape[2:]), e1], 1))
        return torch.sigmoid(self.final(d1))

# Dataset
class BraTS20Dataset(Dataset):
    def __init__(self, root_dir):
        self.cases = [os.path.join(root_dir, f'BraTS20_Training_{i:03d}')
                     for i in range(1, 31) if os.path.exists(os.path.join(root_dir, f'BraTS20_Training_{i:03d}'))]
        self.root_dir = root_dir
        print(f"✅ Dataset: {len(self.cases)} BraTS20 cases")

    def __len__(self): return len(self.cases) * 20  # 20 slices per case

    def __getitem__(self, idx):
        case_idx, slice_idx = divmod(idx, 20)
        case_dir = os.path.join(self.root_dir, f'BraTS20_Training_{case_idx+1:03d}')

        # Find files
        flair_file = seg_file = None
        for file in os.listdir(case_dir):
            if 'flair' in file.lower(): flair_file = os.path.join(case_dir, file)
            if 'seg' in file.lower(): seg_file = os.path.join(case_dir, file)

        # Load slice 70+slice_idx (middle brain region)
        flair_img = nib.load(flair_file).get_fdata()[:,:,70+slice_idx]
        seg_img = nib.load(seg_file).get_fdata()[:,:,70+slice_idx]

        # Normalize FLAIR, binarize SEG
        flair_norm = np.clip((flair_img - flair_img.mean()) / (flair_img.std() + 1e-8), -2, 2)
        seg_bin = (seg_img > 0).astype(np.float32)

        return torch.FloatTensor(flair_norm[None, :, :]), torch.FloatTensor(seg_bin[None, :, :])

# START TRAINING
print("🚀 Starting U-Net training on YOUR 30 BraTS20 cases...")
BRAts_PATH = '/content/drive/MyDrive/BraTS2020/training_data'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

dataset = BraTS20Dataset(BRAts_PATH)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

model = FLAIRUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

model.train()
for epoch in range(10):
    total_loss = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/10')
    for batch_idx, (flair, seg) in enumerate(pbar):
        flair, seg = flair.to(device), seg.to(device)
        optimizer.zero_grad()
        pred = model(flair)
        loss = criterion(pred, seg)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(loader)
    print(f"✅ Epoch {epoch+1}/10 completed, Average Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), f'/content/drive/MyDrive/brats20_unet_epoch_{epoch+1}.pth')

torch.save(model.state_dict(), '/content/drive/MyDrive/brats20_unet_final.pth')
print("\n🎉 STEP 2 COMPLETE!")
print("✅ Model saved: /content/drive/MyDrive/brats20_unet_final.pth")
print("🚀 NOW RUN STEP 3 for RED tumor + Grade prediction!")

import matplotlib.pyplot as plt
import numpy as np
import torch
import nibabel as nib
import cv2
from skimage import exposure, segmentation

class FLAIRUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_block(1, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.bottleneck = self.conv_block(128, 256)
        self.dec3 = self.conv_block(256+128, 128)
        self.dec2 = self.conv_block(128+64, 64)
        self.dec1 = self.conv_block(64+32, 32)
        self.final = torch.nn.Conv2d(32, 1, 1)

    def conv_block(self, in_ch, out_ch):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1), torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(torch.nn.functional.max_pool2d(e1, 2))
        e3 = self.enc3(torch.nn.functional.max_pool2d(e2, 2))
        b = self.bottleneck(torch.nn.functional.max_pool2d(e3, 2))
        d3 = self.dec3(torch.cat([torch.nn.functional.interpolate(b, e3.shape[2:]), e3], 1))
        d2 = self.dec2(torch.cat([torch.nn.functional.interpolate(d3, e2.shape[2:]), e2], 1))
        d1 = self.dec1(torch.cat([torch.nn.functional.interpolate(d2, e1.shape[2:]), e1], 1))
        return torch.nn.functional.sigmoid(self.final(d1))

def preprocessing_pipeline(img):
    print("Preprocessing started:")
    img8 = (img / np.max(img) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img8)

    img_smooth = cv2.GaussianBlur(img_clahe, (3,3), 0)

    # GENTLE skull stripping - higher threshold
    _, img_mask = cv2.threshold(img_smooth, 50, 255, cv2.THRESH_BINARY)  # Increased from 30 to 50
    kernel = np.ones((3,3), np.uint8)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)
    img_skull = cv2.bitwise_and(img_smooth, img_mask)

    img_eq = exposure.equalize_hist(img_skull.astype(np.float32)/255.0)

    img_norm = (img_eq - np.mean(img_eq)) / (np.std(img_eq) + 1e-8)
    img_norm = np.clip(img_norm, -3, 3)
    print("  ✓ Pre-processing Completed")

    return img_norm

def postprocessing_pipeline(mask, original_img):
    """FIXED: Higher threshold + stricter filtering for tumor only"""
    print("Postprocessing started:")

    # 1. HIGHER threshold to reduce false positives
    mask_bin = (mask > 0.6).astype(np.uint8)  # Increased from 0.4 to 0.6

    # 2. Larger morphological operations
    kernel = np.ones((3,3), np.uint8)  # Larger kernel
    mask_open = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)

    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    # 3. STRICT largest component only (remove small false positives)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_close, 8)
    if num_labels > 1:
        # Only keep components > 100 pixels (eliminates small false positives)
        large_components = np.where(stats[1:, cv2.CC_STAT_AREA] > 100)[0] + 1
        if len(large_components) > 0:
            largest_label = large_components[np.argmax(stats[large_components, cv2.CC_STAT_AREA])]
            mask_clean = (labels == largest_label).astype(np.uint8)
        else:
            mask_clean = np.zeros_like(mask_close)
    else:
        mask_clean = mask_close


    # 🔥 4. ACTIVE CONTOUR (only on main tumor)
    contours, _ = cv2.findContours(mask_clean*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours and cv2.contourArea(contours[0]) > 100:
        largest_contour = max(contours, key=cv2.contourArea)
        snake_points = largest_contour.squeeze().astype(np.float32)
        snake_norm = snake_points / np.array(original_img.shape[::-1])

        try:
            snake = segmentation.active_contour(original_img.astype(float), snake_norm,
                                             alpha=0.015, beta=10, gamma=0.001, max_iterations=200)
            snake_pixels = (snake * np.array(original_img.shape[::-1])).astype(np.int32)
            mask_final = np.zeros_like(mask_clean, dtype=np.uint8)
            mask_final = cv2.fillPoly(mask_final, [snake_pixels], 1)
        except:
            mask_final = mask_clean
    else:
        mask_final = mask_clean

    print("  ✓ Pre-processing Completed")
    return mask_final.astype(np.float32)

# Main pipeline
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FLAIRUNet().to(device)
model.load_state_dict(torch.load('/content/drive/MyDrive/brats20_unet_final.pth', map_location=device))
model.eval()

flair_path = '/content/drive/MyDrive/BraTS2020/validation_data/BraTS20_031/flair.nii'
flair_img = nib.load(flair_path).get_fdata()
original = flair_img[:,:,78]

processed = preprocessing_pipeline(original)
input_tensor = torch.FloatTensor(processed[None, None, :, :]).to(device)

with torch.no_grad():
    raw_mask = model(input_tensor)[0, 0].cpu().numpy()

final_mask = postprocessing_pipeline(raw_mask, original)

# Grade prediction - CORRECTED for small tumors
tumor_pixels = np.sum(final_mask > 0)
total_pixels = final_mask.size
tumor_ratio = tumor_pixels / total_pixels

if tumor_pixels == 0:
    print("✅ NO TUMOR DETECTED")
    grade = "No Tumor"
elif tumor_ratio < 0.015:  # <1.5% = Grade 2 (small tumors)
    grade = "Grade 2 (Low-grade)"
elif tumor_ratio < 0.04:   # 1.5-4% = Grade 3 (medium)
    grade = "Grade 3 (High-grade)"
else:                      # >4% = Grade 4 (large)
    grade = "Grade 4 (Glioblastoma)"

print(f"🎯 RESULT: Tumor ratio {tumor_ratio*100:.2f}% → {grade}")


# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

ax1.imshow(original, cmap='gray', vmin=0, vmax=600)
ax1.set_title('Input FLAIR MRI Slice', fontsize=16, fontweight='bold', pad=20)
ax1.axis('off')

ax2.imshow(original, cmap='gray', vmin=0, vmax=600)
ax2.contourf(final_mask, levels=[0.5, 1], colors='red', alpha=0.4)
ax2.set_title(f'Active Contour Refined Tumor Overlay\nPredicted Tumor Grade: {grade}',
              fontsize=16, fontweight='bold', pad=20)
ax2.axis('off')

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/tumor_only_result.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

import torch
import cv2
import os
import numpy as np
import albumentations as A
from PIL import Image

from utils import get_segment_labels, draw_segmentation_map
from config import ALL_CLASSES
from model import ImprovedSegModel

# -----------------------------
# Paths
# -----------------------------
input_dir = 'Test_images'

out_dir = 'Pridicted_Patterns'

os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# Device
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------
# Model
# -----------------------------
model = ImprovedSegModel(num_classes=len(ALL_CLASSES)).to(device)

ckpt = torch.load('/outputsResnet34/best_valid_epoch_dice.pth', map_location=device)

model.load_state_dict(ckpt['model_state_dict'])

model.eval()

# -----------------------------
# SAME normalization as training
# -----------------------------
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(
        mean=[0.45734706, 0.43338275, 0.40058118],
        std=[0.23965294, 0.23532275, 0.2398498],
        max_pixel_value=255.0
    )
])

# -----------------------------
# Inference
# -----------------------------
all_image_paths = sorted(os.listdir(input_dir))

for i, image_name in enumerate(all_image_paths):

    print(f"Processing Image {i+1}: {image_name}")

    image_path = os.path.join(input_dir, image_name)

    image = np.array(Image.open(image_path).convert("RGB"))

    transformed = transform(image=image)
    image = transformed["image"]

    image = np.transpose(image, (2,0,1))
    image = torch.from_numpy(image).float().unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(image)

    pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
    
    segmented_image = (pred * 255).astype(np.uint8)
    
    segmented_image = np.stack([segmented_image]*3, axis=-1)

    # cv2.imshow("Segmented image", segmented_image)
    # cv2.waitKey(1)

    cv2.imwrite(os.path.join(out_dir, image_name), segmented_image)

print("Inference complete! Results saved in:", out_dir)
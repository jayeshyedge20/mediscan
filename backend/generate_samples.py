"""
Sample X-Ray Generator and Manager for MediScan
Organizes high quality medical radiographs into sample_xrays/ and frontend/sample_xrays/
"""

import os
import shutil
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_xrays")
FRONTEND_SAMPLES = os.path.join(BASE_DIR, "frontend", "sample_xrays")

os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(FRONTEND_SAMPLES, exist_ok=True)

def setup_samples():
    img_dir = os.path.join(BASE_DIR, "frontend", "images")

    # 1. Wrist Fracture
    shutil.copy(os.path.join(img_dir, "wrist-xray.jpg"), os.path.join(SAMPLES_DIR, "wrist_fracture.jpg"))

    # 2. Fingers / Hand Fracture
    shutil.copy(os.path.join(img_dir, "wrist-xrayy.jpg"), os.path.join(SAMPLES_DIR, "fingers_fracture.jpg"))

    # 3. Forearm Fracture
    shutil.copy(os.path.join(img_dir, "wrist-xray.jpg"), os.path.join(SAMPLES_DIR, "forearm_fracture.jpg"))

    # 4. Elbow / Upper Extremity Fracture
    shutil.copy(os.path.join(img_dir, "ankle-xray.jpg"), os.path.join(SAMPLES_DIR, "elbow_fracture.jpg"))

    # Copy all to frontend/sample_xrays
    for f in os.listdir(SAMPLES_DIR):
        if f.endswith(".jpg") or f.endswith(".png"):
            shutil.copy(os.path.join(SAMPLES_DIR, f), os.path.join(FRONTEND_SAMPLES, f))

    print(f"Sample images successfully synced to {SAMPLES_DIR} & {FRONTEND_SAMPLES}")

if __name__ == "__main__":
    setup_samples()

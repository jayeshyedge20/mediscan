"""
Automated Test Suite for MediScan AI API Endpoints
"""

import os
import sys
from fastapi.testclient import TestClient

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import app

client = TestClient(app)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("✓ Health endpoint passed!")

def test_samples():
    response = client.get("/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) >= 8
    print(f"✓ Samples endpoint passed ({len(data['samples'])} samples available)!")

def test_fracture_detect():
    img_path = os.path.join(BASE_DIR, "sample_xrays", "wrist_fracture.jpg")
    with open(img_path, "rb") as f:
        response = client.post("/detect", files={"file": ("wrist_fracture.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "fracture"
    assert "result_image" in data
    assert "gradcam_image" in data
    assert "clinical_summary" in data
    print(f"✓ Fracture detection passed! Detections found: {data['detections_count']}")

def test_disease_detect():
    img_path = os.path.join(BASE_DIR, "sample_xrays", "osteoarthritis_knee.jpg")
    with open(img_path, "rb") as f:
        response = client.post("/detect-disease", files={"file": ("osteoarthritis_knee.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "disease"
    assert "report" in data
    assert "disease_heatmap" in data
    print(f"✓ Bone Disease detection passed! Primary: {data['report']['primary_disease']} ({data['report']['confidence']}%)")

def test_comprehensive_scan():
    img_path = os.path.join(BASE_DIR, "sample_xrays", "wrist_fracture.jpg")
    with open(img_path, "rb") as f:
        response = client.post("/comprehensive-scan", files={"file": ("wrist_fracture.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "comprehensive"
    assert "fracture_analysis" in data
    assert "disease_analysis" in data
    assert "master_triage" in data
    print(f"✓ Comprehensive Scan passed! Master Triage: {data['master_triage']}")

if __name__ == "__main__":
    print("Running MediScan Automated API Test Suite...")
    test_health()
    test_samples()
    test_fracture_detect()
    test_disease_detect()
    test_comprehensive_scan()
    print("\n All API tests passed with 100% success!")

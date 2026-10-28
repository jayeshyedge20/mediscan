# CLAUDE.md — MediScan AI

Guidance for Claude Code (and other AI agents) working in this repository. Read this first. It describes what the project actually does today, including its limitations, so changes stay honest and consistent.

> **Clinical disclaimer.** MediScan is a research / demonstration radiographic analysis suite. Its outputs (fracture boxes, disease labels, KL grades, T-score estimates, triage levels, recommendations) are **not clinically validated** and must not be presented as medical diagnoses. Any change that increases the apparent clinical authority of an output must be flagged to the user. See [Known limitations](#known-limitations--technical-debt).

---

## 1. Project overview

**MediScan AI: Bone Fracture & Bone Disease Diagnostic Suite** — a FastAPI backend that serves a static HTML/JS frontend and exposes three analysis modes over X-ray images:

| Mode | Endpoint | Engine | Output |
|---|---|---|---|
| Fracture detection | `POST /detect` | YOLOv8 (`backend/models/model.pt`) + heuristic "Grad-CAM-style" overlay + rule-based clinical summary | Bounding boxes, confidences, annotated image, attention overlay, triage summary |
| Bone disease analysis | `POST /detect-disease` | `disease_analyzer.py`: handcrafted radiographic biomarkers → hand-tuned scores → softmax over 5 classes | Primary class, probabilities, severity, KL-style grade, BMD status, findings, recommendations, heatmap |
| Comprehensive scan | `POST /comprehensive-scan` | Runs both of the above on one image | Combined payload + `master_triage` |

Plus `GET /health`, `GET /status` (alias), `GET /samples` (sample gallery manifest), and Swagger UI at `/docs`.

Copyright: "MediScan AI Team" (see README footer). License: MIT (`LICENSE`).

---

## 2. Repository layout (annotated)

```
MediScan/
├── run.py                      # 1-click launcher: builds sample dirs, opens browser, starts uvicorn on 0.0.0.0:8000
├── requirements.txt            # Python deps (see §4.2 for the unused ones)
├── README.md                   # Public-facing overview & endpoint table
├── LICENSE                     # MIT
├── CLAUDE.md                   # This file
├── backend/
│   ├── app.py                  # FastAPI app: routes, fracture logic, Grad-CAM-style overlay, static mounts
│   ├── disease_analyzer.py     # BoneDiseaseEngine: biomarkers, scoring, grading, disease heatmaps
│   ├── generate_samples.py     # setup_samples(): copies frontend/images → sample_xrays/ and frontend/sample_xrays/
│   ├── test_api.py             # Endpoint tests (pytest-style functions; also runnable as a script)
│   ├── models/model.pt         # YOLOv8 fracture weights (~22 MB, tracked in git)
│   ├── uploads/                # Every uploaded image, named <uuid><ext>  (runtime data)
│   └── results/                # Runtime outputs (see §6)
│       ├── <uuid>_result.jpg           # YOLO annotated image
│       ├── gradcam/<uuid>_gradcam.jpg  # fracture attention overlay
│       ├── disease_heatmaps/<uuid>_disease_heatmap.jpg
│       └── explanations/               # created at startup, currently unused by any route
├── frontend/                   # Static site, served by FastAPI at "/"
│   ├── index.html              # Landing page
│   ├── detect.html             # "Diagnostic Studio": mode tabs, upload, sample gallery, results
│   ├── disease.html            # Dedicated bone-disease page
│   ├── features.html, about.html, gradcam.html
│   ├── detect-script.js        # Controller for detect.html (API calls, rendering) — 457 lines
│   ├── nav-handler.js + nav-template.html      # Injects shared nav into #nav-placeholder via fetch()
│   ├── footer-handler.js + footer-template.html# Injects shared footer via fetch(); year + back-to-top
│   ├── footer-styles.css, styles.css           # Styling (styles.css ~1,390 lines)
│   ├── images/                 # Marketing/demo images + source X-rays used by generate_samples.py
│   └── sample_xrays/           # Web-served copies of the sample radiographs
├── sample_xrays/               # Canonical sample radiographs (served at /sample_xrays)
├── graphs/                     # Training/eval plots (confusion matrices, accuracy, model comparison)
└── models/yolo_model.ipynb     # Colab/Jupyter training notebook (YOLOv8 fine-tuning + evaluation)
```

Note: `backend/models/` lives *inside* `backend/`, while the training notebook lives in the top-level `models/`. Do not confuse them.

---

## 3. Runtime architecture

### 3.1 Startup sequence

1. `run.py` puts `backend/` on `sys.path`.
2. `run.py` calls `generate_samples.setup_samples()` (**overwrites** sample files — see §9). Failure is logged and ignored.
3. A daemon thread opens `http://localhost:8000/` after ~1.2 s.
4. `from app import app` runs **module-level side effects**, in this order:
   - Creates `uploads/`, `results/`, `results/explanations/`, `results/gradcam/`, `results/disease_heatmaps/`, `sample_xrays/`.
   - Loads YOLO weights from `backend/models/model.pt` (raises and aborts startup on failure).
   - Creates the FastAPI app and CORS middleware (`allow_origins=["*"]`).
   - Imports `disease_analyzer`, which instantiates `disease_engine = BoneDiseaseEngine()` at import time (seeds NumPy and Torch globally, builds a MobileNetV3-small on CPU/GPU).
5. `uvicorn.run(app, host="0.0.0.0", port=8000)`.

**Implication:** importing `app` (including in tests) loads a 22 MB model and the disease engine. Plan for that cost.

### 3.2 Request flow (`/detect`)

```
client ──multipart file──▶ detect_fractures()
  1. content_type must start with "image/"        → else 400
  2. write bytes to backend/uploads/<uuid><ext>   (ext taken from client filename, default .jpg)
  3. PIL.Image.open → np.array → fracture_model(image, conf=0.15)
  4. save results[0].plot() → results/<uuid>_result.jpg
  5. build detections[] + detection_boxes[]
  6. generate_gradcam_heatmap(image, boxes) → results/gradcam/<uuid>_gradcam.jpg
  7. generate_fracture_clinical_summary(detections)
  8. return JSON (§5.1)           any exception → 500 "Inference error: ..."
```

### 3.3 Request flow (`/detect-disease`)

```
client ──multipart file──▶ detect_bone_disease()
  1. content_type check (400)
  2. write upload to backend/uploads/<uuid><ext>
  3. analyze_bone_disease(path, original_filename)   # cv2.imread → gray/RGB → biomarkers → scores → softmax
  4. heatmap RGB → BGR → results/disease_heatmaps/<uuid>_disease_heatmap.jpg
  5. del report["heatmap_image"]  (raw ndarray must not reach JSON)
  6. return JSON (§5.2)           any exception → 500 "Disease analysis error: ..."
```

Note the **filename hint**: `analyze()` receives the *original* upload filename and uses substring matching on it to bias scores (§7.4). The on-disk name is a UUID and carries no signal.

### 3.4 Static mounts (order matters)

```
/results        → backend/results
/uploads        → backend/uploads
/sample_xrays   → sample_xrays/
/               → frontend/ (html=True)   ← must stay LAST, it captures everything else
```

If you add a new mount or API route, it must be registered **before** the `/` frontend mount, or it will be shadowed.

### 3.5 Frontend ↔ backend contract

- `detect-script.js` resolves `API_URL`: the page's own origin if served from `localhost` / `127.0.0.1`, otherwise **hardcoded `http://localhost:8000`**.
- Result image paths from the API are origin-relative (e.g. `/results/...`); the frontend prefixes `API_URL`.
- Sample gallery loads `GET /samples` and falls back to a hardcoded list if that fails. Clicking a sample fetches its `url` as a blob, wraps it in a `File` named after `sample.filename`, and shows it in the preview. It also **auto-switches the mode tab**: a `Bone Disease` sample moves a `fracture` tab to `disease`, and a `Fracture` sample moves a `disease` tab to `fracture`. The sample path therefore goes through the same upload endpoints as a user upload.
- Mode is chosen by tabs (`data-mode="fracture" | "disease" | "comprehensive"`) and maps to the three endpoints.
- Nav and footer are fetched as HTML fragments, so pages **must be served over HTTP** (opening `file://` breaks them).

---

## 4. Setup and running

### 4.1 Prerequisites

- Python 3 (no minimum version is pinned; the environment used when writing this file was Python 3.14 with no ML packages installed, so verify your own interpreter and package versions).
- Windows is the primary development platform (paths in the repo and notebook are Windows-style). Use forward slashes in code and docs; `os.path.join` everywhere in Python.
- CPU is sufficient. CUDA is used automatically by the disease engine if `torch.cuda.is_available()`.

### 4.2 Install

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `fastapi==0.103.1`, `uvicorn==0.23.2`, `python-multipart==0.0.6`; everything else is a floor (`>=`). Torch and torchvision should be installed from the index matching your CUDA/CPU setup if the default is wrong for your machine.

Packages listed but **not imported anywhere in the codebase**: `lime`, `scikit-image`, `python-jose`, `aiofiles`, `requests`, `matplotlib`. `uuid` (the PyPI package) is also unnecessary — the code uses the standard-library `uuid`. Do not remove them without checking, but do not add more dependencies without a clear need.

### 4.3 Run

```bash
python run.py                       # full app + browser, http://localhost:8000/ , API docs at /docs
```

Dev alternative (untested in this repo — verify before relying on it):

```bash
cd backend && uvicorn app:app --reload --port 8000
```

This skips `setup_samples()`, so `sample_xrays/` must already be populated.

### 4.4 Test

```bash
python backend/test_api.py                   # runs the four test functions as a script
python -m pytest backend/test_api.py -v      # via pytest
```

Requirements: `models/model.pt` present, `sample_xrays/wrist_fracture.jpg`, `osteoarthritis_knee.jpg` present, and all ML deps installed. The test file exercises the real model; there are **no mocks**. Expect a few seconds of model load.

**Status:** the test suite was not executed during the authoring of this file (dependencies were not installed in that environment). Run it before claiming any behavior is verified.

---

## 5. API reference

All endpoints accept `multipart/form-data` with a single field `file`. Error responses use FastAPI's default shape: `{"detail": "..."}`.

### 5.1 `POST /detect` — fracture detection

Success `200`:

```jsonc
{
  "detection_id": "9f1c…-uuid",
  "mode": "fracture",
  "message": "Bone fracture detection completed successfully",
  "result_image": "/results/<uuid>_result.jpg",
  "gradcam_image": "/results/gradcam/<uuid>_gradcam.jpg",
  "original_image": "/uploads/<uuid>.jpg",
  "detections_count": 1,
  "detections": [
    {
      "id": 1,
      "class": "wrist positive",          // raw YOLO class name from model.names
      "formatted_class": "Wrist Fracture",// " positive" → " fracture", then .title()
      "confidence": 87.3,                 // 0–100, one decimal
      "box": { "x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0 }   // pixel coords, one decimal
    }
  ],
  "clinical_summary": {
    "status": "Positive: Wrist Fracture",
    "severity": "Severe | Moderate | Normal",
    "triage": "Urgent / Orthopedic Consult | Moderate / Splinting | Routine",
    "findings": [ "..." ],
    "recommendations": [ "...", "..." ]
  }
}
```

Errors: `400` non-image content type; `500 "Inference error: …"`.

Confidence threshold for YOLO is **0.15** (deliberately high-sensitivity). Changing it changes recall/precision of the product — treat it as a product decision.

### 5.2 `POST /detect-disease` — bone disease analysis

Success `200`:

```jsonc
{
  "detection_id": "<uuid>",
  "mode": "disease",
  "message": "Bone disease analysis completed successfully",
  "original_image": "/uploads/<uuid>.jpg",
  "disease_heatmap": "/results/disease_heatmaps/<uuid>_disease_heatmap.jpg",
  "report": {
    "primary_disease": "Osteoarthritis (OA)",
    "confidence": 62.4,                  // top-class probability, 0–100
    "severity": "Normal | Mild | Moderate | Severe",
    "triage_level": "Routine | Moderate / Clinical Review | Urgent / DEXA Confirmatory | High Priority / Specialist Consult | Urgent",
    "kl_grade": "Grade 0 … | Grade 1 … | Grade 2 … | Grade 3-4 … | N/A - <category>",
    "bmd_status": "…",
    "biomarkers": {                      // formatted strings, not numbers
      "Bone Radiodensity Score": "41.2 / 100",
      "Joint Narrowing Index": "33.0%",
      "Trabecular Heterogeneity": "12.5%",
      "Cortical Gradient Index": "18.9%"
    },
    "findings": [ "..." ],
    "recommendations": [ "..." ],
    "probabilities": {                   // all 5 classes, sum ≈ 100
      "Normal / Healthy Bone": 10.0,
      "Osteoarthritis (OA)": 62.4,
      "Osteoporosis / Osteopenia": 12.0,
      "Bone Lesion / Tumor": 8.0,
      "Osteomyelitis (Infection)": 7.6
    }
  }
}
```

The `heatmap_image` ndarray is removed from the report before serialization — never return raw arrays.

Errors: `400` non-image; `500 "Disease analysis error: …"`.

### 5.3 `POST /comprehensive-scan` — both engines

Success `200`:

```jsonc
{
  "detection_id": "<uuid>",
  "mode": "comprehensive",
  "message": "Comprehensive radiographic scan completed successfully",
  "original_image": "/uploads/<uuid>.jpg",
  "fracture_result_image": "/results/<uuid>_result.jpg",
  "fracture_gradcam_image": "/results/gradcam/<uuid>_gradcam.jpg",
  "disease_heatmap_image": "/results/disease_heatmaps/<uuid>_disease_heatmap.jpg",
  "fracture_analysis": { "count": 0, "detections": [], "summary": { /* as §5.1 clinical_summary */ } },
  "disease_analysis": { /* the disease report FIELDS directly — NOT nested under "report" */ },
  "master_triage": "Urgent | Moderate"
}
```

Note the asymmetry: `/detect-disease` nests the report under `report`; `/comprehensive-scan` puts the same fields under `disease_analysis`. The frontend handles both. Preserve this unless you change the frontend in the same change.

`master_triage` is `"Urgent"` if fracture severity is `Severe` **or** disease severity is `Severe`/`Critical`; otherwise `"Moderate"`. `"Critical"` is never produced by the disease engine today.

### 5.4 `GET /samples`

Returns `{"samples": [{id, name, category, filename, url, description}]}`. Categories in use: `Fracture`, `Bone Disease`, `Normal Baseline`. The frontend's fallback list uses `Normal` (not `Normal Baseline`) and a badge mapping accepts both — keep both spellings working if you touch the badge logic.

### 5.5 `GET /health` and `GET /status`

Both return `{"status": "healthy", "message": "...", "models": {...}}`. These are **liveness only** — they do not verify that the model is loaded (it is loaded at import, so if the process is up, it is).

### 5.6 Swagger

`/docs` is generated automatically from the FastAPI route signatures. The endpoint `response_model`s are not declared, so the docs show generic bodies. Adding Pydantic response models is a reasonable improvement but is not present today.

---

## 6. Fracture pipeline details

### 6.1 Model

- Weights: `backend/models/model.pt`, loaded with `ultralytics.YOLO`. Health text calls it "YOLOv8 Small".
- Training: `models/yolo_model.ipynb` fine-tunes `yolov8s.pt` for 100 epochs, `imgsz=416`, `batch=16`, `amp=True`. The dataset YAML lists 7 classes: `elbow positive`, `fingers positive`, `forearm fracture`, `humerus`, `humerus fracture`, `shoulder fracture`, `wrist positive`.
- **Class names are not verified in this file.** The authoritative list is `YOLO("backend/models/model.pt").names`. Check it before writing documentation or UI copy that names classes. The notebook's YAML is evidence, not proof, of what the shipped weights contain.

### 6.2 Class formatting

`formatted_class = cls_name.replace(" positive", " fracture").title()`. Example: `"wrist positive"` → `"Wrist Fracture"`. Any class name without `" positive"` is simply title-cased (`"humerus"` → `"Humerus"`).

### 6.3 Clinical summary rules (`generate_fracture_clinical_summary`)

| Condition | status | severity | triage |
|---|---|---|---|
| No detections | `Negative for Acute Fracture` | `Normal` | `Routine` |
| `max_conf > 60%` **or** more than one detection | `Positive: <classes>` | `Severe` | `Urgent / Orthopedic Consult` |
| otherwise (≥1 detection, max ≤ 60%, exactly one) | `Positive: <classes>` | `Moderate` | `Moderate / Splinting` |

**Two unit bugs in this function (see §15):**

1. `max_conf` is taken from `d["confidence"]`, which is already on a **0–100** scale (§5.1). The severity test `max_conf > 0.60` assumes a 0–1 scale. Because YOLO runs at `conf=0.15` (i.e. ≥ 15 on this scale), **every detection exceeds 0.60 and is labelled `Severe`.** The "Moderate" branch is effectively unreachable for single detections.
2. The findings line prints `f"{max_conf*100:.1f}%"`, which multiplies an already-percent value by 100. A 87.3% detection is shown to users as **"Maximum AI localization confidence: 8730.0%"**.

Fix both together, and update `test_api.py` if you add a regression check.

### 6.4 Image handling

- Input is read with `PIL.Image.open` then `np.array`. Grayscale, RGB and RGBA inputs all reach YOLO as whatever channel count PIL produces. Do not assume 3 channels.
- Annotated output uses `results[0].plot()` (Ultralytics' own drawing), saved with `cv2.imwrite`.

---

## 7. Disease analyzer details (`backend/disease_analyzer.py`)

### 7.1 What actually decides the class

**The MobileNetV3-small `feature_net` is constructed, given a custom 5-class head, put in eval mode, and then never called.** `analyze()` does not run a forward pass. The class decision comes entirely from the handcrafted scores in §7.3 plus the filename hint in §7.4.

Consequences you must respect:
- The `feature_net` weights are random (seeded with 42). Do not describe it as a trained or learned model.
- Any real learned classifier would need training data, labels, and validation that the repo does not contain.
- Changing `feature_net` alone changes nothing about predictions.

### 7.2 Preprocessing and bone mask

- Input: path → `cv2.imread` (BGR), converted to gray and RGB. Arrays are also accepted.
- `segment_bone_tissue`: Gaussian blur 5×5 → binary threshold at **30** (of 255) → morphological close with a 7×7 ellipse. Mask = pixels > 0.
- If the mask has fewer than 100 pixels, the **whole image** is used as the bone mask.
- `self.transform` (Resize 224, ImageNet normalize) is defined but **unused**.

### 7.3 Biomarkers and scores

All intensities normalized to [0, 1]. Biomarkers are computed inside the bone mask unless noted.

| Biomarker | Definition | Exposed in report as |
|---|---|---|
| `mean_density` | mean intensity of bone pixels | Bone Radiodensity Score (×100) |
| `std_density` | std of bone-pixel intensity | not exposed |
| `sclerosis_ratio` | fraction of bone pixels > 0.82 | used in findings text |
| `washout_ratio` | fraction of bone pixels < 0.35 | used in osteoporosis findings |
| `cortical_gradient` | mean Sobel gradient magnitude in mask | Cortical Gradient Index (×100) |
| `texture_entropy` | mean of local std (13×13 window) in mask — **this is local standard deviation, not Shannon entropy** | Trabecular Heterogeneity (×100) |
| `narrowing_index` | `1 − min(row_density) / max(row_density)` over rows with density > 0.15; fallback `0.2` if fewer than 11 such rows | Joint Narrowing Index (×100) |
| `focal_contrast` | p95 − p5 of bone intensity | not exposed |
| `has_focal_pocket` | fraction of bone pixels in (0.05, 0.25) | not exposed |

Scores (raw, before softmax):

```
score_normal  = 3.5·max(0, mean_density − 0.40) + 3.0·max(0, 0.30 − washout) + 2.5·max(0, 0.08 − focal_pocket)
score_oa      = 4.0·sclerosis + 2.5·narrowing + 2.0·max(0, mean_density − 0.50)
score_osteo   = 4.5·washout + 4.0·max(0, 0.50 − mean_density) + 2.0·max(0, 0.15 − cortical_gradient)
score_lesion  = 6.0·focal_pocket + 2.5·focal_contrast + 2.0·texture_entropy
score_inf     = 2.5·cortical_gradient + 2.5·texture_entropy + 1.5·washout
```

Then `raw = nan_to_num(raw, nan=0.1, posinf=5.0, neginf=0.0)`, and `probs = softmax((raw − max) · 2.0)`. The top class is `argmax(probs)`; `confidence = probs[top]`.

These coefficients are **hand-tuned** and not fitted to any labeled dataset.

### 7.4 Filename hints (strong, and unrelated to image content)

After scoring, the **original upload filename** (lowercased) is matched by substring, in this order (first match wins):

| Substring | Effect |
|---|---|
| `osteoarthritis` or `oa` | `+2.5` to OA |
| `osteoporosis` or `osteo` | `+2.5` to osteoporosis |
| `lesion` or `tumor` | `+2.5` to lesion |
| `normal` | `+2.5` to normal |
| `fracture` | `+1.0` to normal |

Because `"oa"` is a bare substring, it matches many unrelated filenames. The sample gallery relies on this hint (`osteoarthritis_knee.jpg` → OA). **Consequence:** the disease output for a user-uploaded X-ray depends on what the user named their file. Never describe disease results as image-derived without this caveat. Do not extend the hint system further; if you must touch it, make it explicit and documented.

### 7.5 Clinical text generation

Per primary class, the engine builds `severity`, `kl_grade`, `bmd_status`, `findings`, `recommendations`, `triage_level`:

- **Normal:** severity `Normal`; triage `Routine`.
- **OA:** `Severe` if `confidence > 60%` or narrowing > 50%; `Moderate` if confidence > 40%; else `Mild`. KL grade string follows the same bands.
- **Osteoporosis:** `Severe` if mean density < 0.42 (T-score text `< −2.5`), else `Moderate` (osteopenia text). The T-score wording is **an estimate from image intensity, not a DEXA value**.
- **Lesion:** `Severe` if confidence > 55%, else `Moderate`.
- **Osteomyelitis:** always `Severe`, triage `Urgent`.

Recommendations are generic clinical-style text (e.g. "Dual-energy X-ray Absorptiometry (DEXA) scan…", "MRI with IV contrast…"). They are produced by fixed templates, not by reasoning over the image. Keep that in mind before editing them.

### 7.6 Disease heatmaps

`_generate_disease_heatmap` builds a fixed spatial prior per class, normalizes to 0–1, applies JET, and alpha-blends at 0.45:

| Class | Heatmap logic | Honest description |
|---|---|---|
| OA | Horizontal band at 48% height × gray intensity | "joint-line region" prior |
| Osteoporosis | Blurred inverted intensity within bone mask | diffuse-density map |
| Lesion | **Fixed Gaussian at 48% w / 48% h** | a fixed central hotspot — **not localized to any detected lesion** |
| Osteomyelitis | Blurred Canny edges | edge map |
| Normal | Flat 0.15 field | uniform calm overlay |

Do not describe any of these as "localizing pathology" unless you have implemented localization.

### 7.7 Global side effects

`BoneDiseaseEngine.__init__` calls `np.random.seed(42)` and `torch.manual_seed(42)`. This changes global RNG state for the whole process. Keep this in mind if you add any randomized code elsewhere.

---

## 8. Explainability visuals — accuracy note

`app.py::generate_gradcam_heatmap` **is not Grad-CAM.** It does not use model gradients or activations. It places an isotropic Gaussian at the center of each YOLO box (σ ≈ box size / 4, minimum 15 px), takes the per-pixel max, applies JET at alpha 0.42, and draws the box outlines in green.

Correct wording: "attention-style overlay centered on detected boxes." Do **not** call it Grad-CAM, saliency, or "explainable AI evidence" in new copy. If you want real Grad-CAM, it needs a hook on a specific YOLO layer and is a separate piece of work.

The frontend page `gradcam.html` may still use the Grad-CAM name in its copy. Check before editing it.

---

## 9. Sample data

- Canonical files: `sample_xrays/*.jpg` (8 images). Served at `/sample_xrays/<file>`.
- `frontend/sample_xrays/` is a copy kept in sync by `generate_samples.setup_samples()`.
- **`setup_samples()` has hard-coded source mappings that do not match the sample names:**
  - `fingers_fracture.jpg` ← `frontend/images/wrist-xrayy.jpg`
  - `forearm_fracture.jpg` ← `frontend/images/wrist-xray.jpg` (same source as wrist)
  - `elbow_fracture.jpg` ← `frontend/images/ankle-xray.jpg` (an **ankle** image is labelled as elbow)
  - The four disease/normal samples are not regenerated by this function.
- `setup_samples()` **overwrites** existing sample files every launch via `run.py`. Editing a file in `sample_xrays/` by hand will be lost on the next start.
- Before changing sample names or sources, confirm with the user which images are the real intended ones. Do not silently "fix" the mapping.

The README claims the samples are "curated" and gives clinical descriptions (e.g. "Distal radius fracture with carpal displacement"). Those descriptions are marketing copy, not verified labels. Do not extend them.

---

## 10. Data files and lifecycle

| Path | Written by | Read by | Lifecycle |
|---|---|---|---|
| `backend/uploads/<uuid><ext>` | every analysis route | `/uploads` static mount, result JSON | **Never deleted.** 27 files at time of writing. |
| `backend/results/<uuid>_result.jpg` | `/detect`, `/comprehensive-scan` | `/results` mount | Never deleted |
| `backend/results/gradcam/<uuid>_gradcam.jpg` | same | same | Never deleted |
| `backend/results/disease_heatmaps/<uuid>_disease_heatmap.jpg` | `/detect-disease`, `/comprehensive-scan` | same | Never deleted |
| `backend/results/explanations/` | created at startup | nothing | Empty, unused |

Consequences:
- Uploaded patient images accumulate on disk indefinitely. That is a privacy and retention concern. Do not add or expose any "list all uploads" endpoint without raising it.
- Several sample-derived results are **tracked in git** (`backend/uploads/8bb0e6a4…`, `backend/results/…`). Most generated artifacts are **untracked**. There is **no `.gitignore`** in the repo.
- Failed analyses leave the upload behind (the file is written before analysis starts).

---

## 11. Frontend

### 11.1 Pages

| File | Purpose | Notes |
|---|---|---|
| `index.html` | Landing page | Uses nav/footer fragments |
| `detect.html` | Diagnostic Studio | Mode tabs, drop zone, sample grid, results cards, triage banner |
| `disease.html` | Dedicated bone-disease page | ~150 lines |
| `features.html`, `about.html` | Marketing / info | — |
| `gradcam.html` | Explainer page for overlays | Loads nav/footer itself via `./nav-template.html` (its own inline loader) |

Deleted in the working tree at time of writing: `contact.html`, `team.html`, `images/member1–5.jpg`. Treat the current working-tree deletions as the user's in-progress work (see §13.4).

### 11.2 Scripts

- `detect-script.js`: the controller for `detect.html`. Owns `API_URL`, sample loading, drag/drop, preview, fetch to the three endpoints, progress bar animation, result rendering for all three modes.
- `nav-handler.js` / `footer-handler.js`: fetch the templates, set the active nav link from `window.location.pathname`, wire the mobile menu, update `.current-year`, show/hide `.back-to-top` on scroll.
- Vanilla JS, no bundler, no framework, no build step. Scripts are plain globals on `DOMContentLoaded`.

### 11.3 Styles and icons

- `styles.css` is the main stylesheet. Colors are CSS variables (e.g. `var(--primary)` is used by nav-handler). Prefer the existing variables over raw hex values.
- Font Awesome is loaded from cdnjs. **Two versions are in use** (`5.15.4` on some pages and `6.4.0` on others). Standardize only if asked; otherwise do not add a third.

### 11.4 Known frontend constraints

- `API_URL` is hard-coded to `http://localhost:8000` for any origin that is not localhost. Deploying the frontend to another host will break the API calls. This is intentional for local use only.
- The fallback sample list in `detect-script.js` duplicates the backend `/samples` manifest. If you add a sample, update both or remove the duplication.
- Result images are rendered by writing `src` on `<img>` elements with absolute API URLs.

---

## 12. Coding conventions

### 12.1 Python (backend)

- Python 3, no type-checker configured. Code uses type-less function signatures; keep it that way unless adding a typed module on purpose.
- Route handlers are `async def` and use `UploadFile`. Heavy CPU work (YOLO, OpenCV) currently runs **inside the async handler, blocking the event loop**. If you add concurrency, use `run_in_threadpool` or a plain `def` route. Do not silently change this in unrelated edits.
- Errors: validate input → `HTTPException(400)`; wrap the analysis in `try` and raise `HTTPException(500, detail=f"...: {e}")` after `logger.error` + `traceback.format_exc()`. Match this pattern for new routes.
- Logging: module logger `"MediScan"` via `logging.basicConfig(INFO)`. Use `logger.info/error`, not `print`, in backend code. (`run.py` and `test_api.py` use `print`.)
- IDs: `uuid4()` string for every upload and its outputs. Filenames are `<uuid><ext>` or `<uuid>_<suffix>.jpg`. Keep that scheme so results can be correlated by `detection_id`.
- Paths: always build from `BACKEND_DIR` / `BASE_DIR` computed from `__file__`. Never rely on the current working directory.
- JSON numbers: confidences are percentages rounded to one decimal. Box coordinates are pixel floats rounded to one decimal.
- Keep response keys in `snake_case`, and keep the existing values such as `"Urgent / Orthopedic Consult"` exactly, because the frontend may match on them.
- Duplication: `/detect` and `/comprehensive-scan` duplicate the detection-to-dict block. New code should not copy it a third time; extract a helper if you touch it.

### 12.2 JavaScript (frontend)

- Vanilla ES2015+ inside `DOMContentLoaded` handlers. Use `const`/`let`, arrow functions, template literals — matching the existing files.
- No modules, no bundler: do not introduce `import`/`export` without also changing how the HTML loads the script.
- Use `fetch` with `FormData` for uploads, as `detect-script.js` does.
- Comments: short section headers (`// DOM Elements`, `// Load Samples Gallery`) are the house style. Keep comment density in line with the surrounding file.

### 12.3 Global component-size rule

The user's global rules cap `.tsx` / `.jsx` component files at **250 lines**. This project has **no React/JSX**, so that rule does not apply directly. As a guideline for vanilla JS, prefer splitting a file once it grows past ~400 lines. `detect-script.js` is already 457 lines; if you add a meaningful feature to it, consider extracting a helper module and say so in your summary.

### 12.4 Naming

- Python: `snake_case` functions and variables, `PascalCase` classes (`BoneDiseaseEngine`).
- Disease class names are **display strings** (e.g. `"Osteoarthritis (OA)"`) and are used as dictionary keys in responses. Changing a class name is a breaking API change: update `DISEASE_CLASSES`, the scoring branches, the text branches, the heatmap indices, and the frontend together.
- Heatmap branches are keyed by integer index (`disease_idx == 1` for OA). Keep `DISEASE_CLASSES` ordering stable or update every index.

---

## 13. Git and change-management

### 13.1 Branch and history

- Default branch: `main`. Recent history is short, with descriptive commit subjects (e.g. "implement YOLOv8 detection API with enhanced CORS, logging, and Grad-CAM visualization").
- Write commit subjects in the imperative mood, under ~72 characters.

### 13.2 Commit trailers

When Claude creates a commit in this repo, end the message with:

```
Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>
```

When Claude opens a pull request, end the description with:

```
🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

### 13.3 What must not be committed without explicit user approval

- `backend/uploads/` and `backend/results/` (patient-like imagery and generated outputs). Add to `.gitignore` when the user asks for one.
- `backend/__pycache__/`
- Large binary weight files: `backend/models/model.pt` is already tracked. Do not replace it, re-track it, or move it to LFS without asking.
- `frontend/sample_xrays/` and `sample_xrays/` changes that alter which images are used as samples.

### 13.4 The working tree is not clean

At the time this file was written, `git status` showed many uncommitted modifications, deletions (e.g. `frontend/contact.html`, `frontend/team.html`, `frontend/images/member*.jpg`), and untracked files (`backend/disease_analyzer.py`, `backend/generate_samples.py`, `backend/test_api.py`, `frontend/disease.html`, `run.py`, many `backend/uploads/*`). **Treat these as the user's in-progress work.**

- Run `git status` before any git operation.
- Never run `git checkout -- <file>`, `git restore`, `git reset --hard`, `git clean`, or `git stash` to discard changes without explicit approval.
- Stage specific files by name. Avoid `git add -A` / `git add .`, because it would sweep in uploads and results.

### 13.5 Pull requests

- Target `main`. Keep PRs scoped to one concern (e.g. "disease heatmaps" vs. "frontend team page").
- Include in the PR description: what changed, which endpoints are affected, whether the response shape changed, and whether `test_api.py` was run (and its result).

---

## 14. Testing and verification guidance

### 14.1 What exists

- `backend/test_api.py`: four end-to-end tests with `TestClient`:
  - `test_health` — 200 and `status == "healthy"`.
  - `test_samples` — 200 and ≥ 8 samples.
  - `test_fracture_detect` — 200, `mode == "fracture"`, required keys present.
  - `test_disease_detect` — 200, `mode == "disease"`, required keys present.
  - `test_comprehensive_scan` — 200, `mode == "comprehensive"`, required keys present.
- Tests assert shape and status only. They do **not** assert clinical correctness, detection counts, or class labels, so passing tests say nothing about accuracy.

### 14.2 What to add when you change behavior

- A test per new endpoint, asserting its status code and top-level keys.
- For any change to clinical logic (scores, thresholds, severity bands, triage), add a **regression test on fixed inputs** that pins the current output, and report the before/after values in your summary.
- For sample mapping changes, assert the mapped file exists and has the expected dimensions.
- Do not add tests that only snapshot model outputs without explaining what they protect.

### 14.3 Manual verification

For UI changes, run `python run.py` and check `detect.html` in all three modes with a sample and with an upload. Check the browser console for failed fetches.

---

## 15. Known limitations & technical debt

Ordered by how much they matter for correctness and safety.

1. **Disease classification is heuristic, not learned.** The `feature_net` is never executed (§7.1). Predictions are a function of hand-tuned biomarker thresholds, and of the upload's filename (§7.4). Presenting these as AI diagnoses overstates what the code does.
2. **Filename hints bias results.** Substring matching on the original filename (`"oa"`, `"osteo"`, `"normal"`) changes the class. This is a demo convenience that would be a serious defect on real patient data.
3. **Fracture severity compares different scales.** `max_conf > 0.60` is applied to confidences stored as 0–100 (§6.3). Every detection is labelled `Severe`, so the "Moderate / Splinting" branch is effectively dead. Verify and fix together with any change to the threshold.
3a. **Displayed fracture confidence is wrong by 100×.** `generate_fracture_clinical_summary` prints `max_conf*100` as a percentage, so 87.3% renders as "8730.0%" (§6.3). Visible to every user of `/detect` and `/comprehensive-scan`. Fix this first; it is a one-line change with a clear regression test.
4. **"Grad-CAM" is not Grad-CAM** (§8). Copy and code comments should say "attention-style overlay".
5. **Heatmap for lesions is a fixed center hotspot**, not a localization (§7.6).
6. **`texture_entropy` is local standard deviation**, not entropy. Rename only together with the report key and frontend label, or document it.
7. **`Critical` severity is referenced** in `master_triage` but never produced.
8. **Sample mapping is wrong** for at least `elbow_fracture` (an ankle image) and duplicated for `forearm_fracture` / `wrist_fracture` (§9).
9. **`setup_samples()` runs on every launch and overwrites** sample files.
10. **CORS is `allow_origins=["*"]` with `allow_credentials=True`.** Acceptable for a local demo; not acceptable for any deployment. Restrict origins before exposing the server.
11. **No upload size limit, no content sniffing, no extension whitelist.** The file extension is taken from the client filename and written to disk, then served from `/uploads`. Restrict to `.jpg/.jpeg/.png` and validate the image bytes before deployment.
12. **Uploads and results are never deleted.** Retention policy is undefined (§10).
13. **Heavy inference runs on the event loop** inside `async def` routes (§12.1).
14. **Duplicate code** between `/detect` and `/comprehensive-scan`.
15. **Module-level side effects**: model load, engine construction, RNG seeding, and directory creation all happen at import (§3.1, §7.7). Tests and tooling pay the full cost.
16. **Unused dependencies** in `requirements.txt` (§4.2).
17. **Duplicate route decorators**: `/status` and `/health` are the same handler. Harmless, but intentional only if `/status` is part of the contract.
18. **`self.transform` and `explanations/`** are dead code.
19. **`frontend` API base is hard-coded** to `localhost:8000` off-localhost (§11.4).
20. **No `.gitignore`**, and generated artifacts and a 22 MB weight file are in git history (§13.3).
21. **Tests assert shape, not correctness** (§14.1). Nothing validates the model's accuracy in the repo; the notebook's metrics are not reproduced by any script.
22. **Training notebook has hard-coded Windows and Colab paths** and imports TensorFlow/Keras helpers that appear unused in the YOLO workflow. It is a record of experiments, not a reproducible pipeline.

---

## 16. Guardrails for AI agents working here

**Do**

- State the clinical disclaimer when you change anything that affects what a user would read as a medical result.
- Keep response shapes and key names stable. If you must change one, update the backend, the frontend, `README.md`, and `test_api.py` in the same change, and tell the user.
- Keep the route-before-`/`-mount order (§3.4).
- Compute paths from `__file__`.
- Verify model class names from `model.pt` before writing copy that names them.
- Report what you could not verify (for example, "tests not run because torch is not installed").

**Don't**

- Don't describe the disease engine as trained, learned, validated, or AI-diagnostic without the §7.1 caveat.
- Don't describe the attention overlay as Grad-CAM (§8).
- Don't add new "clinical" severity levels, triage labels, or recommendation text without flagging that they are unvalidated templates.
- Don't add a route that lists, exports, or deletes uploads without asking; these are patient-like images.
- Don't remove the `/` frontend mount's last position, or add `StaticFiles` mounts after it.
- Don't commit `uploads/`, `results/`, `__pycache__/`, or change `model.pt` without approval (§13.3).
- Don't run destructive git commands on the working tree (§13.4).
- Don't "fix" sample images or their mapping silently (§9).
- Don't add a dependency for a one-line need.
- Don't expose the server publicly, or change `host="0.0.0.0"` / CORS, as a side effect of another change.

---

## 17. Playbooks

### 17.1 Add a new analysis endpoint

1. Add the route in `backend/app.py` with `async def name(file: UploadFile = File(...))`.
2. Copy the validation block: content-type check → `uuid4` filename → write bytes → `try/except` with `HTTPException(500)`.
3. Put analysis logic in its own module (like `disease_analyzer.py`), not inline in `app.py`. `app.py` is already ~450 lines.
4. Return a dict with `detection_id`, `mode`, `message`, and the `original_image` URL, following §5.
5. Register any new static output directory **before** the `/` mount.
6. Add a test to `test_api.py`.
7. Add the endpoint to the README table and to §5 of this file.
8. Wire the frontend in `detect-script.js` (endpoint selection + rendering).

### 17.2 Add a disease class

1. Append to `DISEASE_CLASSES` **at the end** to keep existing indices stable.
2. Add a scoring term to `analyze()`, and include it in `raw_scores`. Mind the array order.
3. Add a branch to the grading/text block with `severity`, `kl_grade`, `bmd_status`, `findings`, `recommendations`, `triage`.
4. Add a branch in `_generate_disease_heatmap` keyed by the new index.
5. Update `master_triage` logic if the new severity should escalate.
6. Update frontend rendering if it has class-specific copy.
7. Document that the scoring is heuristic, and add a pinned regression test (§14.2).

### 17.3 Replace or retrain the fracture model

1. Train in `models/yolo_model.ipynb` or equivalent. Record the dataset version, `imgsz`, epochs and the metrics.
2. Export weights to `backend/models/model.pt`. Replacing this 22 MB file is a repo-level change: ask the user first, and note the new size and class list.
3. Check `model.names`. If class names changed, update `formatted_class` rules (§6.2), the clinical summary wording (§6.3), and any frontend copy.
4. Re-run `test_api.py`, and check that detection counts on the sample set are plausible.
5. Note the confidence threshold (0.15) — re-evaluate it for the new model rather than keeping it by default.

### 17.4 Add a sample

1. Put the file in `sample_xrays/`. Confirm with the user that it is a legitimately usable image and that its label is correct.
2. Add the entry to `/samples` in `app.py`, with `id`, `name`, `category`, `filename`, `url`, `description`.
3. Add the matching fallback entry in `detect-script.js` (§11.4).
4. Copy into `frontend/sample_xrays/`.
5. Do **not** add it to `setup_samples()` unless the user wants it regenerated on every launch.

### 17.5 Run the app for a change

Use `python run.py`, then open `detect.html`. For API-only checks, use `/docs` (Swagger) to submit an image. For a quick scripted check, use `TestClient` as in `test_api.py`.

---

## 18. Glossary

- **Bone mask** — binary mask of bone-tissue pixels, derived by thresholding and closing (§7.2).
- **Biomarker** — a scalar image statistic in §7.3. Not a clinical lab value.
- **BMD / T-score** — bone mineral density / its standard deviation relative to a young-adult reference. Here, an *estimate* from intensity only.
- **KL grade** — Kellgren–Lawrence grade (0–4) for knee/joint osteoarthritis. Shown here as text from heuristics.
- **Narrowing index** — heuristic proxy for joint-space narrowing (§7.3).
- **Osteomyelitis** — bone infection; the "Osteomyelitis (Infection)" class.
- **Triage** — urgency routing string returned to the UI.
- **Attention-style overlay** — the Gaussian heatmap from `generate_gradcam_heatmap` (§8).
- **Detection ID** — UUID tying together an upload and all outputs.
- **Sample gallery** — the 8 curated images served by `/samples`.

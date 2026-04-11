# DeepFake Detector — Multi-Branch Ensemble System

> **Engineering Final Project (PCD)**  
> A research-grade deepfake detection system combining four independent analysis branches into an ensemble classifier.

---

## Architecture

```
Video Input
    │
    ├─► Branch 1 · Temporal-Spatial Flow    (3D-CNN / MesoNet)
    ├─► Branch 2 · Audio-Visual Sync        (SyncNet-style)
    ├─► Branch 3 · Spatial-Freq Artifacts   (FFT + 2D CNN)
    └─► Branch 4 · Biological rPPG          (Pulse detection)
              │
              ▼
    Feature Fusion (concat 4×128 → 512)
              │
              ▼
    MLP Ensemble Classifier
              │
              ▼
    Deepfake Probability Score [0, 1]
```

---

## Project Structure

```
PCD/
├── backend/
│   ├── main.py                    # FastAPI app, CORS, /predict endpoint
│   ├── pipeline.py                # Multi-branch orchestration
│   ├── schemas.py                 # Pydantic request/response models
│   ├── requirements.txt
│   ├── core/
│   │   ├── config.py              # Pydantic settings (.env)
│   │   └── utils.py               # Video/audio preprocessing stubs
│   ├── modules/
│   │   ├── temporal_spatial_flow.py      # Branch 1
│   │   ├── audio_visual_sync.py          # Branch 2
│   │   ├── spatial_frequency_artifacts.py # Branch 3
│   │   └── biological_rppg.py            # Branch 4
│   └── models/
│       └── fusion_classifier.py   # MLP feature fusion + scoring
│
├── frontend/
│   ├── package.json
│   └── src/
│       ├── app/
│       │   ├── layout.jsx
│       │   ├── page.jsx           # Main page
│       │   ├── page.module.css
│       │   └── globals.css
│       ├── components/
│       │   ├── VideoUploader.jsx  # Drag-and-drop upload
│       │   ├── VideoUploader.module.css
│       │   ├── ScoreDisplay.jsx   # Result card with gauge
│       │   └── ScoreDisplay.module.css
│       └── services/
│           └── api.js             # Axios API client
│
├── models/
│   └── weights/                   # Place trained .pth files here
│
└── .env.example                   # Environment variable template
```

---

## Quick Start

### 1 — Backend

```bash
# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r backend/requirements.txt

# Copy environment file
copy .env.example .env

# Run the API
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

API docs auto-generated at: **http://localhost:8000/docs**

### 2 — Frontend

```bash
cd frontend
npm install
npm run dev
```

App available at: **http://localhost:3000**

---

## Development Roadmap

| Branch | Status | Next Step |
|--------|--------|-----------|
| Temporal-Spatial Flow | 🟡 Stub | Integrate `torchvision.models.video.r3d_18` |
| Audio-Visual Sync | 🟡 Stub | Train SyncNet on LRS3 / VoxCeleb2 |
| Spatial-Freq Artifacts | 🟡 Stub | Train 2D-CNN on FFT of FaceForensics++ |
| Biological rPPG | 🟡 Stub | Integrate rPPG-Toolbox / PhysNet |
| Fusion Classifier | 🟡 Random weights | Train end-to-end on fused embeddings |

---

## References

- MesoNet: Afchar et al., 2018
- SyncNet: Chung & Zisserman, ACCV 2016
- FFT Forgery Detection: Frank et al., ICML 2020
- rPPG-Toolbox: Liu et al., NeurIPS 2023

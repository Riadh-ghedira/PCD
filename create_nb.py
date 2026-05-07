import json

def cell_md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def cell_code(lines):
    parts = lines.split("\n")
    source = [line + "\n" for line in parts[:-1]] + [parts[-1]]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

GUARD = """\
import sys, os
def _find_backend(root='/content/drive/MyDrive'):
    for dp, dns, _ in os.walk(root):
        if os.path.basename(dp)=='backend' and os.path.isdir(os.path.join(dp,'modules')) and os.path.isdir(os.path.join(dp,'core')):
            return dp
    return None
BACKEND_PATH = _find_backend()
if not BACKEND_PATH: raise FileNotFoundError("backend/ not found under MyDrive. Upload your project first.")
PROJECT_ROOT = os.path.dirname(BACKEND_PATH)
WEIGHTS_PATH = os.path.join(BACKEND_PATH, 'models', 'weights')
for p in [PROJECT_ROOT, BACKEND_PATH]:
    if p not in sys.path: sys.path.insert(0, p)
os.makedirs(WEIGHTS_PATH, exist_ok=True)
print(f"[OK] backend: {BACKEND_PATH}")
"""

cells = []

# ── Cell 1 ────────────────────────────────────────────────────────────────────
cells.append(cell_md("## Cell 1: Environment Setup & Drive Mounting"))
cells.append(cell_code("""\
from google.colab import drive
drive.mount('/content/drive')
!pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 moviepy facenet-pytorch opencv-python pydantic-settings
""" + GUARD))

# ── Cell 1b ───────────────────────────────────────────────────────────────────
cells.append(cell_md("## Cell 1b: Path Diagnostic"))
cells.append(cell_code("""\
import sys, os
print("sys.path[:3]:", sys.path[:3])
for p in sys.path:
    m = os.path.join(p, 'modules')
    if os.path.isdir(m):
        print("[FOUND] modules/ at:", m, "->", os.listdir(m)); break
else:
    print("[MISSING] modules/ not found. Re-run Cell 1.")
    for dp,dns,_ in os.walk('/content/drive/MyDrive'):
        if 'modules' in dns and 'core' in dns: print(" candidate:", dp)
"""))

# ── Cell 2 ────────────────────────────────────────────────────────────────────
cells.append(cell_md("## Cell 2: Global Utilities"))
cells.append(cell_code("""\
import torch, torch.nn as nn, torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)
"""))

# ── Cell 3: SFA full training ─────────────────────────────────────────────────
cells.append(cell_md("## Cell 3: Train Module 3 — Spatial Frequency Artifacts (Celeb-DF)"))
cells.append(cell_code(GUARD + """\
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from modules.spatial_frequency_artifacts import SpatialFrequencyArtifactsModule
from core.preprocessing.face_extractor import FaceExtractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Dataset ──────────────────────────────────────────────────────────────────
class CelebDFDataset(Dataset):
    \"\"\"
    Folder structure expected:
        CELEB_DF_ROOT/
            real/  *.mp4
            fake/  *.mp4
    \"\"\"
    def __init__(self, root_dir, max_frames=4):
        self.samples = []
        self.max_frames = max_frames
        self.face_extractor = FaceExtractor(device='cpu')
        for label, sub in [(0,'real'),(1,'fake')]:
            folder = os.path.join(root_dir, sub)
            if os.path.isdir(folder):
                for f in sorted(os.listdir(folder)):
                    if f.lower().endswith(('.mp4','.avi','.mov')):
                        self.samples.append((os.path.join(folder,f), label))
        print(f"[CelebDFDataset] {len(self.samples)} videos (real+fake)")

    def _fft_map(self, face):                          # face: (C,H,W)
        gray = torch.mean(face.float(), dim=0)         # (H,W)
        mag  = torch.abs(torch.fft.fftshift(torch.fft.fft2(gray)))
        mag  = torch.log(1 + mag)
        return mag.unsqueeze(0).repeat(3,1,1)          # (3,H,W)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = self.face_extractor.extract_face_sequence(path, max_frames=self.max_frames)
        if seq is None:
            return torch.zeros(3,224,224), torch.tensor(label, dtype=torch.float32)
        fft = self._fft_map(seq[0].cpu())
        return fft, torch.tensor(label, dtype=torch.float32)

# ── Model wrapper (adds scalar logit head) ────────────────────────────────────
# SFA fc outputs 128-d features; we add a Linear(128,1) for BCE training.
# Only fc + logit_head are trained; backbone stays frozen.
class SFATrainer(nn.Module):
    def __init__(self, sfa_module):
        super().__init__()
        self.sfa = sfa_module
        self.logit_head = nn.Linear(128, 1)
    def forward(self, x):                              # x: (B,3,H,W)
        feats = self.sfa.model(x)                      # (B,128)
        return self.logit_head(feats)                  # (B,1)

# ── Init ──────────────────────────────────────────────────────────────────────
sfa_module = SpatialFrequencyArtifactsModule(device=str(device))
sfa_trainer = SFATrainer(sfa_module).to(device)

optimizer_sfa = optim.Adam(
    list(sfa_module.model.fc.parameters()) + list(sfa_trainer.logit_head.parameters()),
    lr=1e-4
)
criterion_sfa = nn.BCEWithLogitsLoss()

# ── DataLoader ────────────────────────────────────────────────────────────────
CELEB_DF_ROOT = '/content/drive/MyDrive/datasets/Celeb-DF'   # <-- adjust path
# dataset_sfa = CelebDFDataset(CELEB_DF_ROOT)
# loader_sfa  = DataLoader(dataset_sfa, batch_size=16, shuffle=True, num_workers=2)

# ── Training loop ─────────────────────────────────────────────────────────────
EPOCHS_SFA = 15
print("Starting SFA training...")
for epoch in range(EPOCHS_SFA):
    sfa_trainer.train()
    running_loss, correct, total = 0.0, 0, 0
    # for fft_maps, labels in loader_sfa:
    #     fft_maps = fft_maps.to(device)
    #     labels   = labels.to(device).unsqueeze(1)
    #     optimizer_sfa.zero_grad()
    #     logits = sfa_trainer(fft_maps)
    #     loss   = criterion_sfa(logits, labels)
    #     loss.backward(); optimizer_sfa.step()
    #     running_loss += loss.item() * fft_maps.size(0)
    #     correct += ((torch.sigmoid(logits)>=0.5).float()==labels).sum().item()
    #     total   += labels.size(0)
    # print(f"Epoch {epoch+1}/{EPOCHS_SFA}  loss={running_loss/total:.4f}  acc={correct/total:.4f}")
    print(f"[Placeholder] Epoch {epoch+1}/{EPOCHS_SFA} — uncomment loader_sfa block above")

# ── Save (full SFA module, not the trainer wrapper) ───────────────────────────
sfa_path = os.path.join(WEIGHTS_PATH, 'sfa_module.pth')
torch.save(sfa_module.state_dict(), sfa_path)
print(f"SFA saved -> {sfa_path}")
"""))

# ── Cell 4: rPPG full training ────────────────────────────────────────────────
cells.append(cell_md("## Cell 4: Train Module 4 — Biological rPPG (UBFC-rPPG / FaceForensics++)"))
cells.append(cell_code(GUARD + """\
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from modules.biological_rppg import BiologicalRPPGModule
from core.preprocessing.face_extractor import FaceExtractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Dataset ──────────────────────────────────────────────────────────────────
class RPPGDataset(Dataset):
    \"\"\"
    Folder structure expected:
        RPPG_ROOT/
            real/  *.mp4
            fake/  *.mp4
    Extracts face sequence and computes spatial mean RGB per frame -> (T,3) signal.
    \"\"\"
    def __init__(self, root_dir, max_frames=32):
        self.samples = []
        self.max_frames = max_frames
        self.face_extractor = FaceExtractor(device='cpu')
        for label, sub in [(0,'real'),(1,'fake')]:
            folder = os.path.join(root_dir, sub)
            if os.path.isdir(folder):
                for f in sorted(os.listdir(folder)):
                    if f.lower().endswith(('.mp4','.avi','.mov')):
                        self.samples.append((os.path.join(folder,f), label))
        print(f"[RPPGDataset] {len(self.samples)} videos")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = self.face_extractor.extract_face_sequence(path, max_frames=self.max_frames)
        if seq is None:
            return torch.zeros(self.max_frames, 3), torch.tensor(label, dtype=torch.float32)
        # (T,C,H,W) -> spatial mean -> (T,3)
        signal = seq.float().mean(dim=(2,3))            # (T,3)
        # Pad/trim to fixed length
        T = self.max_frames
        if signal.size(0) < T:
            pad = torch.zeros(T - signal.size(0), 3)
            signal = torch.cat([signal, pad], dim=0)
        else:
            signal = signal[:T]
        return signal, torch.tensor(label, dtype=torch.float32)

# ── Model wrapper (adds scalar logit head for BCE) ────────────────────────────
# rPPG fc outputs (B,128); logit_head reduces to (B,1)
class RPPGTrainer(nn.Module):
    def __init__(self, rppg_module):
        super().__init__()
        self.rppg = rppg_module
        self.logit_head = nn.Linear(128, 1)
    def forward(self, x):                             # x: (B,T,3)
        out, _ = self.rppg.rnn(x)                    # (B,T,64)
        state  = out[:, -1, :]                        # (B,64)
        feats  = self.rppg.fc(state)                  # (B,128)
        return self.logit_head(feats)                 # (B,1)

# ── Init ──────────────────────────────────────────────────────────────────────
rppg_module  = BiologicalRPPGModule(device=str(device))
rppg_trainer = RPPGTrainer(rppg_module).to(device)

optimizer_rppg = optim.Adam(
    list(rppg_module.rnn.parameters()) +
    list(rppg_module.fc.parameters()) +
    list(rppg_trainer.logit_head.parameters()),
    lr=1e-3
)
criterion_rppg = nn.BCEWithLogitsLoss()

# ── DataLoader ────────────────────────────────────────────────────────────────
RPPG_ROOT = '/content/drive/MyDrive/datasets/UBFC-rPPG'   # <-- adjust path
# dataset_rppg = RPPGDataset(RPPG_ROOT, max_frames=32)
# loader_rppg  = DataLoader(dataset_rppg, batch_size=32, shuffle=True, num_workers=2)

# ── Training loop ─────────────────────────────────────────────────────────────
EPOCHS_RPPG = 20
print("Starting rPPG training...")
for epoch in range(EPOCHS_RPPG):
    rppg_trainer.train()
    running_loss, correct, total = 0.0, 0, 0
    # for signals, labels in loader_rppg:
    #     signals = signals.to(device)               # (B,T,3)
    #     labels  = labels.to(device).unsqueeze(1)   # (B,1)
    #     optimizer_rppg.zero_grad()
    #     logits = rppg_trainer(signals)
    #     loss   = criterion_rppg(logits, labels)
    #     loss.backward(); optimizer_rppg.step()
    #     running_loss += loss.item() * signals.size(0)
    #     correct += ((torch.sigmoid(logits)>=0.5).float()==labels).sum().item()
    #     total   += labels.size(0)
    # print(f"Epoch {epoch+1}/{EPOCHS_RPPG}  loss={running_loss/total:.4f}  acc={correct/total:.4f}")
    print(f"[Placeholder] Epoch {epoch+1}/{EPOCHS_RPPG} — uncomment loader_rppg block above")

# ── Save ──────────────────────────────────────────────────────────────────────
rppg_path = os.path.join(WEIGHTS_PATH, 'rppg_module.pth')
torch.save(rppg_module.state_dict(), rppg_path)
print(f"rPPG saved -> {rppg_path}")
"""))

# ── Cell 5: TSF ───────────────────────────────────────────────────────────────
cells.append(cell_md("## Cell 5: Train Module 1 — Temporal Spatial Flow (FaceForensics++)"))
cells.append(cell_code(GUARD + """\
import torch, torch.nn as nn, torch.optim as optim
from modules.temporal_spatial_flow import TemporalSpatialFlowModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tsf_model = TemporalSpatialFlowModule(device=str(device))

# Temp logit head: R3D-18 fc outputs (B,128); add (128->1) for BCE
class TSFTrainer(nn.Module):
    def __init__(self, tsf): super().__init__(); self.tsf=tsf; self.head=nn.Linear(128,1)
    def forward(self,x): return self.head(self.tsf.model(x))

tsf_trainer = TSFTrainer(tsf_model).to(device)
optimizer_tsf = optim.Adam(list(tsf_model.model.fc.parameters())+list(tsf_trainer.head.parameters()), lr=1e-4)
criterion_tsf = nn.BCEWithLogitsLoss()

# FF_ROOT = '/content/drive/MyDrive/datasets/FaceForensics'
# loader_tsf = DataLoader(YourFFDataset(FF_ROOT), batch_size=4, shuffle=True)
# Each item: (B,3,16,224,224) clip tensor, (B,) label

EPOCHS_TSF = 10
print("Starting TSF training...")
for epoch in range(EPOCHS_TSF):
    # tsf_trainer.train(); running_loss,correct,total=0.,0,0
    # for clips,labels in loader_tsf:
    #     clips=clips.to(device); labels=labels.to(device).unsqueeze(1)
    #     optimizer_tsf.zero_grad()
    #     loss=criterion_tsf(tsf_trainer(clips),labels)
    #     loss.backward(); optimizer_tsf.step()
    #     running_loss+=loss.item()*clips.size(0); total+=labels.size(0)
    # print(f"Epoch {epoch+1}/{EPOCHS_TSF}  loss={running_loss/total:.4f}")
    print(f"[Placeholder] Epoch {epoch+1}/{EPOCHS_TSF}")

tsf_path = os.path.join(WEIGHTS_PATH, 'tsf_module.pth')
torch.save(tsf_model.state_dict(), tsf_path)
print(f"TSF saved -> {tsf_path}")
"""))

# ── Cell 6: AVS ───────────────────────────────────────────────────────────────
cells.append(cell_md("## Cell 6: Train Module 2 — Audio-Visual Sync (VoxCeleb2 / DFDC)"))
cells.append(cell_code(GUARD + """\
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from modules.audio_visual_sync import AudioVisualSyncModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
avs_model = AudioVisualSyncModule(device=str(device))

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0): super().__init__(); self.margin=margin
    def forward(self, d, y):
        return torch.mean(y*d.pow(2) + (1-y)*torch.clamp(self.margin-d,min=0).pow(2))

optimizer_avs = optim.Adam(
    list(avs_model.video_encoder.parameters())+
    list(avs_model.audio_encoder.parameters())+
    list(avs_model.fusion_layer.parameters()), lr=1e-4)
criterion_avs = ContrastiveLoss(margin=1.0)

# av_loader yields: ((video (B,3,8,H,W), mel (B,64,L)), label (B,))
EPOCHS_AVS = 10
print("Starting AVS contrastive training...")
for epoch in range(EPOCHS_AVS):
    # avs_model.train(); running_loss,total=0.,0
    # for (vid,aud),labels in av_loader:
    #     vid=vid.to(device); aud=aud.to(device); labels=labels.to(device).float()
    #     optimizer_avs.zero_grad()
    #     dist=F.pairwise_distance(avs_model.video_encoder(vid),avs_model.audio_encoder(aud))
    #     loss=criterion_avs(dist,labels); loss.backward(); optimizer_avs.step()
    #     running_loss+=loss.item()*vid.size(0); total+=vid.size(0)
    # print(f"Epoch {epoch+1}/{EPOCHS_AVS}  contrastive_loss={running_loss/total:.4f}")
    print(f"[Placeholder] Epoch {epoch+1}/{EPOCHS_AVS}")

avs_path = os.path.join(WEIGHTS_PATH, 'avs_module.pth')
torch.save(avs_model.state_dict(), avs_path)
print(f"AVS saved -> {avs_path}")
"""))

# ── Cell 7: Feature Caching ───────────────────────────────────────────────────
cells.append(cell_md("## Cell 7: Cache Features from All 4 Modules (required before Fusion training)"))
cells.append(cell_code(GUARD + """\
import torch
from pathlib import Path
from modules.temporal_spatial_flow import TemporalSpatialFlowModule
from modules.audio_visual_sync import AudioVisualSyncModule
from modules.spatial_frequency_artifacts import SpatialFrequencyArtifactsModule
from modules.biological_rppg import BiologicalRPPGModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load all 4 modules with their trained weights ─────────────────────────────
tsf = TemporalSpatialFlowModule(device=str(device)).eval()
avs = AudioVisualSyncModule(device=str(device)).eval()
sfa = SpatialFrequencyArtifactsModule(device=str(device)).eval()
bio = BiologicalRPPGModule(device=str(device)).eval()

tsf.load_state_dict(torch.load(os.path.join(WEIGHTS_PATH,'tsf_module.pth'), map_location=device))
avs.load_state_dict(torch.load(os.path.join(WEIGHTS_PATH,'avs_module.pth'), map_location=device))
sfa.load_state_dict(torch.load(os.path.join(WEIGHTS_PATH,'sfa_module.pth'), map_location=device))
bio.load_state_dict(torch.load(os.path.join(WEIGHTS_PATH,'rppg_module.pth'), map_location=device))
print("All 4 modules loaded.")

# ── Cache features from a video folder ───────────────────────────────────────
# VIDEO_FOLDER structure: VIDEO_FOLDER/real/*.mp4  and  VIDEO_FOLDER/fake/*.mp4
VIDEO_FOLDER = '/content/drive/MyDrive/datasets/FaceForensics'   # <-- adjust
CACHE_DIR    = '/content/drive/MyDrive/cached_features'
os.makedirs(CACHE_DIR, exist_ok=True)

video_paths = []
for label, sub in [(0,'real'),(1,'fake')]:
    folder = os.path.join(VIDEO_FOLDER, sub)
    if os.path.isdir(folder):
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(('.mp4','.avi','.mov')):
                video_paths.append((os.path.join(folder,f), label))
print(f"Found {len(video_paths)} videos to cache.")

for idx,(vpath,label) in enumerate(video_paths):
    save_path = os.path.join(CACHE_DIR, f"{Path(vpath).stem}_label{label}.pt")
    if os.path.exists(save_path):
        print(f"[{idx+1}] skip (cached): {Path(vpath).name}"); continue
    try:
        with torch.no_grad():
            t = tsf.extract_features(vpath)["temporal_features"].cpu()   # (1,128)
            a = avs.extract_features(vpath)["sync_features"].cpu()        # (1,128)
            s = sfa.extract_features(vpath)["artifact_features"].cpu()    # (1,128)
            b = bio.extract_features(vpath)["rppg_features"].cpu()        # (1,128)
        torch.save({"features": torch.cat([t,a,s,b],dim=1).squeeze(0),   # (512,)
                    "label": torch.tensor(label, dtype=torch.float32)}, save_path)
        print(f"[{idx+1}/{len(video_paths)}] cached: {Path(vpath).name}")
    except Exception as e:
        print(f"[{idx+1}] ERROR {Path(vpath).name}: {e}")

print("Feature caching complete.")
"""))

# ── Cell 8: Fusion Classifier Training ───────────────────────────────────────
cells.append(cell_md("## Cell 8: Train Fusion Classifier (uses cached features from Cell 7)"))
cells.append(cell_code(GUARD + """\
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.fusion_classifier import FusionClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Dataset: loads .pt files saved by Cell 7 ─────────────────────────────────
class CachedFusionDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = [os.path.join(cache_dir,f) for f in os.listdir(cache_dir) if f.endswith('.pt')]
        print(f"[FusionDataset] {len(self.files)} cached samples")
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d = torch.load(self.files[idx])
        return d["features"], d["label"]   # (512,), scalar

CACHE_DIR = '/content/drive/MyDrive/cached_features'   # same as Cell 7
dataset_fusion = CachedFusionDataset(CACHE_DIR)

n_train = int(0.8 * len(dataset_fusion))
n_val   = len(dataset_fusion) - n_train
train_ds, val_ds = torch.utils.data.random_split(dataset_fusion, [n_train, n_val])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=2)

# ── FusionClassifier ─────────────────────────────────────────────────────────
# input_dim=512 (4 x 128-d branch vectors concatenated)
# FusionClassifier.forward returns (B,) probabilities (sigmoid applied internally)
fusion_model = FusionClassifier(input_dim=512, device=device)
fusion_model.to(device)

# FusionClassifier already applies sigmoid, so use BCELoss (not BCEWithLogitsLoss)
criterion_fusion = nn.BCELoss()
optimizer_fusion = optim.Adam(fusion_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler        = optim.lr_scheduler.StepLR(optimizer_fusion, step_size=15, gamma=0.5)

# ── Training loop ─────────────────────────────────────────────────────────────
EPOCHS_FUSION = 50
best_val_acc = 0.0
print("Starting Fusion Classifier training...")

for epoch in range(EPOCHS_FUSION):
    # Train
    fusion_model.train()
    tr_loss, tr_correct, tr_total = 0., 0, 0
    for feats, labels in train_loader:
        feats  = feats.to(device)
        labels = labels.to(device)
        optimizer_fusion.zero_grad()
        probs  = fusion_model(feats)           # (B,)  — sigmoid applied inside
        loss   = criterion_fusion(probs, labels)
        loss.backward(); optimizer_fusion.step()
        tr_loss    += loss.item() * feats.size(0)
        tr_correct += ((probs >= 0.5).float() == labels).sum().item()
        tr_total   += labels.size(0)

    # Validate
    fusion_model.eval()
    val_loss, val_correct, val_total = 0., 0, 0
    with torch.no_grad():
        for feats, labels in val_loader:
            feats  = feats.to(device); labels = labels.to(device)
            probs  = fusion_model(feats)
            val_loss    += criterion_fusion(probs, labels).item() * feats.size(0)
            val_correct += ((probs >= 0.5).float() == labels).sum().item()
            val_total   += labels.size(0)

    tr_acc  = tr_correct  / tr_total
    val_acc = val_correct / val_total
    scheduler.step()
    print(f"Epoch {epoch+1:02d}/{EPOCHS_FUSION} | "
          f"Train loss={tr_loss/tr_total:.4f} acc={tr_acc:.4f} | "
          f"Val   loss={val_loss/val_total:.4f} acc={val_acc:.4f}")

    # Save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_path = os.path.join(WEIGHTS_PATH, 'fusion_classifier.pth')
        torch.save(fusion_model.state_dict(), best_path)
        print(f"  [BEST] saved -> {best_path}")

print(f"Training complete. Best val acc: {best_val_acc:.4f}")
"""))

# ── Write notebook ────────────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = r"d:\Riadh\Projects\PCD\master_training_colab.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Notebook written -> {out}")

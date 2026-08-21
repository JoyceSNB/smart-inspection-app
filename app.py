"""
Smart Industrial Product Inspection - Web Application
Deliverable #2: Web Deployment

Menggunakan PatchCore sebagai metode final (lihat notebook, section m - dipilih
berdasarkan hasil eksperimen: performa terbaik di semua metrik dibanding
Autoencoder, PaDiM, dan Hybrid).

Cara jalanin:
    streamlit run app.py

Konfigurasi ARTIFACT_DIR dan DATA_ROOT di bawah sesuai lokasi kamu.
"""

import os
import io
import json
import pickle
import sqlite3
import time
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors

import streamlit as st

# ============================================================
# KONFIGURASI - versi Streamlit Community Cloud
# ============================================================
# Path LOKAL (disk sementara Streamlit Cloud, bukan Google Drive lagi)
ARTIFACT_DIR = "/tmp/artifacts"
DB_PATH = "/tmp/inspection_history.db"
UPLOAD_DIR = "/tmp/inspection_uploads"
DATA_ROOT = None   # dataset penuh tidak tersedia di Streamlit Cloud -> kalibrasi otomatis dinonaktifkan

# ID file Google Drive untuk artifacts.zip (ganti dengan punya kamu -- lihat README_STREAMLIT_CLOUD.md)
ARTIFACTS_DRIVE_FILE_ID = st.secrets.get("ARTIFACTS_DRIVE_FILE_ID", "GANTI_DENGAN_FILE_ID_DRIVE_KAMU")

IMG_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CALIBRATION_SAMPLE_SIZE = 150   # jumlah gambar normal yang di-sample untuk kalibrasi threshold otomatis
                                  # (dibuat besar supaya persentil tau_low/tau_high stabil, tidak terlalu
                                  # sensitif ke 1-2 gambar outlier seperti saat sample kecil ~25)


# ============================================================
# MODEL CLASSES
# Identik dengan definisi di notebook, supaya artifact (.pkl) yang
# disimpan dari notebook bisa dimuat & dipakai langsung di sini.
# ============================================================
class FeatureExtractor(nn.Module):
    """Ekstrak & gabungkan fitur dari layer2 + layer3 backbone (WideResNet-50, frozen)."""
    def __init__(self, backbone_name="wide_resnet50_2", layers=("layer2", "layer3")):
        super().__init__()
        backbone = getattr(torchvision.models, backbone_name)(weights="IMAGENET1K_V2")
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone
        self.layers = layers
        self._features = {}
        for name in layers:
            getattr(backbone, name).register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(module, inp, out):
            self._features[name] = out
        return hook

    @torch.no_grad()
    def forward(self, x):
        self._features = {}
        _ = self.backbone(x)
        feats = [self._features[name] for name in self.layers]
        target_size = feats[0].shape[-2:]
        feats = [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False) for f in feats]
        return torch.cat(feats, dim=1)


class PatchCore:
    """Memory bank + nearest-neighbor scoring. fit()/_greedy_coreset() disertakan untuk
    kelengkapan kelas, tapi di web app ini hanya score() yang dipakai (memory bank
    dimuat dari artifact hasil training di notebook, bukan dilatih ulang di sini)."""
    def __init__(self, feature_extractor, coreset_ratio=0.1, max_candidates=20000,
                 n_neighbors=1, random_state=42):
        self.fe = feature_extractor
        self.coreset_ratio = coreset_ratio
        self.max_candidates = max_candidates
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.memory_bank = None
        self.grid_shape = None
        self.nn_index = None

    @torch.no_grad()
    def score(self, x, device, out_size=IMG_SIZE):
        feats = self.fe(x.to(device))
        B, C, H, W = feats.shape
        feats_flat = feats.permute(0, 2, 3, 1).reshape(B, H * W, C).cpu().numpy()

        anomaly_maps = np.zeros((B, H * W))
        for b in range(B):
            dists, _ = self.nn_index.kneighbors(feats_flat[b])
            anomaly_maps[b] = dists.mean(axis=1)
        anomaly_maps = anomaly_maps.reshape(B, 1, H, W)
        anomaly_maps_t = torch.from_numpy(anomaly_maps).float()
        anomaly_maps_up = F.interpolate(anomaly_maps_t, size=(out_size, out_size),
                                          mode="bilinear", align_corners=False).squeeze(1).numpy()
        image_scores = anomaly_maps_up.reshape(B, -1).max(axis=1)
        return anomaly_maps_up, image_scores


# ============================================================
# PREPROCESSING
# ============================================================
transform_eval = transforms.Compose([
    transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.Lambda(lambda im: im.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
])


def preprocess_pil_image(pil_image):
    """PIL Image -> tensor siap masuk FeatureExtractor (batch dim ditambahkan)."""
    return transform_eval(pil_image).unsqueeze(0)


def get_display_image(pil_image, resize_size=RESIZE_SIZE, img_size=IMG_SIZE):
    """Resize+crop yang SAMA persis dengan preprocessing model, supaya heatmap
    overlay sejajar pixel-per-pixel dengan gambar yang ditampilkan (bukan crop yang beda)."""
    im = pil_image.resize((resize_size, resize_size), Image.BILINEAR)
    left = (resize_size - img_size) // 2
    im = im.crop((left, left, left + img_size, left + img_size))
    return np.array(im)


# ============================================================
# POST-PROCESSING (identik dengan notebook, section k)
# ============================================================
def postprocess_anomaly_map(anomaly_map, threshold, open_kernel=3, close_kernel=5, min_area_px=20):
    """anomaly_map: array 2D skala mentah (jarak PatchCore, BUKAN [0,1]).
    Return: binary mask final, daftar region defect (area + bbox), persentase luas area anomali."""
    binary = (anomaly_map >= threshold).astype(np.uint8)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    final_mask = np.zeros_like(cleaned)
    regions = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_px:
            final_mask[labels == i] = 1
            regions.append({"area_px": int(area), "bbox": stats[i, :4].tolist(),
                             "centroid": centroids[i].tolist()})
    area_pct = 100.0 * final_mask.sum() / final_mask.size
    return final_mask, regions, area_pct


# ============================================================
# QC RULE ENGINE (identik dengan notebook, section l)
# ============================================================
def qc_decision(anomaly_score, defect_area_pct, tau_low, tau_high, area_reject_pct=5.0):
    if anomaly_score >= tau_high or defect_area_pct > area_reject_pct:
        return "REJECT"
    elif anomaly_score >= tau_low:
        return "NEEDS MANUAL INSPECTION"
    else:
        return "PASS"


def compute_confidence(score, tau_low, tau_high):
    """Confidence model (0-100%) berdasarkan jarak skor ke threshold terdekat.
    Skor tepat di garis batas (tau_low/tau_high) -> confidence rendah (~50%, paling ambigu).
    Skor jauh dari kedua batas (jelas normal atau jelas defect) -> confidence tinggi (~100%)."""
    if score <= tau_low:
        dist_ratio = (tau_low - score) / max(tau_low, 1e-6)
    elif score >= tau_high:
        dist_ratio = (score - tau_high) / max(tau_high, 1e-6)
    else:
        dist_ratio = 0.0  # di zona abu-abu (NEEDS MANUAL INSPECTION) -> paling tidak pasti
    return float(50 + 50 * min(dist_ratio, 1.0))


# ============================================================
# DATABASE (riwayat inspeksi, untuk dashboard historis)
# ============================================================
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            category TEXT,
            filename TEXT,
            image_path TEXT,
            anomaly_score REAL,
            confidence_pct REAL,
            defect_area_pct REAL,
            n_regions INTEGER,
            decision TEXT,
            inference_time_sec REAL,
            model_name TEXT
        )
    """)
    # migrasi ringan: kalau tabel sudah ada dari versi app.py sebelumnya (skema lama),
    # tambahkan kolom yang belum ada tanpa menghapus data yang sudah tersimpan.
    existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(inspections)").fetchall()]
    required_cols = {"image_path": "TEXT", "confidence_pct": "REAL"}
    for col, col_type in required_cols.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE inspections ADD COLUMN {col} {col_type}")
    conn.commit()
    conn.close()


def insert_inspection(record):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO inspections
        (timestamp, category, filename, image_path, anomaly_score, confidence_pct, defect_area_pct, n_regions,
         decision, inference_time_sec, model_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (record["timestamp"], record["category"], record["filename"], record["image_path"],
          record["anomaly_score"], record["confidence_pct"], record["defect_area_pct"], record["n_regions"],
          record["decision"], record["inference_time_sec"], record["model_name"]))
    conn.commit()
    conn.close()


def load_history(category=None):
    conn = sqlite3.connect(DB_PATH)
    if category and category != "Semua kategori":
        df = pd.read_sql_query("SELECT * FROM inspections WHERE category = ? ORDER BY id DESC", conn, params=(category,))
    else:
        df = pd.read_sql_query("SELECT * FROM inspections ORDER BY id DESC", conn)
    conn.close()
    return df


def save_uploaded_image(pil_image, category, filename):
    """Simpan foto yang diupload ke Drive (permanen, bisa diakses dari device manapun),
    dengan nama unik (timestamp) supaya tidak saling menimpa."""
    cat_dir = os.path.join(UPLOAD_DIR, category)
    os.makedirs(cat_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = f"{ts}_{filename}"
    path = os.path.join(cat_dir, safe_name)
    pil_image.save(path)
    return path


# ============================================================
# DOWNLOAD ARTIFACTS DARI GOOGLE DRIVE (sekali saja, di-cache disk)
# ============================================================
def ensure_artifacts_available():
    """Download & extract artifacts.zip dari Google Drive kalau belum ada di disk lokal.
    Streamlit Cloud tidak punya akses Google Drive langsung seperti Colab, jadi
    artifacts (patchcore.pkl + config.json per kategori) diambil sekali di awal
    lalu disimpan ke disk container -- persist selama container tidak di-redeploy."""
    if os.path.isdir(ARTIFACT_DIR) and len(os.listdir(ARTIFACT_DIR)) > 0:
        return None  # sudah ada, skip

    if ARTIFACTS_DRIVE_FILE_ID == "GANTI_DENGAN_FILE_ID_DRIVE_KAMU":
        return ("File ID Google Drive belum diatur. Tambahkan ARTIFACTS_DRIVE_FILE_ID "
                "di Settings > Secrets aplikasi Streamlit Cloud kamu.")

    try:
        import gdown
    except ImportError:
        return "Package 'gdown' belum terpasang -- tambahkan ke requirements.txt."

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    zip_path = "/tmp/artifacts.zip"
    url = f"https://drive.google.com/uc?id={ARTIFACTS_DRIVE_FILE_ID}"
    try:
        gdown.download(url, zip_path, quiet=False)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall("/tmp")
        os.remove(zip_path)
    except Exception as e:
        return f"Gagal download/extract artifacts.zip: {e}"
    return None


# ============================================================
# MODEL LOADING (cached supaya tidak reload tiap interaksi)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_feature_extractor():
    fe = FeatureExtractor().to(DEVICE)
    fe.eval()
    return fe


@st.cache_resource(show_spinner=False)
def load_patchcore(category):
    fe = get_feature_extractor()
    cat_dir = os.path.join(ARTIFACT_DIR, category)
    with open(os.path.join(cat_dir, "patchcore.pkl"), "rb") as f:
        pc_data = pickle.load(f)
    patchcore = PatchCore(fe)
    patchcore.memory_bank = pc_data["memory_bank"]
    patchcore.nn_index = pc_data["nn_index"]
    patchcore.grid_shape = pc_data["grid_shape"]
    return patchcore


@st.cache_data(show_spinner=False)
def calibrate_thresholds(category, _patchcore, sample_size=CALIBRATION_SAMPLE_SIZE,
                          percentile_low=90, percentile_high=99.5):
    """Kalibrasi tau_low/tau_high dari skor PatchCore pada sample gambar NORMAL asli
    (bukan dari config.json hasil training, karena itu dikalibrasi untuk skor Hybrid
    yang skalanya berbeda dari skor PatchCore murni).
    percentile_low diturunkan ke 90 (dari 95) supaya ada jarak yang cukup lebar ke
    percentile_high -- zona "NEEDS MANUAL INSPECTION" jangan sampai nyaris tidak ada."""
    if DATA_ROOT is None:
        return None, None, "Kalibrasi otomatis tidak tersedia di deployment ini (dataset penuh tidak di-bundle). Atur threshold manual di sidebar."
    good_dir = os.path.join(DATA_ROOT, category, "train", "good")
    if not os.path.isdir(good_dir):
        # fallback kalau DATA_ROOT tidak tersedia di environment deployment ini
        return None, None, "DATA_ROOT tidak ditemukan - kalibrasi otomatis dilewati, atur threshold manual di sidebar."

    files = sorted(os.listdir(good_dir))[:sample_size]
    scores = []
    batch_size = 16
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        imgs = [Image.open(os.path.join(good_dir, fn)).convert("RGB") for fn in batch_files]
        x_batch = torch.cat([preprocess_pil_image(im) for im in imgs], dim=0)
        _, batch_scores = _patchcore.score(x_batch, DEVICE)
        scores.extend(batch_scores.tolist())
    tau_low = float(np.percentile(scores, percentile_low))
    tau_high = float(np.percentile(scores, percentile_high))
    return tau_low, tau_high, None


def get_available_categories():
    if not os.path.isdir(ARTIFACT_DIR):
        return []
    return sorted([d for d in os.listdir(ARTIFACT_DIR)
                   if os.path.isdir(os.path.join(ARTIFACT_DIR, d))
                   and os.path.exists(os.path.join(ARTIFACT_DIR, d, "patchcore.pkl"))])


# ============================================================
# VISUALISASI
# ============================================================
def make_result_figure(orig_rgb, anomaly_map, mask):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Gambar produk")
    axes[0].axis("off")

    axes[1].imshow(orig_rgb)
    axes[1].imshow(anomaly_map, cmap="inferno", alpha=0.5)
    axes[1].set_title("Anomaly heatmap")
    axes[1].axis("off")

    overlay = orig_rgb.copy().astype(float) / 255.0
    red = np.zeros_like(overlay)
    red[..., 0] = 1.0
    alpha_mask = np.stack([mask.astype(float)] * 3, axis=-1) * 0.55
    overlay = overlay * (1 - alpha_mask) + red * alpha_mask
    axes[2].imshow(overlay)
    axes[2].set_title("Defect region (post-processing)")
    axes[2].axis("off")

    plt.tight_layout()
    return fig


DECISION_COLOR = {
    "PASS": "#1a9850",
    "NEEDS MANUAL INSPECTION": "#fdae61",
    "REJECT": "#d73027",
}


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Smart Industrial Product Inspection", layout="wide")

# Tema warna aplikasi (semua diatur di sini, tidak pakai config.toml/folder .streamlit sama sekali)
st.markdown("""
<style>
    /* Latar & teks dasar aplikasi (area konten utama, terang + teks gelap) */
    .stApp { background-color: #f7fcff; color: #0b1b30; }
    .stApp * { color: #0b1b30; }

    /* Sembunyikan menu titik-tiga (kanan atas) -- HANYA elemen ini, tidak menyentuh
       elemen lain, supaya tombol panah sidebar tidak ikut hilang seperti sebelumnya */
    #MainMenu { display: none; }

    /* Sidebar (gelap + teks putih) - lebih spesifik jadi menimpa aturan di atas */
    [data-testid="stSidebar"] { background-color: #122638; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }

    /* Tombol panah collapse/expand sidebar -- paksa selalu kelihatan penuh,
       jangan pudar walau kursor jauh (biar jelas kelihatan pas direkam) */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    button[kind="header"] {
        opacity: 1 !important;
    }

    /* Kotak input (dropdown, angka) punya latar putih sendiri -- teks di DALAMNYA
       harus tetap gelap, bukan ikut aturan putih di atas */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="select"] * ,
    [data-testid="stSidebar"] [data-baseweb="input"] * {
        color: #0b1b30 !important;
    }

    /* Tombol & elemen aksen, senada warna sidebar -- termasuk semua elemen teks
       DI DALAM tombol (span/p/div), bukan cuma tombolnya sendiri */
    div.stButton > button,
    [data-testid="stFileUploader"] section button,
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stBaseButton-secondary"],
    [data-testid="stFileUploader"] button {
        background-color: #122638 !important;
        color: #FFFFFF !important;
        border-color: #122638 !important;
    }
    div.stButton > button *,
    [data-testid="stFileUploader"] button *,
    [data-testid="stBaseButton-secondary"] * {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

init_db()

st.title("Smart Industrial Product Inspection")
st.caption("Computer Vision Quality Control — Kelompok 1 — National AI & Deep Learning Acceleration Bootcamp")

with st.spinner("Menyiapkan model (download pertama kali bisa beberapa menit)..."):
    download_error = ensure_artifacts_available()
if download_error:
    st.error(download_error)
    st.stop()

available_categories = get_available_categories()

with st.sidebar:
    st.header("Konfigurasi")
    if not available_categories:
        st.error(f"Tidak ada artifact model ditemukan di:\n{ARTIFACT_DIR}")
        st.stop()

    category = st.selectbox("Kategori produk", available_categories)
    patchcore = load_patchcore(category)

    st.divider()
    st.subheader("Threshold QC")
    st.caption("Kalibrasi otomatis tidak tersedia di deployment ini (dataset penuh tidak di-bundle) — atur manual di bawah. Tips: upload 1 gambar dulu, lihat anomaly score yang keluar, baru sesuaikan tau_low/tau_high di sekitar angka itu.")
    tau_low_default, tau_high_default = 10.0, 20.0

    tau_low = st.number_input("tau_low (batas PASS)", value=float(round(tau_low_default, 3)), format="%.3f")
    tau_high = st.number_input("tau_high (batas REJECT)", value=float(round(tau_high_default, 3)), format="%.3f")
    area_reject_pct = st.slider("Batas luas area REJECT (%)", 0.5, 30.0, 5.0, 0.5)
    threshold_map = st.number_input("Threshold binarisasi mask", value=float(round(tau_low_default, 3)), format="%.3f",
                                     help="Nilai di atas ini dianggap bagian dari defect region pada mask.")

    st.divider()
    st.subheader("Info model")
    st.markdown(f"""
    - **Metode**: PatchCore (nearest-neighbor memory bank)
    - **Backbone**: WideResNet-50 (frozen, ImageNet pretrained)
    - **Memory bank**: {patchcore.memory_bank.shape[0]} patch, dim {patchcore.memory_bank.shape[1]}
    - **Device**: {DEVICE}
    """)

tab_inspect, tab_dashboard = st.tabs(["Inspeksi", "Dashboard"])

# ------------------------------------------------------------
# TAB 1: INSPEKSI
# ------------------------------------------------------------
with tab_inspect:
    uploaded_file = st.file_uploader("Upload foto produk", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        orig_rgb = get_display_image(pil_image)

        t0 = time.time()
        x = preprocess_pil_image(pil_image)
        anomaly_map, image_score = patchcore.score(x, DEVICE)
        anomaly_map, anomaly_score = anomaly_map[0], float(image_score[0])
        mask, regions, area_pct = postprocess_anomaly_map(anomaly_map, threshold_map)
        decision = qc_decision(anomaly_score, area_pct, tau_low, tau_high, area_reject_pct)
        confidence_pct = compute_confidence(anomaly_score, tau_low, tau_high)
        inference_time = time.time() - t0
        saved_image_path = save_uploaded_image(pil_image, category, uploaded_file.name)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = make_result_figure(orig_rgb, anomaly_map, mask)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.markdown(f"""
            <div style="padding:16px;border-radius:8px;background-color:{DECISION_COLOR[decision]}22;
                        border:2px solid {DECISION_COLOR[decision]};text-align:center;margin-bottom:16px;">
                <span style="font-size:14px;color:#555;">KEPUTUSAN QC</span><br>
                <span style="font-size:26px;font-weight:700;color:{DECISION_COLOR[decision]};">{decision}</span>
            </div>
            """, unsafe_allow_html=True)

            st.metric("Anomaly score", f"{anomaly_score:.4f}")
            st.metric("Confidence model", f"{confidence_pct:.1f}%")
            st.metric("Luas area anomali", f"{area_pct:.2f}%")
            st.metric("Jumlah region defect (object count)", len(regions))
            st.metric("Waktu inference", f"{inference_time*1000:.0f} ms")

            with st.expander("Detail region defect"):
                if regions:
                    for i, r in enumerate(regions):
                        st.write(f"Region {i+1}: {r['area_px']} px, bbox={r['bbox']}")
                else:
                    st.write("Tidak ada region defect terdeteksi.")

        insert_inspection({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "category": category,
            "filename": uploaded_file.name,
            "image_path": saved_image_path,
            "anomaly_score": anomaly_score,
            "confidence_pct": confidence_pct,
            "defect_area_pct": area_pct,
            "n_regions": len(regions),
            "decision": decision,
            "inference_time_sec": inference_time,
            "model_name": "PatchCore (wide_resnet50_2)",
        })
        st.success("Hasil inspeksi dan foto tersimpan ke riwayat (bisa dilihat dari device manapun).")

# ------------------------------------------------------------
# TAB 2: DASHBOARD
# ------------------------------------------------------------
with tab_dashboard:
    filter_category = st.selectbox("Filter kategori", ["Semua kategori"] + available_categories, key="dash_filter")
    df_hist = load_history(filter_category)

    if df_hist.empty:
        st.info("Belum ada riwayat inspeksi. Upload gambar di tab Inspeksi terlebih dahulu.")
    else:
        n_total = len(df_hist)
        n_pass = (df_hist["decision"] == "PASS").sum()
        n_inspect = (df_hist["decision"] == "NEEDS MANUAL INSPECTION").sum()
        n_reject = (df_hist["decision"] == "REJECT").sum()
        n_abnormal = n_inspect + n_reject
        defect_rate = 100.0 * n_abnormal / n_total

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total produk diperiksa", n_total)
        c2.metric("Normal (PASS)", n_pass)
        c3.metric("Abnormal (inspect/reject)", n_abnormal)
        c4.metric("Defect rate", f"{defect_rate:.1f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Rata-rata anomaly score", f"{df_hist['anomaly_score'].mean():.4f}")
        c6.metric("Rata-rata confidence model", f"{df_hist['confidence_pct'].mean():.1f}%")
        c7.metric("Rata-rata waktu inference", f"{df_hist['inference_time_sec'].mean()*1000:.0f} ms")
        c8.metric("Model digunakan", df_hist["model_name"].iloc[0])

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Distribusi keputusan QC")
            decision_counts = df_hist["decision"].value_counts()
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            colors = [DECISION_COLOR.get(d, "#999") for d in decision_counts.index]
            ax1.bar(decision_counts.index, decision_counts.values, color=colors)
            ax1.set_ylabel("Jumlah")
            plt.xticks(rotation=15)
            st.pyplot(fig1)
            plt.close(fig1)

        with col_b:
            st.subheader("Distribusi anomaly score")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.hist(df_hist["anomaly_score"], bins=20, color="#4575b4")
            ax2.set_xlabel("Anomaly score")
            ax2.set_ylabel("Jumlah gambar")
            st.pyplot(fig2)
            plt.close(fig2)

        st.subheader("Riwayat inspeksi")
        st.dataframe(df_hist.drop(columns=["image_path"]), use_container_width=True)

        st.subheader("Galeri foto (10 inspeksi terakhir)")
        recent = df_hist.head(10)
        cols = st.columns(5)
        for i, (_, row) in enumerate(recent.iterrows()):
            with cols[i % 5]:
                if row["image_path"] and os.path.exists(row["image_path"]):
                    st.image(row["image_path"], use_container_width=True)
                else:
                    st.caption("Foto tidak tersedia")
                st.markdown(f"<span style='color:{DECISION_COLOR.get(row['decision'],'#999')};font-weight:600;'>{row['decision']}</span>",
                            unsafe_allow_html=True)
                st.caption(f"{row['category']} • score {row['anomaly_score']:.2f}")

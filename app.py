from pathlib import Path
from html import escape
from io import BytesIO
import base64
import hashlib
import json
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except ImportError:
    skimage_ssim = None

try:
    import altair as alt
except ImportError:
    alt = None

try:
    import torchxrayvision as xrv
except ImportError:
    xrv = None

try:
    import cooler
except ImportError:
    cooler = None


APP_TITLE = "BioMedVisionLab"
APP_CAPTION = "Biomedical Imaging Visualization Toolkit for Retrieval and Super-Resolution"
EMBEDDINGS_DIR = Path("embeddings")
LEGACY_IMAGE_PATHS_FILE = EMBEDDINGS_DIR / "image_paths.npy"
LEGACY_LABELS_FILE = EMBEDDINGS_DIR / "labels.npy"
LEGACY_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.npy"
ENCODER_CONFIGS = {
    "resnet18_imagenet": {
        "label": "ResNet18 ImageNet baseline",
        "folder": "resnet18_imagenet",
        "legacy_folders": ["resnet18"],
        "dimension": 512,
    },
    "resnet50_imagenet": {
        "label": "ResNet50 ImageNet baseline",
        "folder": "resnet50_imagenet",
        "legacy_folders": [],
        "dimension": 2048,
    },
    "torchxrayvision_densenet121": {
        "label": "TorchXRayVision DenseNet121 CXR encoder",
        "folder": "torchxrayvision_densenet121",
        "legacy_folders": ["torchxrayvision"],
        "dimension": 1024,
    },
}
LEGACY_ENCODER_KEYS = {
    "resnet18": "resnet18_imagenet",
    "resnet50": "resnet50_imagenet",
    "torchxrayvision": "torchxrayvision_densenet121",
}
CSV_HELPER_TEXT = (
    "Single reports describe the current selected example. Batch benchmarks export many rows "
    "for experiment tracking and method comparison."
)
SR_ARTIFACT_WARNING = (
    "Bicubic interpolation can smooth fine structures and does not recover true missing anatomical detail."
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def inject_css() -> None:
    """Add a polished biomedical research dashboard style."""
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 8% 0%, rgba(20, 184, 166, 0.12), transparent 25rem),
                radial-gradient(circle at 88% 5%, rgba(59, 130, 246, 0.10), transparent 25rem),
                linear-gradient(180deg, #f8fafc 0%, #eef3f8 100%);
            color: #0f172a;
        }

        .block-container {
            max-width: 1400px;
            padding-top: 0.9rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e5e7eb;
        }

        .hero {
            padding: 0.95rem 1.2rem;
            border: 1px solid #c7d2fe;
            border-radius: 16px;
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.96) 0%, rgba(239, 246, 255, 0.94) 46%, rgba(236, 253, 245, 0.92) 100%);
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.10);
            margin-bottom: 0.65rem;
            position: relative;
            overflow: hidden;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 18rem;
            height: 18rem;
            right: -8rem;
            top: -10rem;
            border-radius: 999px;
            background: rgba(14, 165, 233, 0.10);
        }

        .hero-top {
            display: flex;
            gap: 0.45rem;
            align-items: center;
            margin-bottom: 0.45rem;
        }

        .hero h1 {
            margin: 0;
            color: #0f172a;
            font-size: 2.1rem;
            line-height: 1.05;
            letter-spacing: 0;
        }

        .hero .subtitle {
            margin-top: 0.4rem;
            color: #2563eb;
            font-weight: 650;
            font-size: 1.02rem;
        }

        .hero .description {
            margin-top: 0.45rem;
            color: #475569;
            max-width: 900px;
            font-size: 0.96rem;
        }

        .badge-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 0.65rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            width: fit-content;
            border-radius: 999px;
            padding: 0.24rem 0.68rem;
            font-size: 0.78rem;
            font-weight: 760;
            line-height: 1.15;
            border: 1px solid #94a3b8;
            color: #334155;
            background: #f8fafc;
        }

        .badge-research {
            color: #1d4ed8;
            background: #eff6ff;
            border-color: #60a5fa;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }

        .badge-ai {
            color: #0f766e;
            background: #f0fdfa;
            border-color: #2dd4bf;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }

        .badge-imaging {
            color: #0e7490;
            background: #ecfeff;
            border-color: #22d3ee;
        }

        .badge-retrieval {
            color: #3730a3;
            background: #eef2ff;
            border-color: #818cf8;
        }

        .badge-sr {
            color: #6b21a8;
            background: #faf5ff;
            border-color: #c084fc;
        }

        .badge-xai {
            color: #92400e;
            background: #fffbeb;
            border-color: #f59e0b;
        }

        .badge-prototype {
            color: #334155;
            background: #f8fafc;
            border-color: #94a3b8;
        }

        .badge-high {
            color: #065f46;
            background: #ecfdf5;
            border-color: #34d399;
        }

        .badge-medium {
            color: #92400e;
            background: #fffbeb;
            border-color: #f59e0b;
        }

        .badge-low {
            color: #991b1b;
            background: #fef2f2;
            border-color: #f87171;
        }

        .badge-normal {
            color: #047857;
            background: #ecfdf5;
            border-color: #34d399;
        }

        .badge-pneumonia {
            color: #be123c;
            background: #fff1f2;
            border-color: #fb7185;
        }

        .badge-upload {
            color: #4338ca;
            background: #eef2ff;
            border-color: #818cf8;
        }

        .badge-match {
            color: #047857;
            background: #ecfdf5;
            border-color: #34d399;
        }

        .badge-different {
            color: #be123c;
            background: #fff1f2;
            border-color: #fb7185;
        }

        .badge-unknown {
            color: #475569;
            background: #f8fafc;
            border-color: #94a3b8;
        }

        .badge-encoder {
            color: #1d4ed8;
            background: #eff6ff;
            border-color: #60a5fa;
        }

        .badge-text {
            color: #7c2d12;
            background: #fff7ed;
            border-color: #fb923c;
        }

        .badge-micro {
            color: #0f766e;
            background: #f0fdfa;
            border-color: #2dd4bf;
        }

        .badge-genomics {
            color: #4338ca;
            background: #eef2ff;
            border-color: #818cf8;
        }

        .section-title {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 800;
            margin: 0 0 0.55rem 0;
        }

        .section-subtitle {
            color: #475569;
            font-size: 0.92rem;
            margin: -0.25rem 0 0.75rem 0;
        }

        .muted {
            color: #64748b;
            font-size: 0.9rem;
        }

        .disclaimer {
            padding: 0.62rem 0.9rem;
            border: 1px solid #bfdbfe;
            border-left: 4px solid #2563eb;
            border-radius: 12px;
            background: rgba(239, 246, 255, 0.88);
            color: #1e3a8a;
            font-size: 0.9rem;
            margin: 0.6rem 0 0.75rem 0;
        }

        .info-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
        }

        .info-table {
            width: 100%;
        }

        .info-row {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.75rem 0.62rem;
            border-bottom: 1px solid #e8eef6;
            align-items: center;
        }

        .info-row:last-child {
            border-bottom: 0;
        }

        .info-label {
            color: #64748b;
            font-size: 0.82rem;
            white-space: nowrap;
            flex: 0 0 auto;
        }

        .info-value {
            color: #0f172a;
            font-size: 0.86rem;
            font-weight: 700;
            text-align: right;
            overflow-wrap: anywhere;
            flex: 1 1 auto;
            min-width: 0;
        }

        .alignment-table .info-row {
            display: grid;
            grid-template-columns: minmax(145px, 0.34fr) minmax(0, 1fr);
            gap: 1rem;
            align-items: start;
        }

        .alignment-table .info-label {
            white-space: normal;
            line-height: 1.35;
        }

        .alignment-table .info-value {
            text-align: left;
            font-weight: 650;
            line-height: 1.4;
            overflow-wrap: normal;
        }

        @media (max-width: 760px) {
            .alignment-table .info-row {
                grid-template-columns: 1fr;
                gap: 0.25rem;
            }
        }

        .card {
            min-width: 0;
            overflow: visible;
        }

        .metric-grid-5 {
            display: grid;
            grid-template-columns: repeat(5, minmax(150px, 1fr));
            gap: 1rem;
        }

        .metric-grid-4 {
            display: grid;
            grid-template-columns: repeat(4, minmax(160px, 1fr));
            gap: 1rem;
        }

        .metric-grid-3 {
            display: grid;
            grid-template-columns: repeat(3, minmax(180px, 1fr));
            gap: 1rem;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.04);
            min-width: 0;
            overflow: visible;
        }

        .metric-label {
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 650;
            margin-bottom: 0.35rem;
            white-space: nowrap;
        }

        .big-value {
            color: #0f172a;
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1.1;
            white-space: nowrap;
        }

        .big-value.long {
            font-size: 1.05rem;
            line-height: 1.25;
            white-space: normal;
            overflow-wrap: anywhere;
        }

        @media (max-width: 1100px) {
            .metric-grid-5,
            .metric-grid-4,
            .metric-grid-3 {
                grid-template-columns: repeat(2, minmax(180px, 1fr));
            }
        }

        @media (max-width: 700px) {
            .metric-grid-5,
            .metric-grid-4,
            .metric-grid-3 {
                grid-template-columns: 1fr;
            }
        }

        .confidence-card {
            padding: 0.7rem 0.8rem;
            border-radius: 14px;
            border: 1px solid #e5e7eb;
            background: #f8fafc;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.045);
        }

        .confidence-note {
            color: #475569;
            font-size: 0.82rem;
            margin-top: 0.45rem;
        }

        .image-frame {
            height: 245px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 0.45rem;
        }

        .image-frame.compact {
            height: 170px;
        }

        .image-frame.crop {
            height: 210px;
        }

        .image-frame.large {
            height: 420px;
        }

        .image-frame img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .result-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.055);
            padding: 0.65rem;
            margin-bottom: 0.75rem;
        }

        .result-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.45rem;
        }

        .rank-pill {
            color: #1e40af;
            background: #dbeafe;
            border: 1px solid #bfdbfe;
            border-radius: 999px;
            padding: 0.16rem 0.5rem;
            font-size: 0.76rem;
            font-weight: 750;
        }

        .mini-row {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            border-top: 1px solid #f1f5f9;
            padding: 0.32rem 0;
            font-size: 0.82rem;
        }

        .mini-row span:first-child {
            color: #64748b;
        }

        .mini-row span:last-child {
            color: #0f172a;
            font-weight: 650;
            text-align: right;
            overflow-wrap: anywhere;
        }

        .thin-progress {
            height: 6px;
            border-radius: 999px;
            background: #e2e8f0;
            overflow: hidden;
            margin-top: 0.45rem;
        }

        .thin-progress div {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #2563eb, #14b8a6);
        }

        .compact-note {
            color: #64748b;
            font-size: 0.86rem;
            margin: 0.15rem 0 0.65rem 0;
        }

        .artifact-card {
            border: 1px solid #f59e0b;
            background: #fffbeb;
            color: #78350f;
            border-radius: 14px;
            padding: 0.75rem 0.85rem;
            font-size: 0.9rem;
            font-weight: 600;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 0.62rem 0.72rem;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stMetricLabel"] {
            color: #64748b;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.045);
            background: #ffffff;
            border-color: #e2e8f0;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }

        div[data-testid="stDataFrame"] [role="columnheader"] {
            font-weight: 700;
        }

        [data-baseweb="select"] *,
        [data-baseweb="popover"] *,
        [data-baseweb="menu"] *,
        [data-baseweb="menu"] li,
        [data-baseweb="menu"] [role="option"],
        [role="listbox"] *,
        [role="option"],
        .stSelectbox *,
        .stRadio *,
        .stSlider *,
        .stButton button,
        .stDownloadButton button,
        .stFileUploader button,
        .stTabs [role="tab"],
        .stCheckbox * {
            cursor: pointer !important;
        }

        .stProgress > div > div > div > div {
            background-color: #2563eb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def canonical_encoder_name(encoder_name: str) -> str:
    return LEGACY_ENCODER_KEYS.get(encoder_name, encoder_name)


def get_image_transform() -> transforms.Compose:
    """Create the same preprocessing pipeline used while building the index."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_xrv_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: (tensor * 2048.0) - 1024.0),
        ]
    )


@st.cache_resource
def load_encoder(encoder_name: str = "resnet18_imagenet") -> nn.Module:
    """Load the selected encoder for uploaded-image query embeddings."""
    encoder_name = canonical_encoder_name(encoder_name)
    if encoder_name == "torchxrayvision_densenet121":
        if xrv is None:
            raise ImportError("TorchXRayVision is not installed. Falling back to ResNet18.")
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model.op_threshs = None
        model.eval()
        return model

    if encoder_name == "resnet50_imagenet":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    return model


def load_feature_extractor(encoder_key: str = "resnet18_imagenet") -> nn.Module:
    return load_encoder(encoder_key)


def preprocess_image_for_encoder(image: Image.Image, encoder_name: str = "resnet18_imagenet") -> torch.Tensor:
    encoder_name = canonical_encoder_name(encoder_name)
    transform = get_xrv_transform() if encoder_name == "torchxrayvision_densenet121" else get_image_transform()
    return transform(image).unsqueeze(0)


@st.cache_data
def load_index(index_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load image paths, labels, and L2-normalized embeddings."""
    index_path = Path(index_dir)
    image_paths = np.load(index_path / "image_paths.npy", allow_pickle=True)
    labels = np.load(index_path / "labels.npy", allow_pickle=True)
    embeddings = np.load(index_path / "embeddings.npy")
    metadata_path = index_path / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {
            "encoder_key": "resnet18_imagenet",
            "encoder_name": "ResNet18 ImageNet baseline",
            "embedding_dimension": int(embeddings.shape[1]),
            "dataset_size": len(image_paths),
            "build_time": "Legacy index",
            "device": "Unknown",
        }
    metadata["encoder_key"] = canonical_encoder_name(str(metadata.get("encoder_key", "resnet18_imagenet")))
    metadata["encoder_name"] = str(
        ENCODER_CONFIGS.get(metadata["encoder_key"], {}).get("label", metadata.get("encoder_name", "Unknown encoder"))
    )
    return image_paths, labels, embeddings, metadata


@st.cache_data
def load_index_manifest(index_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load lightweight index data for tabs that do not need embeddings."""
    index_path = Path(index_dir)
    image_paths = np.load(index_path / "image_paths.npy", allow_pickle=True)
    labels = np.load(index_path / "labels.npy", allow_pickle=True)
    return image_paths, labels


def index_files_exist(index_dir: Path) -> bool:
    return (
        (index_dir / "image_paths.npy").exists()
        and (index_dir / "labels.npy").exists()
        and (index_dir / "embeddings.npy").exists()
    )


def discover_embedding_indexes() -> dict[str, Path]:
    indexes = {}
    if index_files_exist(EMBEDDINGS_DIR):
        indexes["ResNet18 ImageNet baseline (legacy)"] = EMBEDDINGS_DIR

    for child in sorted(EMBEDDINGS_DIR.iterdir()) if EMBEDDINGS_DIR.exists() else []:
        if not child.is_dir() or not index_files_exist(child):
            continue
        metadata_path = child / "metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                label = str(metadata.get("encoder_name", child.name))
            except (OSError, json.JSONDecodeError):
                label = child.name
        else:
            label = child.name
        indexes[label] = child
    return indexes


def encoder_index_dir(encoder_name: str) -> Path | None:
    encoder_name = canonical_encoder_name(encoder_name)
    config = ENCODER_CONFIGS.get(encoder_name)
    if config is None:
        return None

    candidates = [EMBEDDINGS_DIR / str(config["folder"])]
    candidates.extend(EMBEDDINGS_DIR / folder for folder in config["legacy_folders"])
    if encoder_name == "resnet18_imagenet":
        candidates.append(EMBEDDINGS_DIR)

    for candidate in candidates:
        if index_files_exist(candidate):
            return candidate
    return None


def available_encoder_indexes() -> dict[str, Path]:
    indexes = {}
    for encoder_name, config in ENCODER_CONFIGS.items():
        index_dir = encoder_index_dir(encoder_name)
        if index_dir is not None:
            indexes[str(config["label"])] = index_dir
    return indexes


def load_rgb_image(image_source) -> Image.Image:
    """Open an image and convert grayscale biomedical images to 3-channel RGB."""
    return Image.open(image_source).convert("RGB")


def resize_for_display(image: Image.Image, max_size: int = 512) -> Image.Image:
    image = image.convert("RGB")
    if max(image.size) <= max_size:
        return image
    resized = image.copy()
    resized.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return resized


@st.cache_data(show_spinner=False)
def load_display_image_cached(image_path: str, max_size: int = 512) -> Image.Image | None:
    try:
        return resize_for_display(load_rgb_image(Path(image_path)), max_size=max_size)
    except (OSError, FileNotFoundError, UnidentifiedImageError):
        return None


def safe_load_image(image_path: Path) -> Image.Image | None:
    return load_display_image_cached(str(image_path))


@st.cache_data(show_spinner=False)
def load_uploaded_rgb_image_cached(file_name: str, file_size: int, file_digest: str, image_bytes: bytes) -> Image.Image:
    return load_rgb_image(BytesIO(image_bytes))


def extract_embedding(image: Image.Image, encoder_name: str = "resnet18_imagenet") -> np.ndarray:
    """Convert one PIL image into a L2-normalized embedding."""
    encoder_name = canonical_encoder_name(encoder_name)
    model = load_encoder(encoder_name)
    image_tensor = preprocess_image_for_encoder(image, encoder_name)

    with torch.no_grad():
        if encoder_name == "torchxrayvision_densenet121":
            features = model.features(image_tensor)
            features = torch.nn.functional.relu(features, inplace=False)
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            embedding = torch.flatten(features, 1).cpu().numpy()[0]
        else:
            embedding = model(image_tensor).cpu().numpy()[0]

    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


@st.cache_data(show_spinner=False)
def extract_uploaded_embedding_cached(file_name: str, file_size: int, file_digest: str, image_bytes: bytes, encoder_key: str) -> np.ndarray:
    image = load_rgb_image(BytesIO(image_bytes))
    return extract_embedding(image, encoder_key)


def encoder_options_for_uploads() -> dict[str, str]:
    return {
        "ResNet18 ImageNet baseline": "resnet18_imagenet",
        "ResNet50 ImageNet baseline": "resnet50_imagenet",
        "TorchXRayVision DenseNet121 CXR encoder": "torchxrayvision_densenet121",
    }


def embed_uploaded_bytes_with_fallback(
    file_name: str,
    image_bytes: bytes,
    encoder_key: str,
) -> tuple[np.ndarray, str, str | None]:
    digest = hashlib.sha256(image_bytes).hexdigest()
    try:
        embedding = extract_uploaded_embedding_cached(file_name, len(image_bytes), digest, image_bytes, encoder_key)
        return embedding, encoder_key, None
    except Exception as error:
        if canonical_encoder_name(encoder_key) == "resnet18_imagenet":
            raise
        embedding = extract_uploaded_embedding_cached(file_name, len(image_bytes), digest, image_bytes, "resnet18_imagenet")
        return embedding, "resnet18_imagenet", f"{error} Using ResNet18 ImageNet baseline for uploaded-image retrieval."


def retrieve_similar(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    top_k: int,
    query_index: int | None = None,
) -> pd.DataFrame:
    """Retrieve top-k results using cosine similarity on normalized embeddings."""
    similarities = embeddings @ query_embedding

    if query_index is not None:
        similarities[query_index] = -np.inf

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(top_indices) + 1),
            "index": top_indices,
            "similarity": similarities[top_indices],
        }
    )


def calculate_mse(original: Image.Image, enhanced: Image.Image) -> float:
    """Calculate mean squared error between two RGB images."""
    original_array = np.asarray(original.convert("RGB"), dtype=np.float32)
    enhanced_array = np.asarray(enhanced.convert("RGB"), dtype=np.float32)
    return float(np.mean((original_array - enhanced_array) ** 2))


def calculate_psnr(original: Image.Image, enhanced: Image.Image) -> float:
    """Calculate peak signal-to-noise ratio between two RGB images."""
    mse = calculate_mse(original, enhanced)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def calculate_ssim(original: Image.Image, enhanced: Image.Image) -> float | None:
    """Calculate SSIM if scikit-image is installed."""
    if skimage_ssim is None:
        return None

    original_array = np.asarray(original.convert("L"), dtype=np.float32)
    enhanced_array = np.asarray(enhanced.convert("L"), dtype=np.float32)
    return float(skimage_ssim(original_array, enhanced_array, data_range=255.0))


def simulate_low_resolution(image: Image.Image, scale_factor: int) -> tuple[Image.Image, Image.Image]:
    """Downsample an image, then reconstruct it with bicubic interpolation."""
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    low_size = (max(1, width // scale_factor), max(1, height // scale_factor))
    low_res = rgb_image.resize(low_size, Image.Resampling.BICUBIC)
    enhanced = low_res.resize((width, height), Image.Resampling.BICUBIC)
    return low_res, enhanced


def center_crop(image: Image.Image, crop_size: int) -> Image.Image:
    """Crop the center square of an image."""
    width, height = image.size
    crop_size = int(max(1, min(crop_size, width, height)))
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def center_crop_low_res(low_res: Image.Image, original_size: tuple[int, int], crop_size: int) -> Image.Image:
    """Crop the center region from the low-res image using original-image scale."""
    original_width, original_height = original_size
    scale_x = low_res.size[0] / max(1, original_width)
    scale_y = low_res.size[1] / max(1, original_height)
    low_crop_size = int(max(1, min(low_res.size) if crop_size >= min(original_size) else crop_size * min(scale_x, scale_y)))
    return center_crop(low_res, low_crop_size).resize((crop_size, crop_size), Image.Resampling.NEAREST)


def image_to_data_uri(image: Image.Image, max_size: int = 512) -> str:
    """Convert a PIL image to a PNG data URI for styled HTML image frames."""
    image = resize_for_display(image, max_size=max_size)
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def image_frame_html(image: Image.Image, css_class: str = "") -> str:
    class_name = f"image-frame {css_class}".strip()
    max_size = 768 if "large" in css_class else 384 if ("compact" in css_class or "crop" in css_class) else 512
    return f'<div class="{class_name}"><img src="{image_to_data_uri(image, max_size=max_size)}" alt="Biomedical image"></div>'


def key_value_rows(rows: list[tuple[str, str]]) -> str:
    row_html = "".join(
        '<div class="info-row">'
        f'<span class="info-label">{escape(str(key))}</span>'
        f'<span class="info-value" title="{escape(str(value))}">{escape(str(value))}</span>'
        "</div>"
        for key, value in rows
    )
    return f'<div class="info-grid info-table">{row_html}</div>'


def alignment_rows(rows: list[tuple[str, str]]) -> str:
    row_html = "".join(
        '<div class="info-row">'
        f'<span class="info-label">{escape(str(key))}</span>'
        f'<span class="info-value">{escape(str(value))}</span>'
        "</div>"
        for key, value in rows
    )
    return f'<div class="info-grid info-table alignment-table">{row_html}</div>'


def metric_grid(items: list[tuple], grid_class: str) -> str:
    card_html = []
    for item in items:
        label = str(item[0])
        value = str(item[1])
        title = str(item[2]) if len(item) > 2 else value
        value_class = "big-value long" if len(value) > 16 or "<span" in value else "big-value"
        card_html.append(
            '<div class="metric-card card">'
            f'<div class="metric-label" title="{escape(label)}">{escape(label)}</div>'
            f'<div class="{value_class}" title="{escape(title)}">{value}</div>'
            "</div>"
        )
    cards = "".join(card_html)
    return f'<div class="{grid_class}">{cards}</div>'


def filename_with_tooltip(filename: str, max_length: int = 28) -> str:
    return f'<span title="{escape(filename)}">{escape(shorten_filename(filename, max_length=max_length))}</span>'


def create_intensity_heatmap(image: Image.Image) -> Image.Image:
    """Create a calm viridis-style intensity visualization from grayscale pixels."""
    gray = np.asarray(image.convert("L").resize(image.size), dtype=np.float32) / 255.0
    palette = np.array(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=np.float32,
    )
    scaled = gray * (len(palette) - 1)
    low = np.floor(scaled).astype(int)
    high = np.clip(low + 1, 0, len(palette) - 1)
    mix = (scaled - low)[..., None]
    heatmap = palette[low] * (1.0 - mix) + palette[high] * mix
    return Image.fromarray(np.clip(heatmap, 0, 255).astype(np.uint8), mode="RGB")


def label_badge(label: str) -> str:
    label_text = str(label).upper()
    if label_text == "NORMAL":
        badge_class = "badge badge-normal"
    elif label_text == "PNEUMONIA":
        badge_class = "badge badge-pneumonia"
    else:
        badge_class = "badge badge-upload"
    return f'<span class="{badge_class}">{label_text}</span>'


def shorten_filename(filename: str, max_length: int = 28) -> str:
    if len(filename) <= max_length:
        return filename
    return f"{filename[:16]}...{filename[-9:]}"


def label_agreement(query_label: str, result_label: str, query_index: int | None) -> str:
    if query_index is None:
        return "Unknown"
    return "Match" if str(query_label) == str(result_label) else "Different"


def agreement_badge(agreement: str) -> str:
    badge_class = {
        "Match": "badge badge-match",
        "Different": "badge badge-different",
        "Unknown": "badge badge-unknown",
    }.get(agreement, "badge badge-unknown")
    return f'<span class="{badge_class}">{escape(agreement)}</span>'


def absolute_difference_map(query_image: Image.Image, retrieved_image: Image.Image) -> Image.Image:
    query_gray = np.asarray(query_image.convert("L").resize((224, 224)), dtype=np.float32)
    retrieved_gray = np.asarray(retrieved_image.convert("L").resize((224, 224)), dtype=np.float32)
    difference = np.abs(query_gray - retrieved_gray)
    max_value = float(difference.max())
    if max_value > 0:
        difference = difference / max_value
    heatmap = np.zeros((*difference.shape, 3), dtype=np.float32)
    heatmap[..., 0] = np.clip(difference * 1.7, 0.0, 1.0)
    heatmap[..., 1] = np.clip(difference * 0.9, 0.0, 1.0)
    heatmap[..., 2] = np.clip(0.15 + difference * 0.25, 0.0, 1.0)
    return Image.fromarray((heatmap * 255).astype(np.uint8), mode="RGB")


def matrix_to_heatmap_image(matrix: np.ndarray, colormap: str = "viridis") -> Image.Image:
    matrix = np.asarray(matrix, dtype=np.float32)
    min_value = float(matrix.min())
    max_value = float(matrix.max())
    normalized = (matrix - min_value) / max(1e-8, max_value - min_value)
    palettes = {
        "viridis": [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        "magma": [
            [0, 0, 4],
            [80, 18, 123],
            [182, 54, 121],
            [251, 136, 97],
            [252, 253, 191],
        ],
        "inferno": [
            [0, 0, 4],
            [87, 15, 109],
            [187, 55, 84],
            [249, 142, 8],
            [252, 255, 164],
        ],
    }
    palette = np.array(palettes.get(colormap, palettes["viridis"]), dtype=np.float32)
    scaled = normalized * (len(palette) - 1)
    low = np.floor(scaled).astype(int)
    high = np.clip(low + 1, 0, len(palette) - 1)
    mix = (scaled - low)[..., None]
    heatmap = palette[low] * (1.0 - mix) + palette[high] * mix
    return Image.fromarray(np.clip(heatmap, 0, 255).astype(np.uint8), mode="RGB")


def generate_synthetic_contact_map(size: int, noise_level: float) -> np.ndarray:
    rng = np.random.default_rng(42)
    coords = np.arange(size, dtype=np.float32)
    distance = np.abs(coords[:, None] - coords[None, :])
    matrix = np.exp(-distance / max(4.0, size / 14.0))

    loop_specs = [
        (0.25, 0.58, 0.95),
        (0.38, 0.74, 0.75),
        (0.62, 0.84, 0.65),
    ]
    sigma = max(1.8, size / 42.0)
    x_grid, y_grid = np.meshgrid(coords, coords)
    for left_frac, right_frac, strength in loop_specs:
        left = left_frac * (size - 1)
        right = right_frac * (size - 1)
        spot = np.exp(-(((x_grid - left) ** 2 + (y_grid - right) ** 2) / (2 * sigma**2)))
        mirror = np.exp(-(((x_grid - right) ** 2 + (y_grid - left) ** 2) / (2 * sigma**2)))
        matrix += strength * (spot + mirror)

    if noise_level > 0:
        matrix += rng.normal(0.0, noise_level, size=(size, size)).astype(np.float32)

    matrix = np.clip(matrix, 0.0, None)
    return matrix / max(1e-8, float(matrix.max()))


def simulate_low_resolution_matrix(matrix: np.ndarray, scale_factor: int) -> tuple[np.ndarray, np.ndarray]:
    image = Image.fromarray(np.clip(matrix * 255, 0, 255).astype(np.uint8), mode="L")
    low_size = (max(1, matrix.shape[1] // scale_factor), max(1, matrix.shape[0] // scale_factor))
    low_image = image.resize(low_size, Image.Resampling.BICUBIC)
    upscaled_image = low_image.resize((matrix.shape[1], matrix.shape[0]), Image.Resampling.BICUBIC)
    return np.asarray(low_image, dtype=np.float32) / 255.0, np.asarray(upscaled_image, dtype=np.float32) / 255.0


def calculate_matrix_metrics(original: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float, float | None]:
    mse = float(np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2))
    psnr = float("inf") if mse == 0 else float(20 * np.log10(1.0 / np.sqrt(mse)))
    ssim = None
    if skimage_ssim is not None:
        min_side = int(min(original.shape))
        if min_side >= 3:
            win_size = min(7, min_side)
            if win_size % 2 == 0:
                win_size -= 1
            ssim = float(
                skimage_ssim(
                    original.astype(np.float32),
                    reconstructed.astype(np.float32),
                    data_range=1.0,
                    win_size=win_size,
                )
            )
    return mse, psnr, ssim


def sanitize_contact_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Contact matrix must be 2D.")
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(matrix, 0.0, None)


def minmax_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = sanitize_contact_matrix(matrix)
    min_value = float(matrix.min())
    max_value = float(matrix.max())
    if max_value <= min_value:
        return np.zeros_like(matrix, dtype=np.float32)
    return ((matrix - min_value) / (max_value - min_value)).astype(np.float32)


def prepare_contact_matrix_for_visualization(matrix: np.ndarray, use_log1p: bool) -> np.ndarray:
    matrix = sanitize_contact_matrix(matrix)
    if use_log1p:
        matrix = np.log1p(matrix)
    return minmax_normalize_matrix(matrix)


def crop_matrix_to_square(matrix: np.ndarray) -> np.ndarray:
    size = int(min(matrix.shape))
    return matrix[:size, :size]


def cooler_groups_for_file(file_path: str) -> list[str]:
    if cooler is None:
        return []
    try:
        return list(cooler.fileops.list_coolers(file_path))
    except Exception:
        return []


def save_uploaded_file_to_temp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        return temp_file.name


def open_cooler_from_temp_path(temp_path: str, group: str | None = None):
    if cooler is None:
        raise ImportError("Install cooler to upload .cool or .mcool Hi-C contact maps.")

    suffix = Path(temp_path).suffix.lower()
    groups = cooler_groups_for_file(temp_path)
    uri = temp_path
    if suffix == ".mcool":
        if not groups:
            raise ValueError("No cooler groups were found in this .mcool file.")
        selected_group = group or groups[0]
        uri = f"{temp_path}::{selected_group}"
    elif group:
        uri = f"{temp_path}::{group}"

    return cooler.Cooler(uri), groups


def load_cooler_contact_matrix(
    uploaded_file,
    group: str | None,
    region: str | None,
    max_matrix_size: int,
) -> tuple[np.ndarray, list[str], list[str], str | None]:
    temp_path = save_uploaded_file_to_temp(uploaded_file)
    try:
        cooler_obj, groups = open_cooler_from_temp_path(temp_path, group)
        regions = cooler_region_options(cooler_obj)
        selected_region = region or (regions[0] if regions else None)
        matrix = fetch_cooler_matrix(cooler_obj, selected_region, max_matrix_size)
        return matrix, groups, regions, selected_region
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except OSError:
            pass


def cooler_region_options(cooler_obj, max_options: int = 30) -> list[str]:
    chromnames = list(getattr(cooler_obj, "chromnames", []))
    return chromnames[:max_options]


def fetch_cooler_matrix(cooler_obj, region: str | None, max_matrix_size: int) -> np.ndarray:
    chromsizes = getattr(cooler_obj, "chromsizes", None)
    if region and chromsizes is not None:
        region_length = int(chromsizes[region])
        binsize = int(getattr(cooler_obj, "binsize", 1) or 1)
        span = max_matrix_size * binsize
        end = min(region_length, span)
        fetch_region = f"{region}:0-{end}"
    elif region:
        fetch_region = region
    else:
        chromnames = list(getattr(cooler_obj, "chromnames", []))
        if not chromnames:
            raise ValueError("No chromosomes or regions were found in this cooler file.")
        fetch_region = chromnames[0]

    try:
        matrix = cooler_obj.matrix(balance=True).fetch(fetch_region)
    except Exception:
        matrix = cooler_obj.matrix(balance=False).fetch(fetch_region)

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Selected cooler matrix slice is not 2D.")
    if max(matrix.shape) > max_matrix_size:
        matrix = matrix[:max_matrix_size, :max_matrix_size]
    if max(matrix.shape) > max_matrix_size:
        raise ValueError("Please select a smaller region or lower display size.")
    return sanitize_contact_matrix(matrix)


def load_uploaded_contact_matrix(uploaded_file) -> np.ndarray:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".npy":
        matrix = np.load(BytesIO(uploaded_file.getvalue()))
    elif suffix == ".csv":
        matrix = pd.read_csv(uploaded_file, header=None).values
    else:
        raise ValueError("Upload a .npy or .csv contact matrix.")
    try:
        return sanitize_contact_matrix(matrix)
    except (TypeError, ValueError) as error:
        raise ValueError("Contact matrix values must be numeric.") from error


def diagonal_band_profile(matrix: np.ndarray, max_distance: int = 40) -> pd.DataFrame:
    matrix = np.asarray(matrix, dtype=np.float32)
    max_distance = int(min(max_distance, max(0, matrix.shape[0] - 1)))
    rows = []
    for distance in range(max_distance + 1):
        values = np.diagonal(matrix, offset=distance)
        rows.append({"Distance from diagonal": distance, "Mean signal": float(values.mean()) if len(values) else 0.0})
    return pd.DataFrame(rows)


def diagonal_preservation_score(original: np.ndarray, reconstructed: np.ndarray, max_distance: int = 20) -> float:
    original_profile = diagonal_band_profile(original, max_distance=max_distance)["Mean signal"].to_numpy(dtype=np.float32)
    reconstructed_profile = diagonal_band_profile(reconstructed, max_distance=max_distance)["Mean signal"].to_numpy(dtype=np.float32)
    error = float(np.mean(np.abs(original_profile - reconstructed_profile)))
    scale = float(np.mean(np.abs(original_profile))) + 1e-8
    return float(max(0.0, 1.0 - error / scale))


def calculate_retrieval_quality(
    results: pd.DataFrame,
    labels: np.ndarray,
    query_label: str,
    query_index: int | None,
) -> dict[str, float | int | str | None]:
    similarities = results["similarity"].astype(float)
    mean_similarity = float(similarities.mean()) if len(similarities) else 0.0
    top_similarity = float(similarities.iloc[0]) if len(similarities) else 0.0

    if query_index is None:
        return {
            "agreement_count": None,
            "precision_at_k": None,
            "mean_similarity": mean_similarity,
            "top_similarity": top_similarity,
            "note": "Label agreement unavailable for uploaded image",
        }

    result_labels = get_result_labels(results, labels)
    agreement_count = sum(1 for result_label in result_labels if str(result_label) == str(query_label))
    precision_at_k = agreement_count / max(1, len(results))
    return {
        "agreement_count": agreement_count,
        "precision_at_k": precision_at_k,
        "mean_similarity": mean_similarity,
        "top_similarity": top_similarity,
        "note": None,
    }


def retrieval_confidence_label(quality: dict[str, float | int | str | None], uploaded: bool) -> str:
    top_similarity = float(quality["top_similarity"] or 0.0)
    precision_at_k = quality["precision_at_k"]

    if uploaded:
        if top_similarity >= 0.90:
            return "High"
        if top_similarity >= 0.80:
            return "Medium"
        return "Low"

    if top_similarity >= 0.90 and precision_at_k is not None and float(precision_at_k) >= 0.80:
        return "High"
    if top_similarity >= 0.80:
        return "Medium"
    return "Low"


def confidence_badge(confidence: str) -> str:
    badge_class = {
        "High": "badge badge-high",
        "Medium": "badge badge-medium",
        "Low": "badge badge-low",
    }[confidence]
    return f'<span class="{badge_class}">{confidence} retrieval confidence</span>'


def show_hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-top">
                <span class="badge badge-research">Research Demo</span>
                <span class="badge badge-ai">Biomedical AI</span>
            </div>
            <h1>{APP_TITLE}</h1>
            <div class="subtitle">{APP_CAPTION}</div>
            <div class="description">
                A lightweight research dashboard for visualizing biomedical image retrieval,
                uncertainty, and super-resolution behavior.
            </div>
            <div class="description">
                Use the built-in sample index when available, or upload your own biomedical
                images/contact maps for evaluation.
            </div>
            <div class="badge-row">
                <span class="badge badge-imaging">Biomedical Imaging</span>
                <span class="badge badge-retrieval">Image Retrieval</span>
                <span class="badge badge-sr">Super-Resolution</span>
                <span class="badge badge-xai">Explainable AI</span>
                <span class="badge badge-prototype">Research Prototype</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer">
            This tool is a research visualization prototype and is not intended for medical diagnosis
            or clinical decision-making.
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_startup_checks() -> None:
    available_indexes = available_encoder_indexes()
    if available_indexes:
        st.sidebar.success("Local CXR index available.")
    else:
        st.sidebar.info("No local CXR index found. Upload-only mode is available.")

    if xrv is None:
        st.sidebar.warning("TorchXRayVision is unavailable. Uploaded CXR embeddings fall back to ResNet18.")

    if cooler is None:
        st.sidebar.info("cooler is unavailable. .cool/.mcool upload is disabled; .csv and .npy matrices still work.")


def sidebar_controls(image_paths: np.ndarray, labels: np.ndarray) -> tuple[int, str, int | None, object | None]:
    st.sidebar.markdown("## CXR Retrieval Settings")
    st.sidebar.markdown(
        metric_grid([("Indexed CXR images", str(len(image_paths)))], "metric-grid-3"),
        unsafe_allow_html=True,
    )
    top_k = st.sidebar.slider("Top-k retrieval", min_value=3, max_value=10, value=5)
    query_mode = st.sidebar.radio("Retrieval query mode", ["Select from dataset", "Upload image"])

    selected_index = None
    uploaded_file = None
    if query_mode == "Select from dataset":
        options = [f"{idx}: {Path(str(path)).name} ({labels[idx]})" for idx, path in enumerate(image_paths)]
        selected_option = st.sidebar.selectbox("Retrieval query image", options)
        selected_index = int(selected_option.split(":", 1)[0])
    else:
        uploaded_file = st.sidebar.file_uploader("Upload retrieval image", type=["jpg", "jpeg", "png"])

    st.sidebar.info("This prototype uses visual similarity, not clinical diagnosis.")
    return top_k, query_mode, selected_index, uploaded_file


def prepare_query(
    query_mode: str,
    selected_index: int | None,
    uploaded_file,
    image_paths: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray,
    encoder_key: str,
) -> tuple[Image.Image, np.ndarray, int | None, str, str, str] | None:
    if query_mode == "Select from dataset":
        if selected_index is None:
            return None

        query_path = Path(str(image_paths[selected_index]))
        query_image = safe_load_image(query_path)
        if query_image is None:
            st.error(f"Selected query image is missing or unreadable: {query_path}")
            return None

        return (
            query_image,
            embeddings[selected_index],
            selected_index,
            query_path.name,
            str(labels[selected_index]),
            "Dataset image",
        )

    if uploaded_file is None:
        st.info("Upload a JPG, JPEG, or PNG image from the sidebar to retrieve visually similar examples.")
        return None

    image_bytes = uploaded_file.getvalue()
    digest = hashlib.sha256(image_bytes).hexdigest()
    try:
        query_image = load_uploaded_rgb_image_cached(uploaded_file.name, len(image_bytes), digest, image_bytes)
    except (OSError, UnidentifiedImageError):
        st.error("The uploaded image could not be read. Please upload a valid JPG, JPEG, or PNG file.")
        return None

    try:
        query_embedding = extract_uploaded_embedding_cached(
            uploaded_file.name,
            len(image_bytes),
            digest,
            image_bytes,
            encoder_key,
        )
    except ImportError as error:
        st.warning(f"{error} Using ResNet18 ImageNet baseline for this uploaded query.")
        query_embedding = extract_uploaded_embedding_cached(
            uploaded_file.name,
            len(image_bytes),
            digest,
            image_bytes,
            "resnet18_imagenet",
        )

    return (
        query_image,
        query_embedding,
        None,
        uploaded_file.name,
        "Uploaded image",
        "Upload",
    )


def show_query_card(query_image: Image.Image, filename: str, label: str, source: str) -> None:
    with st.container(border=True):
        st.markdown('<div class="section-title">Query Image</div>', unsafe_allow_html=True)
        st.markdown(image_frame_html(query_image), unsafe_allow_html=True)
        st.caption(filename)
        st.markdown(label_badge(label), unsafe_allow_html=True)
        st.markdown(f'<div class="muted">Source: {source}</div>', unsafe_allow_html=True)


def show_query_analysis_card(
    query_image: Image.Image,
    query_embedding: np.ndarray,
    filename: str,
    label: str,
    source: str,
) -> None:
    with st.container(border=True):
        st.markdown('<div class="section-title">Query Analysis</div>', unsafe_allow_html=True)
        rows = [
            ("Query filename", filename),
            ("Query label", label),
            ("Image resolution", f"{query_image.size[0]} x {query_image.size[1]}"),
            ("Embedding dimension", str(len(query_embedding))),
            ("Query source", source),
        ]
        st.markdown(key_value_rows(rows), unsafe_allow_html=True)


def get_result_labels(results: pd.DataFrame, labels: np.ndarray) -> list[str]:
    return [str(labels[int(index)]) for index in results["index"]]


def show_summary_card(
    results: pd.DataFrame,
    labels: np.ndarray,
    quality: dict[str, float | int | str | None],
    query_index: int | None,
) -> None:
    result_labels = get_result_labels(results, labels)
    normal_count = result_labels.count("NORMAL")
    pneumonia_count = result_labels.count("PNEUMONIA")
    top_index = int(results.iloc[0]["index"])
    confidence = retrieval_confidence_label(quality, uploaded=query_index is None)

    with st.container(border=True):
        st.markdown('<div class="section-title">Retrieval Summary</div>', unsafe_allow_html=True)
        st.markdown(
            metric_grid(
                [
                    ("Top match label", escape(str(labels[top_index]))),
                    ("Top similarity", f"{float(results.iloc[0]['similarity']):.3f}"),
                    ("Retrieved", str(len(results))),
                    ("NORMAL", str(normal_count)),
                    ("PNEUMONIA", str(pneumonia_count)),
                ],
                "metric-grid-5",
            ),
            unsafe_allow_html=True,
        )


def show_retrieval_quality_card(
    results: pd.DataFrame,
    quality: dict[str, float | int | str | None],
    query_index: int | None,
) -> None:
    confidence = retrieval_confidence_label(quality, uploaded=query_index is None)
    with st.container(border=True):
        st.markdown('<div class="section-title">Retrieval Quality</div>', unsafe_allow_html=True)
        if query_index is None:
            agreement_value = "Unavailable"
            precision_value = "Unavailable"
        else:
            agreement_value = f"{quality['agreement_count']}/{len(results)}"
            precision_value = f"{float(quality['precision_at_k']):.2f}"
        st.markdown(
            metric_grid(
                [
                    ("Top-k agreement", agreement_value),
                    ("Precision@K", precision_value),
                    ("Mean cosine", f"{float(quality['mean_similarity']):.3f}"),
                    ("Retrieval confidence", confidence_badge(confidence)),
                ],
                "metric-grid-4",
            ),
            unsafe_allow_html=True,
        )
        if quality["note"]:
            st.info(str(quality["note"]))
        st.caption("Retrieval confidence is not diagnostic confidence.")


def show_model_card(metadata: dict) -> None:
    with st.container(border=True):
        st.markdown('<div class="section-title">Model Card</div>', unsafe_allow_html=True)
        rows = [
            ("Encoder", str(metadata.get("encoder_name", "ResNet18 ImageNet baseline"))),
            ("Embedding dimension", str(metadata.get("embedding_dimension", "512"))),
            ("Similarity metric", "Cosine similarity"),
            ("Dataset", "Pediatric chest X-ray pneumonia dataset"),
            ("Clinical status", "Research prototype only"),
        ]
        if metadata.get("feature_note"):
            rows.append(("Feature representation", str(metadata["feature_note"])))
        st.markdown(
            key_value_rows(rows),
            unsafe_allow_html=True,
        )


def show_retrieval_triage(
    results: pd.DataFrame,
    labels: np.ndarray,
    query_label: str,
    query_index: int | None,
    query_source: str,
) -> None:
    result_labels = get_result_labels(results, labels)
    normal_count = result_labels.count("NORMAL")
    pneumonia_count = result_labels.count("PNEUMONIA")
    if normal_count > pneumonia_count:
        pattern = "NORMAL-majority"
    elif pneumonia_count > normal_count:
        pattern = "PNEUMONIA-majority"
    else:
        pattern = "Mixed"

    similarities = results["similarity"].astype(float)
    agreement = "Unavailable"
    if query_index is not None:
        agreement_count = sum(1 for result_label in result_labels if str(result_label) == str(query_label))
        agreement = f"{agreement_count}/{len(results)}"

    with st.container(border=True):
        st.markdown('<div class="section-title">Retrieval Triage</div>', unsafe_allow_html=True)
        st.markdown(
            key_value_rows(
                [
                    ("Query source", query_source),
                    ("Top retrieved pattern", pattern),
                    ("Retrieval agreement", agreement),
                    ("Similarity range", f"{float(similarities.min()):.3f}-{float(similarities.max()):.3f}"),
                    ("Review status", "Research only"),
                ]
            ),
            unsafe_allow_html=True,
        )


def show_topk_sensitivity(
    query_embedding: np.ndarray,
    query_index: int | None,
    query_label: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> None:
    with st.container(border=True):
        st.markdown('<div class="section-title">Top-k Sensitivity</div>', unsafe_allow_html=True)
        if query_index is None:
            st.info("Top-k label agreement is unavailable for uploaded images.")
            return

        sensitivity_rows = []
        for k in (3, 5, 10):
            k_results = retrieve_similar(query_embedding, embeddings.copy(), top_k=k, query_index=query_index)
            k_labels = get_result_labels(k_results, labels)
            agreement_count = sum(1 for result_label in k_labels if str(result_label) == str(query_label))
            sensitivity_rows.append({"K": f"Top-{k}", "Agreement": agreement_count, "Total": len(k_results)})

        st.markdown(
            metric_grid(
                [(row["K"], f"{row['Agreement']}/{row['Total']}") for row in sensitivity_rows],
                "metric-grid-3",
            ),
            unsafe_allow_html=True,
        )

        chart_data = pd.DataFrame(
            {"K": [row["K"] for row in sensitivity_rows], "Agreement": [row["Agreement"] for row in sensitivity_rows]}
        )
        if alt is not None:
            chart = (
                alt.Chart(chart_data)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#4f46e5")
                .encode(
                    x=alt.X("K:N", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Agreement:Q", title="Same-label count", scale=alt.Scale(domain=[0, 10])),
                    tooltip=["K:N", "Agreement:Q"],
                )
                .properties(height=120)
            )
            st.altair_chart(chart, width="stretch")


def show_query_result_comparison(
    query_image: Image.Image,
    results: pd.DataFrame,
    image_paths: np.ndarray,
) -> None:
    with st.container(border=True):
        st.markdown('<div class="section-title">Query vs Retrieved Comparison</div>', unsafe_allow_html=True)
        rank_options = [f"Rank {int(rank)}" for rank in results["rank"]]
        selected_rank = st.selectbox("Select retrieved rank", rank_options)
        selected_position = rank_options.index(selected_rank)
        selected_row = results.iloc[selected_position]
        selected_path = Path(str(image_paths[int(selected_row["index"])]))
        retrieved_image = safe_load_image(selected_path)
        if retrieved_image is None:
            st.warning("Selected retrieved image is missing or unreadable.")
            return

        col1, col2, col3 = st.columns(3, gap="large")
        with col1:
            st.markdown(image_frame_html(query_image, "crop"), unsafe_allow_html=True)
            st.caption("Query image")
        with col2:
            st.markdown(image_frame_html(retrieved_image, "crop"), unsafe_allow_html=True)
            st.caption(f"{selected_rank}: {selected_path.name}")
        with col3:
            if st.checkbox("Generate difference map", value=False):
                st.markdown(image_frame_html(absolute_difference_map(query_image, retrieved_image), "crop"), unsafe_allow_html=True)
                st.caption("Absolute difference map")
            else:
                st.markdown(
                    '<div class="image-frame crop"><span class="muted">Difference map not generated</span></div>',
                    unsafe_allow_html=True,
                )
                st.caption("Enable map generation to avoid extra processing on reruns")
        st.info("This difference map is a visual comparison aid, not a clinical finding map.")


def make_single_retrieval_report(
    results: pd.DataFrame,
    image_paths: np.ndarray,
    labels: np.ndarray,
    query_filename: str,
    query_label: str,
    query_source: str,
    query_index: int | None,
    quality: dict[str, float | int | str | None],
    confidence: str,
    encoder: str,
) -> pd.DataFrame:
    rows = []
    precision = quality["precision_at_k"]
    for _, row in results.iterrows():
        result_index = int(row["index"])
        result_label = str(labels[result_index])
        rows.append(
            {
                "query_filename": query_filename,
                "query_label": query_label,
                "query_source": query_source,
                "encoder": encoder,
                "top_k": len(results),
                "rank": int(row["rank"]),
                "retrieved_filename": Path(str(image_paths[result_index])).name,
                "retrieved_label": result_label,
                "cosine_similarity": float(row["similarity"]),
                "label_agreement": label_agreement(query_label, result_label, query_index),
                "precision_at_k": "" if precision is None else float(precision),
                "mean_cosine_similarity": float(quality["mean_similarity"]),
                "retrieval_confidence": confidence,
                "report_type": "single_query_retrieval",
            }
        )
    return pd.DataFrame(rows)


def make_batch_retrieval_detailed_report(detail_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "query_id",
        "query_filename",
        "query_label",
        "encoder",
        "k",
        "rank",
        "retrieved_filename",
        "retrieved_label",
        "cosine_similarity",
        "label_agreement",
        "query_precision_at_k",
        "query_top1_match",
        "query_mean_cosine",
        "report_type",
    ]
    if detail_df.empty:
        return pd.DataFrame(columns=columns)
    report = detail_df.copy()
    report["report_type"] = "batch_retrieval_benchmark"
    return report[columns]


def make_batch_retrieval_summary_report(summary_df: pd.DataFrame, detail_df: pd.DataFrame, encoder: str, k: int) -> pd.DataFrame:
    columns = [
        "encoder",
        "num_queries",
        "k",
        "mean_precision_at_k",
        "normal_precision_at_k",
        "pneumonia_precision_at_k",
        "top1_match_rate",
        "mean_cosine_similarity",
        "total_retrieved_normal",
        "total_retrieved_pneumonia",
        "report_type",
    ]
    if summary_df.empty:
        return pd.DataFrame(columns=columns)

    normal_df = summary_df[summary_df["query_label"] == "NORMAL"]
    pneumonia_df = summary_df[summary_df["query_label"] == "PNEUMONIA"]
    row = {
        "encoder": encoder,
        "num_queries": int(len(summary_df)),
        "k": int(k),
        "mean_precision_at_k": float(summary_df["precision_at_k"].mean()),
        "normal_precision_at_k": "" if normal_df.empty else float(normal_df["precision_at_k"].mean()),
        "pneumonia_precision_at_k": "" if pneumonia_df.empty else float(pneumonia_df["precision_at_k"].mean()),
        "top1_match_rate": float(summary_df["top1_label_match"].mean()),
        "mean_cosine_similarity": float(summary_df["mean_cosine_similarity"].mean()),
        "total_retrieved_normal": int(detail_df["retrieved_label"].eq("NORMAL").sum()) if not detail_df.empty else 0,
        "total_retrieved_pneumonia": int(detail_df["retrieved_label"].eq("PNEUMONIA").sum()) if not detail_df.empty else 0,
        "report_type": "batch_retrieval_summary",
    }
    return pd.DataFrame([row], columns=columns)


def show_retrieval_report_download(
    results: pd.DataFrame,
    image_paths: np.ndarray,
    labels: np.ndarray,
    query_filename: str,
    query_label: str,
    query_source: str,
    query_index: int | None,
    quality: dict[str, float | int | str | None],
    metadata: dict,
) -> None:
    confidence = retrieval_confidence_label(quality, uploaded=query_index is None)
    report = make_single_retrieval_report(
        results,
        image_paths,
        labels,
        query_filename,
        query_label,
        query_source,
        query_index,
        quality,
        confidence,
        str(metadata.get("encoder_name", "Unknown encoder")),
    )
    st.caption(CSV_HELPER_TEXT)
    st.download_button(
        "Download single-query retrieval report",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="single_query_retrieval_report.csv",
        mime="text/csv",
    )


def run_batch_retrieval_evaluation(
    image_paths: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray,
    sample_size: int,
    k: int,
    encoder_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(123)
    valid_indices = np.array([idx for idx, label in enumerate(labels) if str(label) in {"NORMAL", "PNEUMONIA"}])
    if len(valid_indices) == 0:
        return pd.DataFrame(), pd.DataFrame()
    sampled_indices = rng.choice(valid_indices, size=min(sample_size, len(valid_indices)), replace=False)

    summary_rows = []
    detail_rows = []
    for query_index in sampled_indices:
        query_label = str(labels[int(query_index)])
        results = retrieve_similar(embeddings[int(query_index)], embeddings.copy(), top_k=k, query_index=int(query_index))
        result_labels = get_result_labels(results, labels)
        same_label_count = sum(1 for result_label in result_labels if result_label == query_label)
        precision_at_k = same_label_count / max(1, len(results))
        top1_match = int(result_labels[0] == query_label) if result_labels else 0
        normal_count = result_labels.count("NORMAL")
        pneumonia_count = result_labels.count("PNEUMONIA")
        summary_rows.append(
            {
                "query_index": int(query_index),
                "query_filename": Path(str(image_paths[int(query_index)])).name,
                "query_label": query_label,
                "precision_at_k": precision_at_k,
                "same_label_count": same_label_count,
                "top1_label_match": top1_match,
                "mean_cosine_similarity": float(results["similarity"].astype(float).mean()),
                "retrieved_normal_count": normal_count,
                "retrieved_pneumonia_count": pneumonia_count,
                "encoder": encoder_name,
                "k": k,
            }
        )
        for _, row in results.iterrows():
            result_index = int(row["index"])
            result_label = str(labels[result_index])
            detail_rows.append(
                {
                    "query_id": int(query_index),
                    "query_filename": Path(str(image_paths[int(query_index)])).name,
                    "query_label": query_label,
                    "encoder": encoder_name,
                    "k": k,
                    "rank": int(row["rank"]),
                    "retrieved_filename": Path(str(image_paths[result_index])).name,
                    "retrieved_label": result_label,
                    "cosine_similarity": float(row["similarity"]),
                    "label_agreement": label_agreement(query_label, result_label, int(query_index)),
                    "query_precision_at_k": precision_at_k,
                    "query_top1_match": top1_match,
                    "query_mean_cosine": float(results["similarity"].astype(float).mean()),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def show_batch_retrieval_evaluation(
    image_paths: np.ndarray,
    labels: np.ndarray,
    embeddings: np.ndarray,
    metadata: dict,
) -> None:
    st.markdown('<div class="section-title">Batch Retrieval Evaluation</div>', unsafe_allow_html=True)
    with st.container(border=True):
        control_col1, control_col2 = st.columns(2)
        sample_size = control_col1.selectbox("Number of query samples", [20, 50, 100], key="batch_retrieval_samples")
        k = control_col2.selectbox("Batch retrieval K", [3, 5, 10], index=1, key="batch_retrieval_k")
        encoder_name = str(metadata.get("encoder_name", "Unknown encoder"))
        summary_df, detail_df = run_batch_retrieval_evaluation(image_paths, labels, embeddings, sample_size, k, encoder_name)
        if summary_df.empty:
            st.info("No labeled query images are available for batch evaluation.")
            return

        normal_df = summary_df[summary_df["query_label"] == "NORMAL"]
        pneumonia_df = summary_df[summary_df["query_label"] == "PNEUMONIA"]
        st.markdown(
            metric_grid(
                [
                    ("Mean Precision@K", f"{summary_df['precision_at_k'].mean():.3f}"),
                    ("NORMAL Precision@K", "n/a" if normal_df.empty else f"{normal_df['precision_at_k'].mean():.3f}"),
                    (
                        "PNEUMONIA Precision@K",
                        "n/a" if pneumonia_df.empty else f"{pneumonia_df['precision_at_k'].mean():.3f}",
                    ),
                    ("Top-1 match rate", f"{summary_df['top1_label_match'].mean():.3f}"),
                ],
                "metric-grid-4",
            ),
            unsafe_allow_html=True,
        )

        if hasattr(st, "segmented_control"):
            batch_view = st.segmented_control(
                "Batch retrieval view",
                ["Graph", "Data table"],
                default="Graph",
            )
        else:
            batch_view = st.radio("Batch retrieval view", ["Graph", "Data table"], horizontal=True, index=0)

        if batch_view == "Graph":
            chart_col1, chart_col2 = st.columns(2, gap="large")
            with chart_col1:
                chart_data = summary_df.groupby("query_label", as_index=False)["precision_at_k"].mean()
                if alt is not None:
                    chart = (
                        alt.Chart(chart_data)
                        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#4f46e5")
                        .encode(
                            x=alt.X("query_label:N", title=None, axis=alt.Axis(labelAngle=0)),
                            y=alt.Y("precision_at_k:Q", title="Mean Precision@K", scale=alt.Scale(domain=[0, 1])),
                            tooltip=["query_label:N", alt.Tooltip("precision_at_k:Q", format=".3f")],
                        )
                        .properties(height=170)
                    )
                    st.altair_chart(chart, width="stretch")
                else:
                    st.bar_chart(chart_data.set_index("query_label"), height=170)
            with chart_col2:
                label_counts = detail_df["retrieved_label"].value_counts().rename_axis("label").reset_index(name="count")
                if alt is not None:
                    chart = (
                        alt.Chart(label_counts)
                        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#14b8a6")
                        .encode(
                            x=alt.X("label:N", title=None, axis=alt.Axis(labelAngle=0)),
                            y=alt.Y("count:Q", title="Retrieved count"),
                            tooltip=["label:N", "count:Q"],
                        )
                        .properties(height=170)
                    )
                    st.altair_chart(chart, width="stretch")
                else:
                    st.bar_chart(label_counts.set_index("label"), height=170)
        else:
            batch_table = summary_df[
                [
                    "query_filename",
                    "query_label",
                    "precision_at_k",
                    "same_label_count",
                    "top1_label_match",
                    "mean_cosine_similarity",
                ]
            ].copy()
            st.dataframe(
                batch_table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "query_filename": st.column_config.TextColumn("Query filename", width="large"),
                    "query_label": st.column_config.TextColumn("Query label", width="small"),
                    "precision_at_k": st.column_config.NumberColumn("Precision@K", width="small", format="%.3f"),
                    "same_label_count": st.column_config.NumberColumn("Same-label count", width="small", format="%d"),
                    "top1_label_match": st.column_config.NumberColumn("Top-1 match", width="small", format="%d"),
                    "mean_cosine_similarity": st.column_config.NumberColumn("Mean cosine", width="small", format="%.3f"),
                },
            )

        detailed_report = make_batch_retrieval_detailed_report(detail_df)
        summary_report = make_batch_retrieval_summary_report(summary_df, detail_df, encoder_name, k)
        st.caption(CSV_HELPER_TEXT)
        st.caption(f"Rows to export: {len(detailed_report)}")
        st.download_button(
            "Download batch retrieval benchmark",
            data=detailed_report.to_csv(index=False).encode("utf-8"),
            file_name="batch_retrieval_detailed_report.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download batch retrieval summary CSV",
            data=summary_report.to_csv(index=False).encode("utf-8"),
            file_name="batch_retrieval_summary_report.csv",
            mime="text/csv",
        )


def run_encoder_comparison_benchmark(sample_size: int, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_dirs = {
        encoder_name: index_dir
        for encoder_name in ENCODER_CONFIGS
        if (index_dir := encoder_index_dir(encoder_name)) is not None
    }
    if len(index_dirs) < 2:
        return pd.DataFrame(), pd.DataFrame()

    loaded = {}
    common_paths: set[str] | None = None
    for encoder_name, index_dir in index_dirs.items():
        image_paths, labels, embeddings, metadata = load_index(str(index_dir))
        path_names = [str(Path(str(path)).resolve()) for path in image_paths]
        loaded[encoder_name] = {
            "paths": image_paths,
            "path_keys": path_names,
            "labels": labels,
            "embeddings": embeddings,
            "metadata": metadata,
            "path_to_index": {path_key: idx for idx, path_key in enumerate(path_names)},
        }
        path_set = set(path_names)
        common_paths = path_set if common_paths is None else common_paths.intersection(path_set)

    if not common_paths:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(777)
    ordered_common_paths = sorted(common_paths)
    sampled_paths = rng.choice(ordered_common_paths, size=min(sample_size, len(ordered_common_paths)), replace=False)

    detail_rows = []
    for encoder_name, state in loaded.items():
        labels = state["labels"]
        embeddings = state["embeddings"]
        encoder_label = str(ENCODER_CONFIGS[encoder_name]["label"])
        for query_path_key in sampled_paths:
            query_index = int(state["path_to_index"][str(query_path_key)])
            query_label = str(labels[query_index])
            results = retrieve_similar(embeddings[query_index], embeddings.copy(), top_k=k, query_index=query_index)
            result_labels = get_result_labels(results, labels)
            same_label_count = sum(1 for result_label in result_labels if result_label == query_label)
            precision_at_k = same_label_count / max(1, len(results))
            top1_match = int(result_labels[0] == query_label) if result_labels else 0
            detail_rows.append(
                {
                    "encoder": encoder_label,
                    "encoder_key": encoder_name,
                    "query_filename": Path(str(state["paths"][query_index])).name,
                    "query_label": query_label,
                    "k": k,
                    "precision_at_k": precision_at_k,
                    "same_label_count": same_label_count,
                    "top1_match": top1_match,
                    "mean_cosine_similarity": float(results["similarity"].astype(float).mean()),
                    "report_type": "encoder_comparison_benchmark",
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    summary_rows = []
    for encoder_label, encoder_df in detail_df.groupby("encoder"):
        normal_df = encoder_df[encoder_df["query_label"] == "NORMAL"]
        pneumonia_df = encoder_df[encoder_df["query_label"] == "PNEUMONIA"]
        summary_rows.append(
            {
                "encoder": encoder_label,
                "num_queries": int(len(encoder_df)),
                "k": int(k),
                "mean_precision_at_k": float(encoder_df["precision_at_k"].mean()),
                "normal_precision_at_k": "" if normal_df.empty else float(normal_df["precision_at_k"].mean()),
                "pneumonia_precision_at_k": "" if pneumonia_df.empty else float(pneumonia_df["precision_at_k"].mean()),
                "top1_match_rate": float(encoder_df["top1_match"].mean()),
                "mean_cosine_similarity": float(encoder_df["mean_cosine_similarity"].mean()),
                "report_type": "encoder_comparison_summary",
            }
        )
    return pd.DataFrame(summary_rows), detail_df


def show_encoder_comparison_benchmark() -> None:
    if sum(1 for encoder_name in ENCODER_CONFIGS if encoder_index_dir(encoder_name) is not None) < 2:
        with st.expander("Encoder Comparison Benchmark", expanded=False):
            st.info(
                "Build at least two encoder indexes to compare generic and CXR-specific embeddings:\n\n"
                "`python build_index.py --encoder resnet18_imagenet`\n\n"
                "`python build_index.py --encoder resnet50_imagenet`\n\n"
                "`python build_index.py --encoder torchxrayvision_densenet121`"
            )
        return

    st.markdown('<div class="section-title">Encoder Comparison Benchmark</div>', unsafe_allow_html=True)
    with st.container(border=True):
        control_col1, control_col2 = st.columns(2)
        sample_size = control_col1.selectbox(
            "Comparison query samples",
            [20, 50, 100],
            key="encoder_comparison_samples",
        )
        k = control_col2.selectbox("Comparison K", [3, 5, 10], index=1, key="encoder_comparison_k")
        summary_df, detail_df = run_encoder_comparison_benchmark(sample_size, k)
        if summary_df.empty:
            st.info("Both encoder indexes exist, but no common query images were found for comparison.")
            return

        def best_encoder(metric_column: str) -> str:
            metric_values = pd.to_numeric(summary_df[metric_column], errors="coerce")
            if metric_values.isna().all():
                return "n/a"
            return str(summary_df.loc[metric_values.idxmax(), "encoder"])

        winner_rows = [
            ("Best Mean Precision@K encoder", best_encoder("mean_precision_at_k")),
            ("Best NORMAL Precision@K encoder", best_encoder("normal_precision_at_k")),
            ("Best PNEUMONIA Precision@K encoder", best_encoder("pneumonia_precision_at_k")),
            ("Best Top-1 Match Rate encoder", best_encoder("top1_match_rate")),
        ]
        with st.container(border=True):
            st.markdown('<div class="section-title">Winner Summary</div>', unsafe_allow_html=True)
            st.markdown(
                metric_grid(
                    [
                        ("Mean P@K", winner_rows[0][1]),
                        ("NORMAL P@K", winner_rows[1][1]),
                        ("PNEUMONIA P@K", winner_rows[2][1]),
                        ("Top-1 Match", winner_rows[3][1]),
                    ],
                    "metric-grid-4",
                ),
                unsafe_allow_html=True,
            )

        metric_items = []
        for _, row in summary_df.iterrows():
            encoder_name = str(row["encoder"])
            mean_precision = float(row["mean_precision_at_k"])
            top1_match = float(row["top1_match_rate"])
            mean_cosine = float(row["mean_cosine_similarity"])
            metric_items.extend(
                [
                    (f"{encoder_name} Mean P@K", f"{mean_precision:.3f}"),
                    (f"{encoder_name} Top-1 Match", f"{top1_match:.3f}"),
                    (f"{encoder_name} Mean cosine", f"{mean_cosine:.3f}"),
                ]
            )
        st.markdown(metric_grid(metric_items, "metric-grid-3"), unsafe_allow_html=True)

        metric_labels = {
            "mean_precision_at_k": "Mean P@K",
            "normal_precision_at_k": "NORMAL P@K",
            "pneumonia_precision_at_k": "PNEUMONIA P@K",
            "top1_match_rate": "Top-1 Match",
        }
        metric_order = list(metric_labels.values())
        encoder_order = [
            "ResNet18 ImageNet baseline",
            "ResNet50 ImageNet baseline",
            "TorchXRayVision DenseNet121 CXR encoder",
        ]
        chart_data = summary_df[["encoder", *metric_labels.keys()]].melt(
            id_vars="encoder",
            var_name="Metric",
            value_name="Score",
        )
        chart_data["Metric"] = chart_data["Metric"].map(metric_labels)
        chart_data["Score"] = pd.to_numeric(chart_data["Score"], errors="coerce")
        chart_data = chart_data.dropna(subset=["Score"])
        chart_data["metric_order"] = chart_data["Metric"].map({metric: index for index, metric in enumerate(metric_order)})
        chart_data["encoder_order"] = chart_data["encoder"].map({encoder: index for index, encoder in enumerate(encoder_order)})
        chart_data = chart_data.sort_values(["metric_order", "encoder_order"]).reset_index(drop=True)

        comparison_table = chart_data[["encoder", "Metric", "Score"]].rename(columns={"encoder": "Encoder"})
        comparison_table["Score"] = comparison_table["Score"].map(lambda value: f"{value:.3f}")
        comparison_table = comparison_table.reset_index(drop=True)

        title_col, view_col = st.columns([0.68, 0.32], vertical_alignment="center")
        with title_col:
            st.markdown('<div class="section-title">Encoder Comparison: Retrieval Quality</div>', unsafe_allow_html=True)
        with view_col:
            st.markdown('<div style="height: 0.28rem;"></div>', unsafe_allow_html=True)
            if hasattr(st, "segmented_control"):
                comparison_view = st.segmented_control(
                    "View",
                    ["Graph", "Values table"],
                    default="Graph",
                    label_visibility="collapsed",
                )
            else:
                comparison_view = st.radio(
                    "View",
                    ["Graph", "Values table"],
                    horizontal=True,
                    index=0,
                    label_visibility="collapsed",
                )
        st.caption(
            "Higher Precision@K and Top-1 Match are better. Mean cosine is not shown in the chart "
            "because it reflects embedding similarity scale, not retrieval correctness."
        )

        if comparison_view == "Graph":
            if alt is not None:
                chart = (
                    alt.Chart(chart_data)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X(
                            "Metric:N",
                            title="Metric",
                            sort=list(metric_labels.values()),
                            axis=alt.Axis(labelAngle=0, labelLimit=160),
                        ),
                        xOffset=alt.XOffset("encoder:N", sort=encoder_order),
                        y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("encoder:N", title="Encoder", sort=encoder_order, legend=alt.Legend(orient="top")),
                        tooltip=[
                            alt.Tooltip("encoder:N", title="Encoder"),
                            alt.Tooltip("Metric:N", title="Metric"),
                            alt.Tooltip("Score:Q", title="Score", format=".3f"),
                        ],
                    )
                    .properties(title="Encoder Comparison: Retrieval Quality", height=260)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.bar_chart(
                    chart_data.pivot(index="Metric", columns="encoder", values="Score"),
                    height=260,
                    use_container_width=True,
                )
        else:
            st.dataframe(
                comparison_table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Encoder": st.column_config.TextColumn("Encoder", width="large"),
                    "Metric": st.column_config.TextColumn("Metric", width="medium"),
                    "Score": st.column_config.TextColumn("Score", width="small"),
                },
            )

        st.caption(CSV_HELPER_TEXT)
        st.caption(f"Rows to export: {len(detail_df)}")
        st.download_button(
            "Download encoder comparison CSV",
            data=detail_df.to_csv(index=False).encode("utf-8"),
            file_name="encoder_comparison_benchmark.csv",
            mime="text/csv",
        )


def show_result_card(
    row: pd.Series,
    image_paths: np.ndarray,
    labels: np.ndarray,
    query_label: str,
    query_index: int | None,
) -> None:
    result_index = int(row["index"])
    image_path = Path(str(image_paths[result_index]))
    label = str(labels[result_index])
    similarity = float(row["similarity"])
    agreement = label_agreement(query_label, label, query_index)

    image = safe_load_image(image_path)
    if image is None:
        image_html = '<div class="image-frame compact"><span class="muted">Image unavailable</span></div>'
    else:
        image_html = image_frame_html(image, "compact")

    progress_width = float(np.clip(similarity, 0.0, 1.0)) * 100.0
    card_html = f"""
    <div class="result-card">
        <div class="result-head">
            <span class="rank-pill">Rank {int(row['rank'])}</span>
            {label_badge(label)}
        </div>
        {image_html}
        <div class="mini-row"><span>Filename</span><span>{filename_with_tooltip(image_path.name)}</span></div>
        <div class="mini-row"><span>Cosine similarity</span><span>{similarity:.3f}</span></div>
        <div class="mini-row"><span>Agreement</span><span>{agreement_badge(agreement)}</span></div>
        <div class="thin-progress"><div style="width: {progress_width:.1f}%"></div></div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def build_result_table(
    results: pd.DataFrame,
    image_paths: np.ndarray,
    labels: np.ndarray,
    query_label: str,
    query_index: int | None,
) -> pd.DataFrame:
    rows = []
    for _, row in results.iterrows():
        result_index = int(row["index"])
        result_label = str(labels[result_index])
        rows.append(
            {
                "Rank": str(int(row["rank"])),
                "Filename": Path(str(image_paths[result_index])).name,
                "Label": str(result_label).upper(),
                "Cosine similarity": f"{float(row['similarity']):.3f}",
                "Label agreement": label_agreement(query_label, result_label, query_index),
            }
        )
    return pd.DataFrame(rows, columns=["Rank", "Filename", "Label", "Cosine similarity", "Label agreement"])


def show_result_grid(
    results: pd.DataFrame,
    image_paths: np.ndarray,
    labels: np.ndarray,
    query_label: str,
    query_index: int | None,
) -> None:
    st.markdown('<div class="section-title">Similar Case Board</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Visual neighbors ranked by embedding cosine similarity.</div>',
        unsafe_allow_html=True,
    )
    columns_per_row = 3

    for row_start in range(0, len(results), columns_per_row):
        columns = st.columns(columns_per_row)
        for column, (_, row) in zip(columns, results.iloc[row_start : row_start + columns_per_row].iterrows()):
            with column:
                show_result_card(row, image_paths, labels, query_label, query_index)

    st.markdown('<div class="section-title">Compact Result Table</div>', unsafe_allow_html=True)
    result_df = build_result_table(results, image_paths, labels, query_label, query_index)
    st.dataframe(
        result_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width="small"),
            "Filename": st.column_config.TextColumn("Filename", width="large"),
            "Label": st.column_config.TextColumn("Label", width="small"),
            "Cosine similarity": st.column_config.TextColumn("Cosine similarity", width="small"),
            "Label agreement": st.column_config.TextColumn("Label agreement", width="medium"),
        },
    )


def show_similarity_chart(results: pd.DataFrame) -> None:
    chart_data = pd.DataFrame(
        {
            "Rank": [f"Rank {int(rank)}" for rank in results["rank"]],
            "Cosine similarity": results["similarity"].astype(float),
        }
    )
    if alt is not None:
        chart = (
            alt.Chart(chart_data)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#2563eb")
            .encode(
                x=alt.X("Rank:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Cosine similarity:Q", title="Cosine similarity", scale=alt.Scale(domain=[0, 1])),
                tooltip=["Rank:N", alt.Tooltip("Cosine similarity:Q", format=".3f")],
            )
            .properties(height=190)
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.bar_chart(chart_data.set_index("Rank"), height=190)


def show_label_distribution(results: pd.DataFrame, labels: np.ndarray) -> None:
    result_labels = get_result_labels(results, labels)
    label_counts = pd.Series(result_labels).value_counts().sort_index()
    distribution = pd.DataFrame({"Label": label_counts.index, "Count": label_counts.values})
    if alt is not None:
        chart = (
            alt.Chart(distribution)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#14b8a6")
            .encode(
                x=alt.X("Label:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Count:Q", title="Count", scale=alt.Scale(domainMin=0)),
                tooltip=["Label:N", "Count:Q"],
            )
            .properties(height=170)
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.bar_chart(distribution.set_index("Label"), height=170)


def show_intensity_heatmap_preview(query_image: Image.Image) -> None:
    with st.expander("Visual explanation preview", expanded=False):
        if not st.checkbox("Generate intensity heatmap", value=False):
            st.caption("Enable this preview only when needed to avoid extra processing on reruns.")
            return
        with st.container(border=True):
            st.markdown('<div class="section-title">Intensity Heatmap Preview</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.markdown(image_frame_html(query_image), unsafe_allow_html=True)
                st.caption("Original query image")
            with col2:
                st.markdown(image_frame_html(create_intensity_heatmap(query_image)), unsafe_allow_html=True)
                st.caption("Viridis-style grayscale intensity heatmap")
            st.info(
                "This is a simple intensity visualization, not Grad-CAM. Future work will add "
                "model-based visual grounding."
            )


def show_retrieval_explanation_cards() -> None:
    with st.expander("Method notes", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1.container(border=True):
            st.markdown("**Embedding**")
            st.write("The model converts each X-ray into a numerical vector.")

        with col2.container(border=True):
            st.markdown("**Similarity**")
            st.write("Cosine similarity measures how close two image vectors are.")

        with col3.container(border=True):
            st.markdown("**Retrieval**")
            st.write("Higher similarity means stronger visual resemblance to the query.")

    with st.expander("Limitations", expanded=False):
        st.write(
            "Retrieval results show visual similarity, not diagnosis. Retrieval confidence is not clinical "
            "confidence, folder labels are only used for visualization, and the intensity heatmap is not Grad-CAM."
        )


def show_missing_index_message() -> None:
    st.info(
        "A matching saved CXR embedding index is not available for the current selection. "
        "Upload-first mode is available below, or build a local index with `python build_index.py`."
    )


def show_uploaded_gallery_results(
    results: pd.DataFrame,
    gallery_items: list[dict],
) -> None:
    st.markdown('<div class="section-title">Similar Uploaded Reference Board</div>', unsafe_allow_html=True)
    st.caption("Uploaded gallery retrieval is temporary for this browser session and is not saved.")

    columns_per_row = 3
    for row_start in range(0, len(results), columns_per_row):
        columns = st.columns(columns_per_row)
        for column, (_, row) in zip(columns, results.iloc[row_start : row_start + columns_per_row].iterrows()):
            item = gallery_items[int(row["index"])]
            similarity = float(row["similarity"])
            progress_width = float(np.clip(similarity, 0.0, 1.0)) * 100.0
            with column:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-head">
                            <span class="rank-pill">Rank {int(row['rank'])}</span>
                            <span class="badge badge-unknown">Unknown label</span>
                        </div>
                        {image_frame_html(item["image"], "compact")}
                        <div class="mini-row"><span>Filename</span><span>{filename_with_tooltip(item["filename"])}</span></div>
                        <div class="mini-row"><span>Cosine similarity</span><span>{similarity:.3f}</span></div>
                        <div class="mini-row"><span>Agreement</span><span>{agreement_badge("Unknown")}</span></div>
                        <div class="thin-progress"><div style="width: {progress_width:.1f}%"></div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    table = pd.DataFrame(
        [
            {
                "Rank": str(int(row["rank"])),
                "Filename": gallery_items[int(row["index"])]["filename"],
                "Label": "Unknown",
                "Cosine similarity": f"{float(row['similarity']):.3f}",
                "Label agreement": "Unknown",
            }
            for _, row in results.iterrows()
        ],
        columns=["Rank", "Filename", "Label", "Cosine similarity", "Label agreement"],
    )
    st.markdown('<div class="section-title">Compact Result Table</div>', unsafe_allow_html=True)
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width="small"),
            "Filename": st.column_config.TextColumn("Filename", width="large"),
            "Label": st.column_config.TextColumn("Label", width="small"),
            "Cosine similarity": st.column_config.TextColumn("Cosine similarity", width="small"),
            "Label agreement": st.column_config.TextColumn("Label agreement", width="medium"),
        },
    )


def show_uploaded_retrieval_mode() -> None:
    st.markdown('<span class="badge badge-prototype">Upload-only mode</span>', unsafe_allow_html=True)
    show_missing_index_message()
    st.caption("Upload a query image and a small reference gallery to build temporary in-session embeddings.")

    encoder_options = encoder_options_for_uploads()
    selected_encoder_label = st.selectbox("Encoder for uploaded images", list(encoder_options.keys()))
    encoder_key = encoder_options[selected_encoder_label]
    top_k = st.slider("Top-k uploaded reference retrieval", min_value=1, max_value=10, value=5)

    upload_col1, upload_col2 = st.columns([1, 1], gap="large")
    with upload_col1:
        query_file = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png"], key="upload_first_query")
    with upload_col2:
        gallery_files = st.file_uploader(
            "Upload reference gallery images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="upload_first_gallery",
        )

    if query_file is None or not gallery_files:
        st.info("Upload one query image and at least one reference image to run temporary retrieval.")
        return

    try:
        query_bytes = query_file.getvalue()
        query_digest = hashlib.sha256(query_bytes).hexdigest()
        query_image = load_uploaded_rgb_image_cached(query_file.name, len(query_bytes), query_digest, query_bytes)
        query_embedding, active_encoder_key, warning = embed_uploaded_bytes_with_fallback(
            query_file.name,
            query_bytes,
            encoder_key,
        )
    except (OSError, UnidentifiedImageError, RuntimeError, ImportError, ValueError) as error:
        st.error(f"Could not process the uploaded query image: {error}")
        return

    if warning:
        st.warning(warning)

    gallery_items = []
    gallery_embeddings = []
    for gallery_file in gallery_files:
        try:
            image_bytes = gallery_file.getvalue()
            image_digest = hashlib.sha256(image_bytes).hexdigest()
            image = load_uploaded_rgb_image_cached(gallery_file.name, len(image_bytes), image_digest, image_bytes)
            embedding, active_encoder_key, warning = embed_uploaded_bytes_with_fallback(
                gallery_file.name,
                image_bytes,
                active_encoder_key,
            )
            if warning:
                st.warning(warning)
            gallery_items.append({"filename": gallery_file.name, "image": image})
            gallery_embeddings.append(embedding)
        except (OSError, UnidentifiedImageError, RuntimeError, ImportError, ValueError) as error:
            st.warning(f"Skipped {gallery_file.name}: {error}")

    if not gallery_embeddings:
        st.error("No readable reference gallery images were available for retrieval.")
        return

    embeddings = np.vstack(gallery_embeddings)
    results = retrieve_similar(query_embedding, embeddings, top_k=min(top_k, len(gallery_items)))
    quality = calculate_retrieval_quality(
        results,
        np.array(["Uploaded reference"] * len(gallery_items)),
        "Uploaded image",
        None,
    )
    confidence = retrieval_confidence_label(quality, uploaded=True)

    query_col, analysis_col = st.columns([0.45, 0.55], gap="large")
    with query_col:
        show_query_card(query_image, query_file.name, "Uploaded image", "Upload")
    with analysis_col:
        show_query_analysis_card(query_image, query_embedding, query_file.name, "Uploaded image", "Upload")
        st.markdown(
            metric_grid(
                [
                    ("Reference images", str(len(gallery_items))),
                    ("Top similarity", f"{float(quality['top_similarity']):.3f}"),
                    ("Mean cosine", f"{float(quality['mean_similarity']):.3f}"),
                ],
                "metric-grid-3",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="confidence-card">
                {confidence_badge(confidence)}
                <div class="confidence-note">Retrieval confidence, not diagnostic confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    model_rows = [
        (
            "Encoder",
            "ResNet18 ImageNet baseline"
            if canonical_encoder_name(active_encoder_key) == "resnet18_imagenet"
            else selected_encoder_label,
        ),
        ("Embedding dimension", str(embeddings.shape[1])),
        ("Similarity metric", "Cosine similarity"),
        ("Reference source", "Uploaded temporary gallery"),
        ("Clinical status", "Research prototype only"),
    ]
    with st.container(border=True):
        st.markdown('<div class="section-title">Upload Gallery Model Card</div>', unsafe_allow_html=True)
        st.markdown(key_value_rows(model_rows), unsafe_allow_html=True)

    show_intensity_heatmap_preview(query_image)
    show_uploaded_gallery_results(results, gallery_items)


def render_retrieval_dashboard(
    query_image: Image.Image,
    query_embedding: np.ndarray,
    query_index: int | None,
    filename: str,
    query_label: str,
    query_source: str,
    embeddings: np.ndarray,
    image_paths: np.ndarray,
    labels: np.ndarray,
    top_k: int,
    metadata: dict,
) -> None:
    results = retrieve_similar(query_embedding, embeddings.copy(), top_k=top_k, query_index=query_index)
    quality = calculate_retrieval_quality(results, labels, query_label, query_index)
    st.session_state["last_retrieval_context"] = {
        "query_filename": filename,
        "query_label": query_label,
        "query_source": query_source,
        "top_k": top_k,
        "encoder": metadata.get("encoder_name", "Unknown encoder"),
    }

    query_col, analysis_col = st.columns([1, 1], gap="large")
    with query_col:
        show_query_card(query_image, filename, query_label, query_source)
    with analysis_col:
        show_query_analysis_card(query_image, query_embedding, filename, query_label, query_source)

    show_summary_card(results, labels, quality, query_index)
    show_retrieval_quality_card(results, quality, query_index)

    triage_col, model_col = st.columns(2, gap="large")
    with triage_col:
        show_retrieval_triage(results, labels, query_label, query_index, query_source)
    with model_col:
        show_model_card(metadata)

    show_intensity_heatmap_preview(query_image)
    show_topk_sensitivity(query_embedding, query_index, query_label, embeddings, labels)
    show_query_result_comparison(query_image, results, image_paths)
    show_retrieval_report_download(
        results,
        image_paths,
        labels,
        filename,
        query_label,
        query_source,
        query_index,
        quality,
        metadata,
    )

    show_result_grid(results, image_paths, labels, query_label, query_index)

    if hasattr(st, "segmented_control"):
        retrieval_chart_view = st.segmented_control(
            "Similarity and label distribution view",
            ["Graph", "Data table"],
            default="Graph",
        )
    else:
        retrieval_chart_view = st.radio(
            "Similarity and label distribution view",
            ["Graph", "Data table"],
            horizontal=True,
            index=0,
        )

    if retrieval_chart_view == "Graph":
        chart_col, distribution_col = st.columns([1.15, 0.85], gap="large")
        with chart_col:
            with st.container(border=True):
                st.markdown('<div class="section-title">Similarity Scores</div>', unsafe_allow_html=True)
                show_similarity_chart(results)

        with distribution_col:
            with st.container(border=True):
                st.markdown('<div class="section-title">Label Distribution</div>', unsafe_allow_html=True)
                show_label_distribution(results, labels)
    else:
        similarity_table = pd.DataFrame(
            {
                "Rank": [str(int(rank)) for rank in results["rank"]],
                "Cosine similarity": [f"{float(value):.3f}" for value in results["similarity"]],
            }
        )
        result_labels = get_result_labels(results, labels)
        distribution_table = (
            pd.Series(result_labels)
            .value_counts()
            .sort_index()
            .rename_axis("Label")
            .reset_index(name="Count")
        )
        data_col1, data_col2 = st.columns([1.15, 0.85], gap="large")
        with data_col1:
            with st.container(border=True):
                st.markdown('<div class="section-title">Similarity Scores</div>', unsafe_allow_html=True)
                st.dataframe(
                    similarity_table,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Rank": st.column_config.TextColumn("Rank", width="small"),
                        "Cosine similarity": st.column_config.TextColumn("Cosine similarity", width="small"),
                    },
                )
        with data_col2:
            with st.container(border=True):
                st.markdown('<div class="section-title">Label Distribution</div>', unsafe_allow_html=True)
                st.dataframe(
                    distribution_table,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Label": st.column_config.TextColumn("Label", width="medium"),
                        "Count": st.column_config.NumberColumn("Count", width="small", format="%d"),
                    },
                )

    show_batch_retrieval_evaluation(image_paths, labels, embeddings, metadata)
    show_encoder_comparison_benchmark()
    show_retrieval_explanation_cards()


def render_retrieval_tab(index_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> None:
    st.markdown('<div class="section-title">CXR Retrieval Visualization</div>', unsafe_allow_html=True)
    encoder_label_to_key = {str(config["label"]): encoder_name for encoder_name, config in ENCODER_CONFIGS.items()}
    selected_encoder_label = st.sidebar.selectbox("CXR encoder index", list(encoder_label_to_key.keys()))
    selected_encoder_key = encoder_label_to_key[selected_encoder_label]
    selected_index_dir = encoder_index_dir(selected_encoder_key)

    if selected_index_dir is None:
        st.warning(f"Index not found for this encoder. Run `python build_index.py --encoder {selected_encoder_key}`.")
        show_uploaded_retrieval_mode()
        return

    st.markdown('<span class="badge badge-normal">Local CXR index available</span>', unsafe_allow_html=True)
    try:
        image_paths, labels, embeddings, metadata = load_index(str(selected_index_dir))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not load embeddings: {error}")
        return

    if len(image_paths) == 0:
        st.error("The embeddings index is empty. Add images to data/chest_xray and run python build_index.py.")
        return

    encoder_key = canonical_encoder_name(str(metadata.get("encoder_key", selected_encoder_key)))
    if encoder_key == "torchxrayvision_densenet121" and xrv is None:
        st.warning(
            "TorchXRayVision embeddings are selected, but torchxrayvision is not installed. "
            "Dataset queries still work from saved embeddings; uploaded queries will fall back to ResNet18."
        )
    top_k, query_mode, selected_index, uploaded_file = sidebar_controls(image_paths, labels)
    query = prepare_query(query_mode, selected_index, uploaded_file, image_paths, labels, embeddings, encoder_key)
    if query is None:
        return

    query_image, query_embedding, query_index, filename, query_label, query_source = query
    if query_embedding.shape[0] != embeddings.shape[1]:
        st.warning(
            "Uploaded query embedding dimension does not match the selected saved index. "
            "Select a matching index or rebuild embeddings for the desired encoder."
        )
        return

    render_retrieval_dashboard(
        query_image,
        query_embedding,
        query_index,
        filename,
        query_label,
        query_source,
        embeddings,
        image_paths,
        labels,
        top_k,
        metadata,
    )


def sr_image_options(image_paths: np.ndarray, labels: np.ndarray) -> list[str]:
    return [f"{idx}: {Path(str(path)).name} ({labels[idx]})" for idx, path in enumerate(image_paths)]


def prepare_super_resolution_image(
    index_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[Image.Image, str, str] | None:
    has_index = index_state is not None and len(index_state[0]) > 0
    source_options = ["Upload image"]
    if has_index:
        source_options.insert(0, "Select from indexed CXR dataset")

    source = st.radio("Super-resolution image source", source_options, horizontal=True)

    if source == "Select from indexed CXR dataset" and index_state is not None:
        image_paths, labels, _ = index_state
        selected_option = st.selectbox("Image for super-resolution demo", sr_image_options(image_paths, labels))
        selected_index = int(selected_option.split(":", 1)[0])
        image_path = Path(str(image_paths[selected_index]))
        image = safe_load_image(image_path)
        if image is None:
            st.error(f"Selected image is missing or unreadable: {image_path}")
            return None
        return image, image_path.name, str(labels[selected_index])

    uploaded_file = st.file_uploader("Upload biomedical image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Upload an image, or build the CXR index to select from the indexed dataset.")
        return None

    try:
        image_bytes = uploaded_file.getvalue()
        image_digest = hashlib.sha256(image_bytes).hexdigest()
        return (
            load_uploaded_rgb_image_cached(uploaded_file.name, len(image_bytes), image_digest, image_bytes),
            uploaded_file.name,
            "Uploaded image",
        )
    except (OSError, UnidentifiedImageError):
        st.error("The uploaded image could not be read. Please upload a valid JPG, JPEG, or PNG file.")
        return None


def show_sr_panel(title: str, image: Image.Image, caption: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(image_frame_html(image, "crop"), unsafe_allow_html=True)
        st.caption(caption)


def show_zoomed_crop_comparison(
    original: Image.Image,
    low_res: Image.Image,
    enhanced: Image.Image,
    crop_size: int,
) -> None:
    st.markdown('<div class="section-title">Zoomed Center Crop Comparison</div>', unsafe_allow_html=True)
    original_crop = center_crop(original, crop_size)
    low_res_crop = center_crop_low_res(low_res, original.size, crop_size)
    enhanced_crop = center_crop(enhanced, crop_size)

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        show_sr_panel("Original crop", original_crop, f"{crop_size} x {crop_size} center crop")
    with col2:
        show_sr_panel("Low-resolution crop", low_res_crop, "Equivalent low-res crop, enlarged")
    with col3:
        show_sr_panel("Bicubic enhanced crop", enhanced_crop, "Center crop after reconstruction")


def show_sr_model_card() -> None:
    with st.container(border=True):
        st.markdown('<div class="section-title">Baseline Model Card</div>', unsafe_allow_html=True)
        st.markdown(
            key_value_rows(
                [
                    ("Degradation", "Downsampling"),
                    ("Reconstruction", "Bicubic interpolation"),
                    ("Metrics", "PSNR, SSIM, MSE"),
                    ("Clinical status", "Educational baseline only"),
                ]
            ),
            unsafe_allow_html=True,
        )


def show_sr_large_preview(original: Image.Image, low_res: Image.Image, enhanced: Image.Image) -> None:
    preview_choice = st.radio(
        "Large preview",
        ["Original", "Simulated low-resolution", "Bicubic enhanced"],
        horizontal=True,
    )
    preview_image = {
        "Original": original,
        "Simulated low-resolution": low_res,
        "Bicubic enhanced": enhanced,
    }[preview_choice]
    with st.container(border=True):
        st.markdown(f'<div class="section-title">{preview_choice}</div>', unsafe_allow_html=True)
        st.markdown(image_frame_html(preview_image, "large"), unsafe_allow_html=True)


def sr_metrics_row(method: str, original: Image.Image, candidate: Image.Image) -> dict:
    candidate = candidate.convert("RGB").resize(original.size, Image.Resampling.BICUBIC)
    mse = calculate_mse(original, candidate)
    psnr = calculate_psnr(original, candidate)
    ssim = calculate_ssim(original, candidate)
    return {"method": method, "psnr": psnr, "ssim": ssim, "mse": mse}


def make_sr_method_comparison_report(
    image_filename: str,
    scale_factor: str,
    bicubic_metrics: dict,
    uploaded_metrics: dict,
) -> pd.DataFrame:
    delta_psnr = float(uploaded_metrics["psnr"] - bicubic_metrics["psnr"])
    bicubic_ssim = bicubic_metrics["ssim"]
    uploaded_ssim = uploaded_metrics["ssim"]
    delta_ssim = "" if bicubic_ssim is None or uploaded_ssim is None else float(uploaded_ssim - bicubic_ssim)
    delta_mse = float(bicubic_metrics["mse"] - uploaded_metrics["mse"])
    return pd.DataFrame(
        [
            {
                "image_filename": image_filename,
                "scale_factor": scale_factor,
                "method": "Bicubic baseline",
                "psnr": bicubic_metrics["psnr"],
                "ssim": "" if bicubic_ssim is None else bicubic_ssim,
                "mse": bicubic_metrics["mse"],
                "delta_psnr_vs_bicubic": 0.0,
                "delta_ssim_vs_bicubic": "" if bicubic_ssim is None else 0.0,
                "delta_mse_vs_bicubic": 0.0,
            },
            {
                "image_filename": image_filename,
                "scale_factor": scale_factor,
                "method": "Uploaded model output",
                "psnr": uploaded_metrics["psnr"],
                "ssim": "" if uploaded_ssim is None else uploaded_ssim,
                "mse": uploaded_metrics["mse"],
                "delta_psnr_vs_bicubic": delta_psnr,
                "delta_ssim_vs_bicubic": delta_ssim,
                "delta_mse_vs_bicubic": delta_mse,
            },
        ]
    )


def show_model_output_comparison(
    original: Image.Image,
    low_res: Image.Image,
    enhanced: Image.Image,
    image_filename: str,
    scale_label: str,
) -> list[dict]:
    uploaded_model_output = st.file_uploader(
        "Upload model-enhanced output for comparison",
        type=["jpg", "jpeg", "png"],
        key="model_enhanced_upload",
    )
    bicubic_metrics = sr_metrics_row("Bicubic baseline", original, enhanced)
    metric_rows = [bicubic_metrics]
    if uploaded_model_output is None:
        return metric_rows

    try:
        model_bytes = uploaded_model_output.getvalue()
        model_digest = hashlib.sha256(model_bytes).hexdigest()
        model_output = load_uploaded_rgb_image_cached(
            uploaded_model_output.name,
            len(model_bytes),
            model_digest,
            model_bytes,
        ).resize(original.size, Image.Resampling.BICUBIC)
    except (OSError, UnidentifiedImageError):
        st.error("The uploaded model-enhanced image could not be read.")
        return metric_rows

    uploaded_metrics = sr_metrics_row("Uploaded model output", original, model_output)
    metric_rows.append(uploaded_metrics)
    st.info(
        "This enables comparison of future learned or diffusion-based biomedical super-resolution "
        "outputs against a transparent bicubic baseline."
    )

    st.markdown('<div class="section-title">SR Method Output Comparison</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4, gap="large")
    with col1:
        show_sr_panel("Original", original, "Reference")
    with col2:
        show_sr_panel("Simulated low-resolution", low_res, "Degraded input")
    with col3:
        show_sr_panel("Bicubic enhanced", enhanced, "Transparent baseline")
    with col4:
        show_sr_panel("Uploaded model output", model_output, "External model result")

    bicubic_ssim = bicubic_metrics["ssim"]
    uploaded_ssim = uploaded_metrics["ssim"]
    delta_psnr = float(uploaded_metrics["psnr"] - bicubic_metrics["psnr"])
    delta_ssim = None if bicubic_ssim is None or uploaded_ssim is None else float(uploaded_ssim - bicubic_ssim)
    delta_mse = float(bicubic_metrics["mse"] - uploaded_metrics["mse"])
    st.markdown(
        metric_grid(
            [
                ("PSNR improvement", f"{delta_psnr:+.2f} dB"),
                ("SSIM improvement", "n/a" if delta_ssim is None else f"{delta_ssim:+.3f}"),
                ("MSE reduction", f"{delta_mse:+.2f}"),
            ],
            "metric-grid-3",
        ),
        unsafe_allow_html=True,
    )

    comparison_table = pd.DataFrame(
        [
            {
                "method": "Bicubic baseline",
                "PSNR": bicubic_metrics["psnr"],
                "SSIM": "" if bicubic_ssim is None else bicubic_ssim,
                "MSE": bicubic_metrics["mse"],
                "note": "Transparent interpolation baseline",
            },
            {
                "method": "Uploaded model output",
                "PSNR": uploaded_metrics["psnr"],
                "SSIM": "" if uploaded_ssim is None else uploaded_ssim,
                "MSE": uploaded_metrics["mse"],
                "note": "External model output resized to original size",
            },
        ]
    )
    st.dataframe(comparison_table, width="stretch", hide_index=True)

    max_crop = max(16, min(original.size))
    default_crop = min(128, max_crop)
    comparison_crop_size = st.slider(
        "Model comparison crop size",
        min_value=16,
        max_value=max_crop,
        value=default_crop,
        step=8 if max_crop >= 64 else 1,
        key="model_output_crop_size",
    )
    original_crop = center_crop(original, comparison_crop_size)
    bicubic_crop = center_crop(enhanced, comparison_crop_size)
    model_crop = center_crop(model_output, comparison_crop_size)
    difference_crop = absolute_difference_map(original_crop, model_crop)

    st.markdown('<div class="section-title">Uploaded Model Crop Comparison</div>', unsafe_allow_html=True)
    crop_col1, crop_col2, crop_col3, crop_col4 = st.columns(4, gap="large")
    with crop_col1:
        show_sr_panel("Original crop", original_crop, f"{comparison_crop_size} x {comparison_crop_size} center crop")
    with crop_col2:
        show_sr_panel("Bicubic crop", bicubic_crop, "Baseline crop")
    with crop_col3:
        show_sr_panel("Uploaded model output crop", model_crop, "External model crop")
    with crop_col4:
        show_sr_panel("Difference map", difference_crop, "Original vs uploaded model output")

    report = make_sr_method_comparison_report(image_filename, scale_label, bicubic_metrics, uploaded_metrics)
    st.caption(CSV_HELPER_TEXT)
    st.download_button(
        "Download SR method comparison CSV",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="sr_method_comparison.csv",
        mime="text/csv",
    )
    return metric_rows


def make_single_sr_report(
    image_filename: str,
    scale_factor: str,
    original_size: tuple[int, int],
    psnr: float,
    ssim: float | None,
    mse: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_filename": image_filename,
                "scale_factor": scale_factor,
                "original_size": f"{original_size[0]} x {original_size[1]}",
                "reconstruction_method": "Bicubic interpolation",
                "psnr": "inf" if np.isinf(psnr) else psnr,
                "ssim": "" if ssim is None else ssim,
                "mse": mse,
                "artifact_warning": SR_ARTIFACT_WARNING,
                "report_type": "single_image_sr",
            }
        ]
    )


def make_batch_sr_detailed_report(report: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "image_filename",
        "scale_factor",
        "original_width",
        "original_height",
        "reconstruction_method",
        "psnr",
        "ssim",
        "mse",
        "artifact_warning",
        "report_type",
    ]
    if report.empty:
        return pd.DataFrame(columns=columns)
    detailed = report.copy()
    detailed["reconstruction_method"] = detailed["method"]
    detailed["artifact_warning"] = SR_ARTIFACT_WARNING
    detailed["report_type"] = "batch_sr_benchmark"
    return detailed[columns]


def make_batch_sr_summary_report(report: pd.DataFrame, scale_factor: str) -> pd.DataFrame:
    columns = [
        "mean_psnr",
        "mean_ssim",
        "mean_mse",
        "best_image_by_ssim",
        "worst_image_by_ssim",
        "num_images",
        "scale_factor",
        "report_type",
    ]
    if report.empty:
        return pd.DataFrame(columns=columns)

    numeric_ssim = pd.to_numeric(report["ssim"], errors="coerce")
    if numeric_ssim.isna().all():
        best_image = ""
        worst_image = ""
        mean_ssim = ""
    else:
        best_image = str(report.loc[numeric_ssim.idxmax(), "image_filename"])
        worst_image = str(report.loc[numeric_ssim.idxmin(), "image_filename"])
        mean_ssim = float(numeric_ssim.mean())

    return pd.DataFrame(
        [
            {
                "mean_psnr": float(report["psnr"].mean()),
                "mean_ssim": mean_ssim,
                "mean_mse": float(report["mse"].mean()),
                "best_image_by_ssim": best_image,
                "worst_image_by_ssim": worst_image,
                "num_images": int(len(report)),
                "scale_factor": scale_factor,
                "report_type": "batch_sr_summary",
            }
        ],
        columns=columns,
    )


def run_batch_sr_evaluation(image_paths: np.ndarray, sample_size: int, scale_factor: int, scale_label: str) -> pd.DataFrame:
    rng = np.random.default_rng(321)
    indices = np.arange(len(image_paths))
    sampled_indices = rng.choice(indices, size=min(sample_size, len(indices)), replace=False)
    rows = []
    for index in sampled_indices:
        image_path = Path(str(image_paths[int(index)]))
        image = safe_load_image(image_path)
        if image is None:
            continue
        low_res, enhanced = simulate_low_resolution(image, scale_factor)
        mse = calculate_mse(image, enhanced)
        psnr = calculate_psnr(image, enhanced)
        ssim = calculate_ssim(image, enhanced)
        rows.append(
            {
                "image_filename": image_path.name,
                "scale_factor": scale_label,
                "original_size": f"{image.size[0]} x {image.size[1]}",
                "original_width": image.size[0],
                "original_height": image.size[1],
                "method": "Bicubic interpolation",
                "psnr": psnr,
                "ssim": "" if ssim is None else ssim,
                "mse": mse,
            }
        )
    return pd.DataFrame(rows)


def show_batch_sr_evaluation(index_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> None:
    st.markdown('<div class="section-title">Batch Super-Resolution Evaluation</div>', unsafe_allow_html=True)
    with st.container(border=True):
        if index_state is None or len(index_state[0]) == 0:
            st.info("Build the CXR index to run batch super-resolution evaluation on indexed images.")
            return
        image_paths, _, _ = index_state
        control_col1, control_col2 = st.columns(2)
        sample_size = control_col1.selectbox("Number of images", [20, 50, 100], key="batch_sr_samples")
        scale_label = control_col2.selectbox("Batch scale factor", ["2x", "4x"], key="batch_sr_scale")
        scale_factor = int(scale_label.replace("x", ""))
        report = run_batch_sr_evaluation(image_paths, sample_size, scale_factor, scale_label)
        if report.empty:
            st.info("No readable images were available for batch evaluation.")
            return

        numeric_ssim = pd.to_numeric(report["ssim"], errors="coerce")
        metric_items = [
            ("Mean PSNR", f"{report['psnr'].mean():.2f} dB"),
            ("Mean SSIM", "n/a" if numeric_ssim.isna().all() else f"{numeric_ssim.mean():.3f}"),
            ("Mean MSE", f"{report['mse'].mean():.2f}"),
        ]
        if numeric_ssim.isna().all():
            metric_items.extend([("Worst SSIM", "n/a"), ("Best SSIM", "n/a")])
        else:
            worst_filename = str(report.loc[numeric_ssim.idxmin(), "image_filename"])
            best_filename = str(report.loc[numeric_ssim.idxmax(), "image_filename"])
            metric_items.extend(
                [
                    ("Worst-case image by SSIM", filename_with_tooltip(worst_filename, max_length=22), worst_filename),
                    ("Best-case image by SSIM", filename_with_tooltip(best_filename, max_length=22), best_filename),
                ]
            )
        st.markdown(metric_grid(metric_items, "metric-grid-5"), unsafe_allow_html=True)

        detailed_report = make_batch_sr_detailed_report(report)
        summary_report = make_batch_sr_summary_report(report, scale_label)
        st.caption(CSV_HELPER_TEXT)
        st.caption(f"Rows to export: {len(detailed_report)}")
        st.download_button(
            "Download batch SR detailed CSV",
            data=detailed_report.to_csv(index=False).encode("utf-8"),
            file_name="batch_sr_detailed_report.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download batch SR summary CSV",
            data=summary_report.to_csv(index=False).encode("utf-8"),
            file_name="batch_sr_summary_report.csv",
            mime="text/csv",
        )


def show_why_sr_matters() -> None:
    with st.expander("Research context", expanded=False):
        col1, col2, col3 = st.columns(3, gap="large")

        with col1.container(border=True):
            st.markdown("**Throughput vs resolution tradeoff**")
            st.write("Resolution gains can increase acquisition, storage, and model-compute costs.")

        with col2.container(border=True):
            st.markdown("**Microscopy/pathology relevance**")
            st.write("Local structure matters when reviewing tissue, cellular, or slide-level imagery.")

        with col3.container(border=True):
            st.markdown("**Future enhancement models**")
            st.write("Bicubic is a transparent baseline for later learned or diffusion-based methods.")


def show_value_statement() -> None:
    st.markdown(
        """
        <div class="disclaimer">
            BioMedVisionLab is designed as a reusable visual evaluation workbench for biomedical
            image-like data, including radiology images, microscopy images, and genomic contact maps.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-title">Why this prototype matters</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="large")
    with col1.container(border=True):
        st.markdown('<span class="badge badge-xai">Inspect</span>', unsafe_allow_html=True)
        st.markdown("**Inspect model behavior visually**")
        st.write("Compare queries, retrieved cases, and image-like outputs side by side.")
    with col2.container(border=True):
        st.markdown('<span class="badge badge-retrieval">Evaluate</span>', unsafe_allow_html=True)
        st.markdown("**Evaluate retrieval reliability**")
        st.write("Track similarity, agreement, confidence, and Top-k sensitivity.")
    with col3.container(border=True):
        st.markdown('<span class="badge badge-sr">Compare</span>', unsafe_allow_html=True)
        st.markdown("**Compare resolution behavior**")
        st.write("Review low-resolution versus enhanced biomedical image-like data.")


def show_alignment_card(title: str, badge_html: str, rows: list[tuple[str, str]]) -> None:
    with st.container(border=True):
        st.markdown(badge_html, unsafe_allow_html=True)
        st.markdown(f"**{title}**")
        st.markdown(alignment_rows(rows), unsafe_allow_html=True)


def make_hic_report(
    matrix_source: str,
    matrix_size: int,
    transform_used: str,
    downsample_factor: str,
    matrix_mse: float,
    matrix_psnr: float,
    matrix_ssim: float | None,
    diagonal_score: float,
    note: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "matrix_source": matrix_source,
                "matrix_size": matrix_size,
                "transform_used": transform_used,
                "downsample_factor": downsample_factor,
                "mse": matrix_mse,
                "psnr": matrix_psnr,
                "ssim": "" if matrix_ssim is None else matrix_ssim,
                "diagonal_preservation_score": diagonal_score,
                "note": note,
            }
        ]
    )


def render_hic_mini_demo() -> None:
    st.markdown('<div class="section-title">Hi-C Contact Map Mini-Demo</div>', unsafe_allow_html=True)
    contact_upload_types = ["npy", "csv", "cool", "mcool"] if cooler is not None else ["npy", "csv"]
    if cooler is None:
        st.info("Install cooler to enable .cool and .mcool uploads. CSV and NPY contact matrices are still supported.")
    uploaded_matrix = st.file_uploader("Upload real contact matrix", type=contact_upload_types)
    control_col1, control_col2, control_col3 = st.columns(3)
    matrix_size = control_col1.selectbox("Synthetic matrix size", [64, 128, 256], index=1)
    downsample_label = control_col2.selectbox("Downsample factor", ["2x", "4x"])
    colormap = control_col3.selectbox("Colormap", ["viridis", "magma", "inferno"])
    option_col1, option_col2, option_col3 = st.columns(3)
    use_log1p = option_col1.checkbox("Use log1p transform", value=True)
    noise_level = option_col2.slider("Synthetic noise level", min_value=0.0, max_value=0.25, value=0.05, step=0.01)
    max_matrix_size = option_col3.selectbox("Max matrix size for display", [256, 512, 1024], index=0)
    downsample_factor = int(downsample_label.replace("x", ""))

    data_note = "Synthetic data only; not biological results"
    matrix_source = "synthetic"
    is_synthetic = True
    if uploaded_matrix is not None:
        try:
            suffix = Path(uploaded_matrix.name).suffix.lower()
            if suffix in {".cool", ".mcool"}:
                if cooler is None:
                    st.error("Install cooler to upload .cool or .mcool Hi-C contact maps.")
                    return
                temp_path = save_uploaded_file_to_temp(uploaded_matrix)
                try:
                    groups = cooler_groups_for_file(temp_path)
                    selected_group = None
                    if suffix == ".mcool":
                        if not groups:
                            st.error("No cooler groups were found in this .mcool file.")
                            return
                        st.caption(f"Available .mcool groups: {', '.join(groups[:8])}" + (" ..." if len(groups) > 8 else ""))
                        selected_group = st.selectbox("Cooler resolution/group", groups)
                    elif groups:
                        st.caption(f"Available cooler groups: {', '.join(groups[:8])}" + (" ..." if len(groups) > 8 else ""))

                    cooler_obj, _ = open_cooler_from_temp_path(temp_path, selected_group)
                    region_options = cooler_region_options(cooler_obj)
                    selected_region = None
                    if region_options:
                        selected_region = st.selectbox("Chromosome/region", region_options)
                    raw_matrix = fetch_cooler_matrix(cooler_obj, selected_region, max_matrix_size)
                    if max(raw_matrix.shape) >= max_matrix_size:
                        st.warning("Please select a smaller region or lower display size.")
                    st.info(
                        "This viewer supports small-region visualization from .cool/.mcool files. "
                        "It is not a full Hi-C browser."
                    )
                finally:
                    try:
                        Path(temp_path).unlink(missing_ok=True)
                    except OSError:
                        pass
            else:
                raw_matrix = load_uploaded_contact_matrix(uploaded_matrix)
            if raw_matrix.shape[0] != raw_matrix.shape[1]:
                st.warning(f"Uploaded matrix is {raw_matrix.shape[0]} x {raw_matrix.shape[1]}, not square.")
                if not st.checkbox("Crop uploaded matrix to square", value=True):
                    st.info("Enable square cropping to continue with image-like contact-map evaluation.")
                    return
                raw_matrix = crop_matrix_to_square(raw_matrix)
            original_matrix = prepare_contact_matrix_for_visualization(raw_matrix, use_log1p)
            matrix_size = int(original_matrix.shape[0])
            data_note = (
                "Uploaded contact map visualized as a normalized matrix image; no loop, compartment, "
                "TAD, or biological interpretation."
            )
            matrix_source = uploaded_matrix.name
            is_synthetic = False
            st.info(
                "Contact maps are visualized as matrix images. This prototype does not perform loop calling, "
                "compartment calling, TAD calling, or biological interpretation."
            )
        except ValueError as error:
            st.error(str(error))
            return
    else:
        original_matrix = prepare_contact_matrix_for_visualization(
            generate_synthetic_contact_map(matrix_size, noise_level),
            use_log1p,
        )

    st.markdown(
        '<span class="badge badge-genomics">Uploaded contact map</span>'
        if not is_synthetic
        else '<span class="badge badge-prototype">Synthetic contact map</span>',
        unsafe_allow_html=True,
    )

    low_matrix, upscaled_matrix = simulate_low_resolution_matrix(original_matrix, downsample_factor)
    difference_matrix = np.abs(original_matrix - upscaled_matrix)
    matrix_mse, matrix_psnr, matrix_ssim = calculate_matrix_metrics(original_matrix, upscaled_matrix)
    diagonal_score = diagonal_preservation_score(original_matrix, upscaled_matrix)

    st.markdown(
        metric_grid(
            [
                ("Matrix MSE", f"{matrix_mse:.5f}"),
                ("Matrix PSNR", "inf dB" if np.isinf(matrix_psnr) else f"{matrix_psnr:.2f} dB"),
                ("Matrix SSIM", "Install scikit-image" if matrix_ssim is None else f"{matrix_ssim:.3f}"),
                ("Diagonal preservation", f"{diagonal_score:.3f}"),
            ],
            "metric-grid-4",
        ),
        unsafe_allow_html=True,
    )

    image_col1, image_col2, image_col3, image_col4 = st.columns(4, gap="large")
    with image_col1:
        st.markdown(image_frame_html(matrix_to_heatmap_image(original_matrix, colormap), "crop"), unsafe_allow_html=True)
        st.caption("Uploaded contact map" if not is_synthetic else "Original synthetic contact map")
    with image_col2:
        st.markdown(image_frame_html(matrix_to_heatmap_image(low_matrix, colormap), "crop"), unsafe_allow_html=True)
        st.caption("Simulated low-resolution contact map")
    with image_col3:
        st.markdown(image_frame_html(matrix_to_heatmap_image(upscaled_matrix, colormap), "crop"), unsafe_allow_html=True)
        st.caption("Bicubic-upscaled contact map")
    with image_col4:
        st.markdown(image_frame_html(matrix_to_heatmap_image(difference_matrix, colormap), "crop"), unsafe_allow_html=True)
        st.caption("Absolute difference map")

    with st.expander("Diagonal-insulation style simple profile", expanded=False):
        st.caption("Mean normalized signal by distance from the diagonal; this is a visual summary, not biological interpretation.")
        original_profile = diagonal_band_profile(original_matrix)
        upscaled_profile = diagonal_band_profile(upscaled_matrix)
        profile_df = pd.concat(
            [
                original_profile.assign(Matrix="Uploaded/original"),
                upscaled_profile.assign(Matrix="Bicubic-upscaled"),
            ],
            ignore_index=True,
        )
        if alt is not None:
            chart = (
                alt.Chart(profile_df)
                .mark_line()
                .encode(
                    x=alt.X("Distance from diagonal:Q", title="Distance from diagonal"),
                    y=alt.Y("Mean signal:Q", title="Mean normalized signal"),
                    color=alt.Color("Matrix:N", title=None),
                    tooltip=["Matrix:N", "Distance from diagonal:Q", alt.Tooltip("Mean signal:Q", format=".4f")],
                )
                .properties(height=220)
            )
            st.altair_chart(chart, width="stretch")
        else:
            st.line_chart(profile_df.pivot(index="Distance from diagonal", columns="Matrix", values="Mean signal"))

    if uploaded_matrix is None:
        st.info(
            "Synthetic contact maps are used only to demonstrate the visualization interface. "
            "Future work should use real Hi-C/Micro-C data."
        )
    st.warning(
        "Contact maps are visualized as matrix images. This prototype does not perform loop calling, "
        "compartment calling, TAD calling, or biological interpretation."
    )

    report = make_hic_report(
        matrix_source,
        matrix_size,
        "log1p + min-max" if use_log1p else "min-max",
        downsample_label,
        matrix_mse,
        matrix_psnr,
        matrix_ssim,
        diagonal_score,
        data_note,
    )
    st.caption(CSV_HELPER_TEXT)
    st.download_button(
        "Download contact-map evaluation CSV",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="hic_matrix_evaluation_report.csv",
        mime="text/csv",
    )


def render_grant_alignment_lab() -> None:
    st.markdown('<div class="section-title">Grant Alignment Lab</div>', unsafe_allow_html=True)
    show_value_statement()

    chem_col, hic_col = st.columns(2, gap="large")
    with chem_col:
        show_alignment_card(
            "Chemical Imaging Super-Resolution",
            '<span class="badge badge-sr">ChemDiffuse Alignment</span>',
            [
                ("Research challenge", "Low-resolution biomedical/chemical imaging limits fine-structure inspection."),
                (
                    "Current prototype support",
                    "Downsampling simulation, bicubic baseline, PSNR, SSIM, MSE, crop inspection, artifact warning.",
                ),
                (
                    "Next extension",
                    "Diffusion-based super-resolution, microscopy/pathology datasets, SRS/Raman-style image support.",
                ),
            ],
        )
    with hic_col:
        show_alignment_card(
            "Hi-C Contact Map Visualization",
            '<span class="badge badge-genomics">Chromatin Maps</span>',
            [
                ("Research challenge", "Hi-C contact matrices are sparse, noisy, high-dimensional, and image-like."),
                (
                    "Current prototype support",
                    "Matrix visualization, similarity/retrieval evaluation, resolution enhancement baseline concepts.",
                ),
                (
                    "Next extension",
                    "Real Hi-C upload, low-to-high matrix comparison, contact-map retrieval, RNA-seq/ATAC-seq metadata.",
                ),
            ],
    )

    render_hic_mini_demo()
    show_grant_exploration_hub()

    with st.expander("Grant relevance notes", expanded=False):
        st.write(
            "The workbench pattern can be reused across biomedical image-like data: radiology images, "
            "microscopy and chemical imaging, pathology tiles, and genomic contact matrices."
        )

    with st.expander("Limitations", expanded=False):
        st.write(
            "This prototype is not for diagnosis. Bicubic super-resolution is not a trained AI model. "
            "Synthetic Hi-C contact maps are interface demos, not biological results."
        )


def show_research_extension_roadmap() -> None:
    st.markdown('<div class="section-title">Research Extension Roadmap</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Possible extensions toward current biomedical imaging and multimodal biology research.</div>',
        unsafe_allow_html=True,
    )
    roadmap_items = [
        ("Encoder", "badge-encoder", "CXR-specific encoders", "TorchXRayVision or medical foundation models"),
        ("Explainability", "badge-xai", "Grad-CAM grounding", "Model-based localization previews"),
        ("Text", "badge-text", "Report-aware retrieval", "Radiology text plus image signals"),
        ("Microscopy", "badge-micro", "Pathology datasets", "Biomedical super-resolution benchmarks"),
        ("Diffusion", "badge-sr", "Image enhancement", "Diffusion-based enhancement comparisons"),
        ("Genomics", "badge-genomics", "Hi-C contact maps", "Contact matrix visualization and retrieval"),
    ]

    for row_start in range(0, len(roadmap_items), 3):
        columns = st.columns(3, gap="large")
        for column, (tag, badge_class, title, text) in zip(columns, roadmap_items[row_start : row_start + 3]):
            with column.container(border=True):
                st.markdown(f'<span class="badge {badge_class}">{escape(tag)}</span>', unsafe_allow_html=True)
                st.markdown(f"**{title}**")
                st.write(text)


def show_sr_exploration_studio() -> None:
    st.markdown('<div class="section-title">Super-Resolution Exploration Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Use these prompts to compare baseline behavior across biomedical image types and model outputs.</div>',
        unsafe_allow_html=True,
    )
    items = [
        ("Model Output", "badge-sr", "Upload learned output", "Compare diffusion, CNN, or transformer SR outputs against bicubic."),
        ("Crops", "badge-imaging", "Inspect local structure", "Review center crops for smoothing, edge loss, and texture changes."),
        ("Metrics", "badge-retrieval", "Track method deltas", "Use PSNR, SSIM, and MSE deltas for experiment tracking."),
        ("Artifacts", "badge-xai", "Check risk patterns", "Look for oversmoothing or hallucinated structure before claiming improvement."),
    ]
    columns = st.columns(4, gap="large")
    for column, (tag, badge_class, title, text) in zip(columns, items):
        with column.container(border=True):
            st.markdown(f'<span class="badge {badge_class}">{escape(tag)}</span>', unsafe_allow_html=True)
            st.markdown(f"**{title}**")
            st.write(text)

    with st.expander("Suggested exploration workflows", expanded=False):
        st.write(
            "Try the same image at 2x and 4x, compare the bicubic baseline to an uploaded model output, "
            "then inspect crop-level differences before interpreting metric changes."
        )


def show_grant_exploration_hub() -> None:
    st.markdown('<div class="section-title">Multimodal Exploration Hub</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Explore how the same visual evaluation workbench can support multiple biomedical image-like data types.</div>',
        unsafe_allow_html=True,
    )
    items = [
        ("Matrix Upload", "badge-genomics", "Real contact-map slices", "Use CSV, NPY, COOL, or MCOOL small regions for visual evaluation."),
        ("Resolution", "badge-sr", "Low-to-high comparison", "Compare downsampled and bicubic-upscaled matrix images transparently."),
        ("Profiles", "badge-xai", "Diagonal signal profile", "Inspect distance-from-diagonal summaries without biological calls."),
        ("Metadata", "badge-text", "Future multimodal links", "Connect imaging outputs to RNA-seq, ATAC-seq, reports, or sample metadata."),
    ]
    columns = st.columns(4, gap="large")
    for column, (tag, badge_class, title, text) in zip(columns, items):
        with column.container(border=True):
            st.markdown(f'<span class="badge {badge_class}">{escape(tag)}</span>', unsafe_allow_html=True)
            st.markdown(f"**{title}**")
            st.write(text)

    with st.expander("Explore beyond the demo", expanded=False):
        st.write(
            "Use smaller representative regions, compare transform settings, and export CSV metrics to document "
            "which visualization choices were used. This is an evaluation interface, not a biological interpretation engine."
        )


def render_super_resolution_tab(index_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> None:
    st.markdown('<div class="section-title">Biomedical Image Super-Resolution Demo</div>', unsafe_allow_html=True)
    st.markdown('<span class="badge badge-ai">Baseline method: Bicubic interpolation</span>', unsafe_allow_html=True)
    if index_state is None or len(index_state[0]) == 0:
        st.markdown('<span class="badge badge-prototype">Upload-only mode</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-normal">Local CXR index available</span>', unsafe_allow_html=True)
    scale_label = st.selectbox("Scale factor", ["2x", "4x"], help="Downsample first, then reconstruct to original size.")
    scale_factor = int(scale_label.replace("x", ""))

    prepared = prepare_super_resolution_image(index_state)
    if prepared is None:
        return

    original, filename, label = prepared
    low_res, enhanced = simulate_low_resolution(original, scale_factor)
    mse = calculate_mse(original, enhanced)
    psnr = calculate_psnr(original, enhanced)
    ssim = calculate_ssim(original, enhanced)

    st.markdown(
        metric_grid(
            [
                ("Scale factor", scale_label),
                ("PSNR", "inf dB" if np.isinf(psnr) else f"{psnr:.2f} dB"),
                ("SSIM", "Install scikit-image" if ssim is None else f"{ssim:.3f}"),
                ("MSE", f"{mse:.2f}"),
                ("Original size", f"{original.size[0]} x {original.size[1]}"),
            ],
            "metric-grid-5",
        ),
        unsafe_allow_html=True,
    )

    sr_info_col, artifact_col = st.columns([1, 1], gap="large")
    with sr_info_col:
        show_sr_model_card()
    with artifact_col:
        st.markdown(
            """
            <div class="artifact-card">
                Artifact risk: Bicubic interpolation can smooth fine structures and does not recover
                true missing anatomical detail.
            </div>
            """,
            unsafe_allow_html=True,
        )

    sr_report = make_single_sr_report(filename, scale_label, original.size, psnr, ssim, mse)
    st.caption(CSV_HELPER_TEXT)
    st.download_button(
        "Download single-image SR report",
        data=sr_report.to_csv(index=False).encode("utf-8"),
        file_name="single_sr_report.csv",
        mime="text/csv",
    )

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        show_sr_panel("Original image", original, f"{filename} | {label}")
    with col2:
        show_sr_panel("Simulated low-resolution image", low_res, f"Downsampled by {scale_label}")
    with col3:
        show_sr_panel("Bicubic enhanced image", enhanced, "Reconstructed to original size")

    show_model_output_comparison(original, low_res, enhanced, filename, scale_label)
    show_sr_large_preview(original, low_res, enhanced)

    max_crop = max(16, min(original.size))
    default_crop = min(128, max_crop)
    crop_size = st.slider(
        "Center crop size",
        min_value=16,
        max_value=max_crop,
        value=default_crop,
        step=8 if max_crop >= 64 else 1,
    )
    show_zoomed_crop_comparison(original, low_res, enhanced, crop_size)
    show_batch_sr_evaluation(index_state)
    show_sr_exploration_studio()
    show_why_sr_matters()

    with st.expander("Method notes", expanded=False):
        st.write(
            "This module demonstrates degradation and reconstruction with a transparent bicubic baseline. "
            "It is not a clinical enhancement system."
        )
        st.warning("Bicubic super-resolution is an educational baseline, not a trained AI model.")


def try_load_index() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    available_indexes = discover_embedding_indexes()
    if not available_indexes:
        return None
    try:
        image_paths, labels = load_index_manifest(str(next(iter(available_indexes.values()))))
        return image_paths, labels, np.empty((len(image_paths), 0), dtype=np.float32)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not load embeddings: {error}")
        return None


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="BVL", layout="wide")
    inject_css()
    show_hero()
    show_disclaimer()
    show_startup_checks()

    index_state = try_load_index()
    retrieval_tab, sr_tab, grant_tab = st.tabs(["CXR Retrieval", "Super-Resolution Demo", "Grant Alignment Lab"])

    with retrieval_tab:
        render_retrieval_tab(index_state)
        show_research_extension_roadmap()

    with sr_tab:
        render_super_resolution_tab(index_state)

    with grant_tab:
        render_grant_alignment_lab()


if __name__ == "__main__":
    main()

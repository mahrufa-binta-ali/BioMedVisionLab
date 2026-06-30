import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

try:
    import torchxrayvision as xrv
except ImportError:
    xrv = None


DATA_DIR = Path("data/chest_xray")
EMBEDDINGS_ROOT = Path("embeddings")
ENCODER_TYPE = "resnet18_imagenet"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VALID_LABELS = {"NORMAL", "PNEUMONIA"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EncoderBundle:
    def __init__(self, key: str, name: str, model: nn.Module, transform: transforms.Compose, output_dim: int):
        self.key = key
        self.name = name
        self.model = model
        self.transform = transform
        self.output_dim = output_dim


ENCODER_CONFIGS = {
    "resnet18_imagenet": {
        "name": "ResNet18 ImageNet baseline",
        "output_dim": 512,
    },
    "resnet50_imagenet": {
        "name": "ResNet50 ImageNet baseline",
        "output_dim": 2048,
    },
    "torchxrayvision_densenet121": {
        "name": "TorchXRayVision DenseNet121 CXR encoder",
        "output_dim": 1024,
    },
}


def canonical_encoder_name(encoder_name: str) -> str:
    aliases = {
        "resnet18": "resnet18_imagenet",
        "resnet50": "resnet50_imagenet",
        "torchxrayvision": "torchxrayvision_densenet121",
    }
    return aliases.get(encoder_name, encoder_name)


def find_image_files(data_dir: Path) -> list[Path]:
    """Find supported image files that live under NORMAL or PNEUMONIA folders."""
    image_files = []

    for path in data_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        if infer_label(path) in VALID_LABELS:
            image_files.append(path)

    return sorted(image_files)


def infer_label(image_path: Path) -> str:
    """Infer a chest X-ray label from any parent folder named NORMAL or PNEUMONIA."""
    for parent in image_path.parents:
        label = parent.name.upper()
        if label in VALID_LABELS:
            return label
    return "UNKNOWN"


def get_resnet_transform() -> transforms.Compose:
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


def load_encoder(encoder_name: str, device: torch.device | None = None) -> EncoderBundle:
    encoder_name = canonical_encoder_name(encoder_name)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if encoder_name == "torchxrayvision_densenet121":
        if xrv is None:
            raise ImportError(
                "torchxrayvision is not installed. Install requirements or use --encoder resnet18_imagenet."
            )
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model.op_threshs = None
        model.to(device)
        model.eval()
        return EncoderBundle(
            key="torchxrayvision_densenet121",
            name="TorchXRayVision DenseNet121 CXR encoder",
            model=model,
            transform=get_xrv_transform(),
            output_dim=1024,
        )

    if encoder_name == "resnet50_imagenet":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        output_dim = 2048
        encoder_label = "ResNet50 ImageNet baseline"
    else:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        output_dim = 512
        encoder_label = "ResNet18 ImageNet baseline"
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return EncoderBundle(
        key=encoder_name,
        name=encoder_label,
        model=model,
        transform=get_resnet_transform(),
        output_dim=output_dim,
    )


def preprocess_image_for_encoder(image: Image.Image, encoder_name: str) -> torch.Tensor:
    encoder_name = canonical_encoder_name(encoder_name)
    transform = get_xrv_transform() if encoder_name == "torchxrayvision_densenet121" else get_resnet_transform()
    return transform(image).unsqueeze(0)


def load_rgb_image(image_path: Path) -> Image.Image:
    """Open a chest X-ray and convert grayscale images to 3-channel RGB."""
    return Image.open(image_path).convert("RGB")


def extract_embedding(
    image: Image.Image,
    encoder_name: str,
    encoder: EncoderBundle | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Extract one L2-normalized embedding."""
    encoder_name = canonical_encoder_name(encoder_name)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder or load_encoder(encoder_name, device)
    image_tensor = preprocess_image_for_encoder(image, encoder_name).to(device)

    with torch.no_grad():
        if encoder_name == "torchxrayvision_densenet121":
            features = encoder.model.features(image_tensor)
            features = torch.nn.functional.relu(features, inplace=False)
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            embedding = torch.flatten(features, 1).cpu().numpy()[0]
        else:
            embedding = encoder.model(image_tensor).cpu().numpy()[0]

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.astype(np.float32)


def extract_embedding_from_path(
    image_path: Path,
    encoder: EncoderBundle,
    device: torch.device,
) -> np.ndarray | None:
    try:
        image = load_rgb_image(image_path)
    except (OSError, UnidentifiedImageError) as error:
        print(f"Skipping unreadable image: {image_path} ({error})")
        return None
    return extract_embedding(image, encoder.key, encoder, device)


def encoder_feature_note(encoder_key: str) -> str:
    if encoder_key == "torchxrayvision_densenet121":
        return "TorchXRayVision DenseNet121 intermediate convolutional features with global average pooling."
    if encoder_key == "resnet50_imagenet":
        return "ResNet50 penultimate 2048-D feature vector."
    return "ResNet18 penultimate 512-D feature vector."


def build_index(encoder_type: str) -> None:
    encoder_type = canonical_encoder_name(encoder_type)
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            "Dataset folder was not found. Please place images under data/chest_xray/ "
            "with folders such as train/NORMAL and train/PNEUMONIA."
        )

    image_files = find_image_files(DATA_DIR)
    if not image_files:
        raise ValueError(
            "No JPG, JPEG, or PNG images were found under data/chest_xray inside "
            "NORMAL or PNEUMONIA folders."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = load_encoder(encoder_type, device)
    print(f"Using encoder: {encoder.name}")

    indexed_paths = []
    labels = []
    embeddings = []

    for image_path in image_files:
        embedding = extract_embedding_from_path(image_path, encoder, device)
        if embedding is None:
            continue

        indexed_paths.append(str(image_path))
        labels.append(infer_label(image_path))
        embeddings.append(embedding)

    if not embeddings:
        raise ValueError("Images were found, but none could be read successfully.")

    output_dir = EMBEDDINGS_ROOT / encoder.key
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "image_paths.npy", np.array(indexed_paths, dtype=object))
    np.save(output_dir / "labels.npy", np.array(labels, dtype=object))
    np.save(output_dir / "embeddings.npy", np.vstack(embeddings).astype(np.float32))

    metadata = {
        "encoder_key": encoder.key,
        "encoder_name": encoder.name,
        "embedding_dimension": int(np.vstack(embeddings).shape[1]),
        "dataset_size": len(indexed_paths),
        "build_time": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "feature_note": encoder_feature_note(encoder.key),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Indexed {len(indexed_paths)} images.")
    print(f"Saved embeddings to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BioMedVisionLab CXR embedding index.")
    parser.add_argument(
        "--encoder",
        choices=[
            "resnet18_imagenet",
            "resnet50_imagenet",
            "torchxrayvision_densenet121",
            "resnet18",
            "resnet50",
            "torchxrayvision",
        ],
        default=ENCODER_TYPE,
        help="Encoder used to build embeddings.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        build_index(args.encoder)
    except (FileNotFoundError, ValueError, ImportError) as error:
        print(f"Error: {error}")
        raise SystemExit(1)

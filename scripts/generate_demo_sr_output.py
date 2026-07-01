from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter


def main() -> None:
    src = Path("data/chest_xray/test/NORMAL/IM-0001-0001.jpeg")
    out_dir = Path("demo_outputs")
    out_dir.mkdir(exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(
            "Expected demo source image not found. "
            "Please place the chest_xray dataset under data/chest_xray first."
        )

    img = Image.open(src).convert("RGB")

    low = img.resize(
        (max(1, img.width // 2), max(1, img.height // 2)),
        Image.Resampling.BICUBIC,
    )
    up = low.resize(img.size, Image.Resampling.BICUBIC)

    enhanced = ImageEnhance.Sharpness(up).enhance(1.8)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.08)
    enhanced = enhanced.filter(
        ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3)
    )

    output_path = out_dir / "demo_external_sr_output_IM-0001-0001.png"
    enhanced.save(output_path)
    print(f"Saved demo output to: {output_path}")


if __name__ == "__main__":
    main()
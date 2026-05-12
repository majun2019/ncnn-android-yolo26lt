#!/usr/bin/env python3
"""Generate styled PNG assets for paper figures."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "runs" / "paper_figures"

BACKGROUND = (250, 249, 245)
LABEL_BG = (35, 40, 48)
LABEL_FG = (245, 247, 250)
BORDER = (198, 202, 208)
PANEL_BORDER = 2
LABEL_HEIGHT = 34
OUTER_PADDING = 16

FIGURES = [
    {
        "label": "08A TRAIN",
        "source": ROOT / "runs" / "detect" / "runs" / "train" / "yolo26n_safehat" / "results.png",
        "target": OUTPUT_DIR / "figure8a_train_results_styled.png",
        "max_width": 1400,
    },
    {
        "label": "08B VAL",
        "source": ROOT / "runs" / "detect" / "runs" / "train" / "yolo26n_safehat" / "val_batch0_pred.jpg",
        "target": OUTPUT_DIR / "figure8b_val_pred_styled.png",
        "max_width": 1240,
    },
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    target_height = round(image.height * max_width / image.width)
    return image.resize((max_width, target_height), Image.Resampling.LANCZOS)


def framed_figure(image: Image.Image, label: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> Image.Image:
    panel_width = image.width + OUTER_PADDING * 2
    panel_height = image.height + OUTER_PADDING * 2 + LABEL_HEIGHT
    canvas = Image.new("RGB", (panel_width, panel_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 0, panel_width - 1, panel_height - 1), outline=BORDER, width=1)
    draw.rectangle((PANEL_BORDER, PANEL_BORDER, panel_width - PANEL_BORDER - 1, LABEL_HEIGHT), fill=LABEL_BG)
    draw.rectangle(
        (PANEL_BORDER, LABEL_HEIGHT, panel_width - PANEL_BORDER - 1, panel_height - PANEL_BORDER - 1),
        outline=BORDER,
        width=1,
    )
    draw.text((14, 8), label, fill=LABEL_FG, font=font)
    canvas.paste(image, (OUTER_PADDING, LABEL_HEIGHT + OUTER_PADDING))
    return canvas


def build_single_assets(font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> None:
    for figure in FIGURES:
        image = Image.open(figure["source"]).convert("RGB")
        image = fit_image(image, figure["max_width"])
        styled = framed_figure(image, figure["label"], font)
        styled.save(figure["target"], format="PNG")
        print(f"saved: {figure['target']}")


def export_montage_png() -> None:
    png_source = ROOT / "runs" / "paper_figures" / "figure8c_five_task_montage.png"
    jpg_source = ROOT / "runs" / "androidtest" / "figure8_five_task_montage.jpg"
    target = OUTPUT_DIR / "figure8c_five_task_montage.png"
    source = png_source if png_source.exists() else jpg_source
    image = Image.open(source).convert("RGB")
    image.save(target, format="PNG")
    print(f"saved: {target}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    font = load_font(18)
    build_single_assets(font)
    export_montage_png()


if __name__ == "__main__":
    main()
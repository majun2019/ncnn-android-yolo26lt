#!/usr/bin/env python3
"""Build a paper-ready five-task montage for Figure 8(c)."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PREVIEW_DIR = ROOT / "video" / "previews"
OUTPUT_PATH = ROOT / "runs" / "androidtest" / "figure8_five_task_montage.jpg"
OUTPUT_PNG_PATH = ROOT / "runs" / "paper_figures" / "figure8c_five_task_montage.png"

PANELS = [
    ("01 DET", PREVIEW_DIR / "det_40.jpg"),
    ("02 SEG", PREVIEW_DIR / "seg_80.jpg"),
    ("03 POSE", PREVIEW_DIR / "pose_140.jpg"),
    ("04 CLS", PREVIEW_DIR / "cls_160.jpg"),
    ("05 OBB", PREVIEW_DIR / "obb1_200.jpg"),
]

TILE_WIDTH = 280
LABEL_HEIGHT = 34
OUTER_MARGIN = 24
COLUMN_GAP = 18
ROW_GAP = 18
PANEL_BORDER = 2
BACKGROUND = (250, 249, 245)
LABEL_BG = (35, 40, 48)
LABEL_FG = (245, 247, 250)
BORDER = (198, 202, 208)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]
    for font_path in candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def resize_panel(image_path: Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    target_height = round(image.height * TILE_WIDTH / image.width)
    return image.resize((TILE_WIDTH, target_height), Image.Resampling.LANCZOS)


def build_panel(label: str, image_path: Path, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> Image.Image:
    image = resize_panel(image_path)
    panel_width = TILE_WIDTH + PANEL_BORDER * 2
    panel_height = LABEL_HEIGHT + image.height + PANEL_BORDER * 2
    panel = Image.new("RGB", (panel_width, panel_height), BACKGROUND)
    draw = ImageDraw.Draw(panel)

    draw.rectangle((0, 0, panel_width - 1, panel_height - 1), outline=BORDER, width=1)
    draw.rectangle((PANEL_BORDER, PANEL_BORDER, panel_width - PANEL_BORDER - 1, LABEL_HEIGHT), fill=LABEL_BG)
    draw.rectangle(
        (PANEL_BORDER, LABEL_HEIGHT, panel_width - PANEL_BORDER - 1, panel_height - PANEL_BORDER - 1),
        outline=BORDER,
        width=1,
    )
    draw.text((16, 8), label, fill=LABEL_FG, font=font)
    panel.paste(image, (PANEL_BORDER, LABEL_HEIGHT))
    return panel


def main() -> None:
    font = load_font(18)
    panels = [build_panel(label, image_path, font) for label, image_path in PANELS]

    top_row = panels[:3]
    bottom_row = panels[3:]
    panel_width = top_row[0].width
    panel_height = top_row[0].height

    canvas_width = OUTER_MARGIN * 2 + panel_width * 3 + COLUMN_GAP * 2
    canvas_height = OUTER_MARGIN * 2 + panel_height * 2 + ROW_GAP
    canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)

    top_y = OUTER_MARGIN
    for index, panel in enumerate(top_row):
        x = OUTER_MARGIN + index * (panel_width + COLUMN_GAP)
        canvas.paste(panel, (x, top_y))

    bottom_total_width = panel_width * 2 + COLUMN_GAP
    bottom_start_x = (canvas_width - bottom_total_width) // 2
    bottom_y = OUTER_MARGIN + panel_height + ROW_GAP
    for index, panel in enumerate(bottom_row):
        x = bottom_start_x + index * (panel_width + COLUMN_GAP)
        canvas.paste(panel, (x, bottom_y))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PATH, quality=95)
    canvas.save(OUTPUT_PNG_PATH, format="PNG")
    print(f"saved: {OUTPUT_PATH}")
    print(f"saved: {OUTPUT_PNG_PATH}")
    print(f"size: {canvas_width}x{canvas_height}")


if __name__ == "__main__":
    main()
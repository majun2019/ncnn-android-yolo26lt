from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PREVIEW_DIR = ROOT / "testpics" / "previews"
OUTPUT_PNG_PATH = ROOT / "runs" / "paper_figures" / "figure8c_five_task_montage.png"
JOURNAL_PATH = ROOT / "testpics" / "Figure_8c.png"

PANELS = [
    ("(a) Detection",      PREVIEW_DIR / "det_40.jpg"),
    ("(b) Segmentation",   PREVIEW_DIR / "seg_80.jpg"),
    ("(c) Pose",           PREVIEW_DIR / "pose_140.jpg"),
    ("(d) Classification", PREVIEW_DIR / "cls_160.jpg"),
    ("(e) OBB",            PREVIEW_DIR / "obb_road.png"),
]

PANEL_IMG_H = 210
LABEL_HEIGHT = 24
PANEL_BORDER = 1
COLUMN_GAP = 12
ROW_GAP = 14
OUTER_MARGIN = 16

BACKGROUND  = (255, 255, 255)
BORDER_COLOR = (180, 185, 192)
LABEL_BG    = (246, 247, 248)
LABEL_FG    = (30, 30, 30)

def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]
    for fp in candidates:
        if fp.exists():
            return ImageFont.truetype(str(fp), size=size)
    return ImageFont.load_default()

def prepare_image(image_path: Path) -> Image.Image:
    im = Image.open(image_path).convert("RGB")
    rotated = im.rotate(90, expand=True)
    rw, rh = rotated.size
    new_w = round(rw * PANEL_IMG_H / rh)
    return rotated.resize((new_w, PANEL_IMG_H), Image.Resampling.LANCZOS)

def build_panel(
    label: str,
    image_path: Path,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    img = prepare_image(image_path)
    iw, ih = img.size
    pw = iw + PANEL_BORDER * 2
    ph = ih + LABEL_HEIGHT + PANEL_BORDER * 2
    panel = Image.new("RGB", (pw, ph), BACKGROUND)
    draw = ImageDraw.Draw(panel)

    panel.paste(img, (PANEL_BORDER, PANEL_BORDER))

    ly0 = PANEL_BORDER + ih
    draw.rectangle((0, ly0, pw - 1, ph - 1), fill=LABEL_BG)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((pw - tw) // 2, ly0 + (LABEL_HEIGHT - th) // 2), label, fill=LABEL_FG, font=font)

    draw.rectangle((0, 0, pw - 1, ph - 1), outline=BORDER_COLOR, width=PANEL_BORDER)
    return panel

def main() -> None:
    font = load_font(13)
    panels = [build_panel(label, path, font) for label, path in PANELS]

    top_row    = panels[:3]
    bottom_row = panels[3:]

    pw = top_row[0].width
    ph = top_row[0].height

    row_w = pw * 3 + COLUMN_GAP * 2
    canvas_w = OUTER_MARGIN * 2 + row_w
    canvas_h = OUTER_MARGIN * 2 + ph * 2 + ROW_GAP

    canvas = Image.new("RGB", (canvas_w, canvas_h), BACKGROUND)

    y_top = OUTER_MARGIN
    for i, panel in enumerate(top_row):
        canvas.paste(panel, (OUTER_MARGIN + i * (pw + COLUMN_GAP), y_top))

    y_bot = OUTER_MARGIN + ph + ROW_GAP
    bottom_total = pw * 2 + COLUMN_GAP
    x_bot_start = (canvas_w - bottom_total) // 2
    for i, panel in enumerate(bottom_row):
        canvas.paste(panel, (x_bot_start + i * (pw + COLUMN_GAP), y_bot))

    OUTPUT_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    canvas.save(OUTPUT_PNG_PATH, format="PNG")
    canvas.save(JOURNAL_PATH, format="PNG")
    print(f"saved: {OUTPUT_PNG_PATH}")
    print(f"saved: {JOURNAL_PATH}")
    print(f"size:  {canvas_w}x{canvas_h} px")

if __name__ == "__main__":
    main()

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PREVIEW_DIR = ROOT / "testpics" / "previews"
OUTPUT_PNG_PATH = ROOT / "runs" / "paper_figures" / "figure8c_five_task_montage.png"
JOURNAL_PATH = ROOT / "testpics" / "Figure_8c.png"

UI_CROP_TOP = 333

PANELS = [
    ("(a) Detection",      PREVIEW_DIR / "det_40.jpg"),
    ("(b) Segmentation",   PREVIEW_DIR / "seg_80.jpg"),
    ("(c) Pose",           PREVIEW_DIR / "pose_140.jpg"),
    ("(d) Classification", PREVIEW_DIR / "cls_160.jpg"),
    ("(e) OBB",            PREVIEW_DIR / "obb_road.png"),
]

PANEL_IMG_HEIGHT = 230
LABEL_HEIGHT = 26
PANEL_BORDER = 1
COLUMN_GAP = 10
OUTER_MARGIN = 16

BACKGROUND = (255, 255, 255)
BORDER_COLOR = (180, 185, 192)
LABEL_BG = (246, 247, 248)
LABEL_FG = (30, 30, 30)

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

def prepare_image(image_path: Path) -> Image.Image:
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    cropped = im.crop((0, UI_CROP_TOP, w, h))
    rotated = cropped.rotate(90, expand=True)
    rw, rh = rotated.size
    new_w = round(rw * PANEL_IMG_HEIGHT / rh)
    return rotated.resize((new_w, PANEL_IMG_HEIGHT), Image.Resampling.LANCZOS)

def build_panel(label: str, image_path: Path, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> Image.Image:
    img = prepare_image(image_path)
    iw, ih = img.size
    panel_w = iw + PANEL_BORDER * 2
    panel_h = ih + LABEL_HEIGHT + PANEL_BORDER * 2
    panel = Image.new("RGB", (panel_w, panel_h), BACKGROUND)
    draw = ImageDraw.Draw(panel)

    panel.paste(img, (PANEL_BORDER, PANEL_BORDER))

    label_y0 = PANEL_BORDER + ih
    draw.rectangle((0, label_y0, panel_w - 1, panel_h - 1), fill=LABEL_BG)

    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (panel_w - tw) // 2
    ty = label_y0 + (LABEL_HEIGHT - th) // 2
    draw.text((tx, ty), label, fill=LABEL_FG, font=font)

    draw.rectangle((0, 0, panel_w - 1, panel_h - 1), outline=BORDER_COLOR, width=PANEL_BORDER)
    return panel

def main() -> None:
    font = load_font(13)
    panels = [build_panel(label, path, font) for label, path in PANELS]

    total_w = (
        OUTER_MARGIN * 2
        + sum(p.width for p in panels)
        + COLUMN_GAP * (len(panels) - 1)
    )
    total_h = OUTER_MARGIN * 2 + panels[0].height

    canvas = Image.new("RGB", (total_w, total_h), BACKGROUND)
    x = OUTER_MARGIN
    for panel in panels:
        canvas.paste(panel, (x, OUTER_MARGIN))
        x += panel.width + COLUMN_GAP

    OUTPUT_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    canvas.save(OUTPUT_PNG_PATH, format="PNG")
    canvas.save(JOURNAL_PATH, format="PNG")
    print(f"saved: {OUTPUT_PNG_PATH}")
    print(f"saved: {JOURNAL_PATH}")
    print(f"size:  {total_w}x{total_h} px")

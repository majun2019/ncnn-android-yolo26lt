#!/usr/bin/env python3
"""Build Figure 8(c): five-task Android deployment montage.

Displays full phone screenshots rotated 90° CCW to landscape orientation so
both the App UI (proving on-device deployment) and the inference result are
visible.  Layout: 3 panels on top row + 2 panels centered on bottom row.

Source images: testpics/previews/*.jpg  (616×1280 portrait screenshots)
"""

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
    ("(e) OBB",            PREVIEW_DIR / "obb1_200.jpg"),
]

# After 90° CCW rotation: portrait 616×1280 → landscape 1280×616
# Scale so each rotated phone image is PANEL_IMG_H pixels tall
PANEL_IMG_H = 210        # target height of the rotated phone image (px)
LABEL_HEIGHT = 24        # label bar below each panel (px)
PANEL_BORDER = 1         # thin border around each panel
COLUMN_GAP = 12          # horizontal gap between panels
ROW_GAP = 14             # vertical gap between rows
OUTER_MARGIN = 16        # outer canvas margin

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
    """Load portrait screenshot, rotate 90° CCW → landscape, scale to target height."""
    im = Image.open(image_path).convert("RGB")
    # Rotate 90° CCW: phone lying on its right side (UI on left, scene on right)
    rotated = im.rotate(90, expand=True)   # 616×1280 → 1280×616
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

    # Image area
    panel.paste(img, (PANEL_BORDER, PANEL_BORDER))

    # Label bar
    ly0 = PANEL_BORDER + ih
    draw.rectangle((0, ly0, pw - 1, ph - 1), fill=LABEL_BG)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((pw - tw) // 2, ly0 + (LABEL_HEIGHT - th) // 2), label, fill=LABEL_FG, font=font)

    # Outer border
    draw.rectangle((0, 0, pw - 1, ph - 1), outline=BORDER_COLOR, width=PANEL_BORDER)
    return panel


def main() -> None:
    font = load_font(13)
    panels = [build_panel(label, path, font) for label, path in PANELS]

    top_row    = panels[:3]
    bottom_row = panels[3:]

    pw = top_row[0].width    # all panels share the same width (same source aspect)
    ph = top_row[0].height

    row_w = pw * 3 + COLUMN_GAP * 2          # width driven by the 3-panel top row
    canvas_w = OUTER_MARGIN * 2 + row_w
    canvas_h = OUTER_MARGIN * 2 + ph * 2 + ROW_GAP

    canvas = Image.new("RGB", (canvas_w, canvas_h), BACKGROUND)

    # Top row: 3 panels flush left from OUTER_MARGIN
    y_top = OUTER_MARGIN
    for i, panel in enumerate(top_row):
        canvas.paste(panel, (OUTER_MARGIN + i * (pw + COLUMN_GAP), y_top))

    # Bottom row: 2 panels centered within canvas_w
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

# Phone app UI header to crop from each screenshot before rotating.
# Determined empirically: status bar + app title + button + selectors + sliders
# end at approximately y=333 in a 616×1280 screenshot.
UI_CROP_TOP = 333

PANELS = [
    ("(a) Detection",      PREVIEW_DIR / "det_40.jpg"),
    ("(b) Segmentation",   PREVIEW_DIR / "seg_80.jpg"),
    ("(c) Pose",           PREVIEW_DIR / "pose_140.jpg"),
    ("(d) Classification", PREVIEW_DIR / "cls_160.jpg"),
    ("(e) OBB",            PREVIEW_DIR / "obb1_200.jpg"),
]

# Layout constants
PANEL_IMG_HEIGHT = 230   # target image area height per panel (px)
LABEL_HEIGHT = 26        # label bar below each image (px)
PANEL_BORDER = 1         # border width around each panel
COLUMN_GAP = 10          # gap between panels
OUTER_MARGIN = 16        # outer canvas margin

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
    """Crop UI header, rotate 90° CCW, resize to target height."""
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    # Crop off app UI header
    cropped = im.crop((0, UI_CROP_TOP, w, h))
    # Rotate 90° counter-clockwise: makes landscape scene upright
    rotated = cropped.rotate(90, expand=True)
    # Scale to target height
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

    # Paste image
    panel.paste(img, (PANEL_BORDER, PANEL_BORDER))

    # Label bar below image
    label_y0 = PANEL_BORDER + ih
    draw.rectangle((0, label_y0, panel_w - 1, panel_h - 1), fill=LABEL_BG)

    # Center text in label bar
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = (panel_w - tw) // 2
    ty = label_y0 + (LABEL_HEIGHT - th) // 2
    draw.text((tx, ty), label, fill=LABEL_FG, font=font)

    # Outer border
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

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "runs" / "paper_figures"
TESTPICS_DIR = ROOT / "testpics"

BACKGROUND = (255, 255, 255)
BORDER = (180, 185, 192)
PANEL_BORDER = 1
OUTER_PADDING = 0

GRID_CROP_TOP = 50

FIGURES = [
    {
        "source": ROOT / "runs" / "detect" / "runs" / "train" / "yolo26n_safehat" / "results.png",
        "target": OUTPUT_DIR / "figure8a_train_results_styled.png",
        "journal_name": "Figure_8a.png",
        "max_width": 1400,
    },
    {
        "source": ROOT / "runs" / "detect" / "runs" / "train" / "yolo26n_safehat" / "val_batch0_pred.jpg",
        "target": OUTPUT_DIR / "figure8b_val_pred_styled.png",
        "journal_name": "Figure_8b.png",
        "max_width": 1240,
        "clean_grid": True,
    },
]

def fit_image(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    target_height = round(image.height * max_width / image.width)
    return image.resize((max_width, target_height), Image.Resampling.LANCZOS)

def framed_figure(image: Image.Image) -> Image.Image:
    panel_width = image.width + PANEL_BORDER * 2
    panel_height = image.height + PANEL_BORDER * 2
    canvas = Image.new("RGB", (panel_width, panel_height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, panel_width - 1, panel_height - 1), outline=BORDER, width=PANEL_BORDER)
    canvas.paste(image, (PANEL_BORDER, PANEL_BORDER))
    return canvas

def clean_val_grid(image: Image.Image, cols: int = 3) -> Image.Image:
    W, H = image.size
    rows = H // (W // cols)
    cell_w = W // cols
    cell_h = H // rows
    out_cell_h = cell_h - GRID_CROP_TOP
    out = Image.new("RGB", (W, rows * out_cell_h), BACKGROUND)
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell_w, r * cell_h
            cell = image.crop((x0, y0, x0 + cell_w, y0 + cell_h))
            cell_clean = cell.crop((0, GRID_CROP_TOP, cell_w, cell_h))
            out.paste(cell_clean, (c * cell_w, r * out_cell_h))
    return out

def build_single_assets() -> None:
    for figure in FIGURES:
        image = Image.open(figure["source"]).convert("RGB")
        if figure.get("clean_grid"):
            image = clean_val_grid(image)
        image = fit_image(image, figure["max_width"])
        styled = framed_figure(image)
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

def export_to_testpics() -> None:
    TESTPICS_DIR.mkdir(parents=True, exist_ok=True)
    for figure in FIGURES:
        src = figure["target"]
        dst = TESTPICS_DIR / figure["journal_name"]
        img = Image.open(src)
        img.save(dst, format="PNG")
        print(f"testpics: {dst}")
    src_8c = OUTPUT_DIR / "figure8c_five_task_montage.png"
    dst_8c = TESTPICS_DIR / "Figure_8c.png"
    img = Image.open(src_8c)
    img.save(dst_8c, format="PNG")
    print(f"testpics: {dst_8c}")

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    build_single_assets()
    export_montage_png()
    export_to_testpics()

if __name__ == "__main__":
    main()
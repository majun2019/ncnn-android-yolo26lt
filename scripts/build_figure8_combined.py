from PIL import Image, ImageDraw, ImageFont
import pathlib, shutil

ROOT    = pathlib.Path(__file__).parent.parent
FIG_DIR = ROOT / 'runs' / 'paper_figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

SRC_A   = FIG_DIR / 'figure8a_train_results_styled.png'
SRC_VAL = (ROOT / 'runs' / 'detect' / 'runs' / 'train' /
           'yolo26n_safehat' / 'val_batch0_pred.jpg')
SRC_C   = FIG_DIR / 'figure8c_five_task_montage.png'

GRID_CROP_TOP = 50
ROW_HEIGHT    = 440
ROW_GAP       = 18
FONT_SIZE     = 22
BG            = (255, 255, 255)
FG            = (30,  30,  30)

_font_paths = [
    r'C:\Windows\Fonts\arialbd.ttf',
    r'C:\Windows\Fonts\arial.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]
font = None
for _fp in _font_paths:
    try:
        font = ImageFont.truetype(_fp, FONT_SIZE)
        break
    except OSError:
        pass
if font is None:
    font = ImageFont.load_default()

def scale_to_height(img: Image.Image, h: int) -> Image.Image:
    w = round(img.width * h / img.height)
    return img.resize((w, h), Image.LANCZOS)

img_a = Image.open(SRC_A).convert('RGB')
row1  = scale_to_height(img_a, ROW_HEIGHT)
print(f'row1 (8a):  src={img_a.size}  →  {row1.size}')

val    = Image.open(SRC_VAL).convert('RGB')
cw, ch = val.width // 3, val.height // 3

cells = []
for r in range(3):
    for c in range(3):
        cell = val.crop((
            c * cw,
            r * ch + GRID_CROP_TOP,
            (c + 1) * cw,
            (r + 1) * ch,
        ))
        cells.append(cell)
        if len(cells) == 8:
            break
    if len(cells) == 8:
        break

cell_w, cell_h = cells[0].size
grid = Image.new('RGB', (4 * cell_w, 2 * cell_h), BG)
for i, cell in enumerate(cells):
    grid.paste(cell, ((i % 4) * cell_w, (i // 4) * cell_h))

row2 = scale_to_height(grid, ROW_HEIGHT)
print(f'row2 (8b):  grid={grid.size}  →  {row2.size}')

img_c = Image.open(SRC_C).convert('RGB')
row3  = scale_to_height(img_c, ROW_HEIGHT)
print(f'row3 (8c):  src={img_c.size}  →  {row3.size}')

_tmp  = ImageDraw.Draw(Image.new('RGB', (1, 1)))
_bb   = _tmp.textbbox((0, 0), '(a)', font=font)
LABEL_H = (_bb[3] - _bb[1]) + 8

canvas_w = max(row1.width, row2.width, row3.width)
slot_h   = ROW_HEIGHT + LABEL_H
canvas_h = 3 * slot_h + 2 * ROW_GAP

canvas = Image.new('RGB', (canvas_w, canvas_h), BG)
draw   = ImageDraw.Draw(canvas)

def place(img: Image.Image, label: str, row_idx: int) -> None:
    y0 = row_idx * (slot_h + ROW_GAP)
    x0 = (canvas_w - img.width) // 2
    canvas.paste(img, (x0, y0))
    bb = draw.textbbox((0, 0), label, font=font)
    lw = bb[2] - bb[0]
    draw.text(
        ((canvas_w - lw) // 2, y0 + ROW_HEIGHT + 4),
        label, fill=FG, font=font,
    )

place(row1, '(a)', 0)
place(row2, '(b)', 1)
place(row3, '(c)', 2)

out = FIG_DIR / 'figure8_combined.png'
canvas.save(out, dpi=(300, 300))
print(f'\nSaved  {out.resolve()}')
print(f'Size   {canvas_w} × {canvas_h} px  (row_h={ROW_HEIGHT})')

tc = ROOT / 'testpics' / 'Figure_8.png'
tc.parent.mkdir(exist_ok=True)
shutil.copy(out, tc)
print(f'Copied {tc}')

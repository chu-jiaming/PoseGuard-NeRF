#!/usr/bin/env python
"""Build the 1280x640 GitHub social-preview card (Settings -> Social preview).

Reads docs/assets/perturbation_grid.png only, so it runs without the server delivery.
Uses Helvetica Neue when available (macOS) and falls back to matplotlib's DejaVu Sans.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INK, MUTED, FAINT, BLUE, GRAY, PANEL = '#0b0b0b', '#52514e', '#8a8984', '#2a78d6', '#9a9994', '#f6f5f2'
W, H, S = 1280, 640, 2  # drawn at 2x and downsampled for clean edges
HELVETICA = Path('/System/Library/Fonts/HelveticaNeue.ttc')
# Translation 5% row of perturbation_grid.png: fixed cameras, naive + warm-up.
FIXED_BOX, WARMUP_BOX = (262, 58, 509, 245), (1005, 58, 1252, 245)


def font_loader():
    from PIL import ImageFont
    if HELVETICA.exists():
        faces = {'regular': 0, 'bold': 1, 'medium': 10}
        return lambda size, weight='regular': ImageFont.truetype(str(HELVETICA), size*S, index=faces[weight])
    import matplotlib
    fonts = Path(matplotlib.get_data_path())/'fonts/ttf'
    files = {'regular': 'DejaVuSans.ttf', 'bold': 'DejaVuSans-Bold.ttf', 'medium': 'DejaVuSans.ttf'}
    return lambda size, weight='regular': ImageFont.truetype(str(fonts/files[weight]), round(size*.92)*S)


def social_preview(grid_path, output):
    from PIL import Image, ImageDraw
    font = font_loader()
    grid = Image.open(grid_path).convert('RGB')
    card = Image.new('RGB', (W*S, H*S), 'white')
    draw = ImageDraw.Draw(card)

    def text(x, y, value, size, fill, weight='regular'):
        draw.text((x*S, y*S), value, font=font(size, weight), fill=fill)

    x = 72
    text(x, 112, 'NeRF  ·  NOISY CAMERA POSES', 19, BLUE, 'bold')
    text(x, 146, 'PoseGuard-NeRF', 64, INK, 'bold')
    for i, line in enumerate(('When does joint camera-pose', 'optimization help NeRF, and when',
                              'does it just move the world?')):
        text(x, 244 + i*40, line, 30, MUTED)
    # Headline numbers are the full-test-set PSNR from the README, not the single view shown.
    text(x, 400, '14.1', 46, GRAY, 'bold')
    ax, ay = x + draw.textlength('14.1', font=font(46, 'bold'))/S + 16, 424
    draw.line([ax*S, ay*S, (ax+34)*S, ay*S], fill=BLUE, width=5*S)  # Helvetica Neue has no → glyph
    draw.polygon([((ax+44)*S, ay*S), ((ax+30)*S, (ay-10)*S), ((ax+30)*S, (ay+10)*S)], fill=BLUE)
    text(ax + 60, 400, '22.7 dB', 46, INK, 'bold')
    text(x, 458, 'test PSNR at 5% translation noise,', 21, MUTED)
    text(x, 486, 'with a camera learning-rate warm-up', 21, MUTED)
    text(x, 556, '45 runs  ·  9 noise levels × 5 methods  ·  Sim(3) drift diagnostic', 18, FAINT, 'medium')

    panel, size = (684, 72, 1208, 568), 224
    draw.rounded_rectangle([v*S for v in panel], radius=20*S, fill=PANEL)
    rows = [(FIXED_BOX, GRAY, ('Fixed cameras', 'blurry'), 88),
            (WARMUP_BOX, BLUE, ('Joint pose opt.', '+ LR warm-up'), 328)]
    for box, color, (line1, line2), top in rows:
        render = grid.crop(box)
        render = render.resize((round(size*S*render.width/render.height), size*S), Image.LANCZOS)
        left = panel[0] + 16
        draw.rounded_rectangle([left*S, top*S, (panel[2]-16)*S, (top+size)*S], radius=14*S, fill='white')
        card.paste(render, ((left+8)*S, top*S))
        tx = left + 8 + render.width//S + 4
        draw.rectangle([tx*S, (top+84)*S, (tx+4)*S, (top+144)*S], fill=color)
        text(tx + 18, top + 86, line1, 21, INK, 'bold')
        highlight = color == BLUE
        text(tx + 18, top + 114, line2, 21, INK if highlight else MUTED, 'bold' if highlight else 'regular')

    card.resize((W, H), Image.LANCZOS).save(output, optimize=True)
    print('%-28s %7.0f KB' % (output.name, output.stat().st_size/1024))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=Path, default=ROOT/'docs/assets/perturbation_grid.png')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/assets/social_preview.png')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    social_preview(args.grid, args.output)


if __name__ == '__main__':
    main()

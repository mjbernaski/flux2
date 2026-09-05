#!/usr/bin/env python3
"""Crop and resample a generated image to an exact display resolution.

The generator's `widescreen` preset is 21:9-ish (3136x1344 at size=2mp), which
is close to but not identical to a 3440x1440 ultrawide. Scaling straight to the
panel would stretch it, and padding would letterbox it, so this centre-crops to
the display's exact aspect ratio first and only then resamples — the result is
pixel-exact for the screen with no distortion and no bars.

    python make_wallpaper.py web-generated/flux2_*.png -o wallpapers/ [-r 3440x1440]
"""
import argparse
import os
import sys

from PIL import Image


def fit(img, target_w, target_h):
    """Centre-crop `img` to the target aspect ratio, then resample to size."""
    target_aspect = target_w / target_h
    w, h = img.size
    if w / h > target_aspect:          # too wide: trim the sides
        new_w = round(h * target_aspect)
        left = (w - new_w) // 2
        box = (left, 0, left + new_w, h)
    else:                              # too tall: trim top and bottom
        new_h = round(w / target_aspect)
        top = (h - new_h) // 2
        box = (0, top, w, top + new_h)
    cropped = img.crop(box)
    if cropped.size == (target_w, target_h):
        return cropped, box
    return cropped.resize((target_w, target_h), Image.LANCZOS), box


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('images', nargs='+', help='source image(s)')
    ap.add_argument('-r', '--resolution', default='3440x1440',
                    help='target WIDTHxHEIGHT (default: 3440x1440)')
    ap.add_argument('-o', '--out-dir', default='wallpapers',
                    help='output directory (default: wallpapers/)')
    ap.add_argument('-q', '--quality', type=int, default=95,
                    help='JPEG quality when the output is .jpg (default: 95)')
    ap.add_argument('--jpeg', action='store_true',
                    help='write .jpg instead of .png')
    args = ap.parse_args()

    try:
        target_w, target_h = (int(v) for v in args.resolution.lower().split('x'))
    except ValueError:
        sys.exit(f'bad --resolution {args.resolution!r}; expected e.g. 3440x1440')

    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.images:
        with Image.open(path) as src:
            img = src.convert('RGB')
            out, box = fit(img, target_w, target_h)
        stem = os.path.splitext(os.path.basename(path))[0]
        ext = '.jpg' if args.jpeg else '.png'
        dest = os.path.join(args.out_dir, f'{stem}_{target_w}x{target_h}{ext}')
        if args.jpeg:
            out.save(dest, quality=args.quality, subsampling=0)
        else:
            out.save(dest)
        print(f'{path}  {img.size[0]}x{img.size[1]}'
              f'  -> crop {box[2]-box[0]}x{box[3]-box[1]}'
              f'  -> {target_w}x{target_h}  {dest}')


if __name__ == '__main__':
    main()

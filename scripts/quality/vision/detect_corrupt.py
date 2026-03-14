"""
Detect corrupt, truncated, or unreadable images.

Usage:
    uv run scripts/quality/vision/detect_corrupt.py --input data/images/
    uv run scripts/quality/vision/detect_corrupt.py --input data/images/ \
        --save-list data/corrupt_files.txt --output data/corrupt_report.json

Requirements: uv add Pillow
"""

import argparse
import json
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}


def main():
    parser = argparse.ArgumentParser(description="Detect corrupt/unreadable images")
    parser.add_argument("--input", required=True)
    parser.add_argument("--ext", default=",".join(IMAGE_EXTS))
    parser.add_argument("--check-size", action="store_true",
                        help="Also flag images with 0 bytes or suspiciously small file size")
    parser.add_argument("--min-size-bytes", type=int, default=100,
                        help="Minimum file size in bytes (default: 100)")
    parser.add_argument("--output", default="")
    parser.add_argument("--save-list", default="", help="Save list of corrupt file paths")
    args = parser.parse_args()

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        print("Pillow required: uv add Pillow", file=sys.stderr); sys.exit(1)

    root = Path(args.input)
    exts = {e.strip().lower() for e in args.ext.split(",")}
    files = sorted(f for f in root.rglob("*") if f.suffix.lower() in exts)

    print(f"Checking {len(files)} images...", file=sys.stderr)

    corrupt = []
    for i, f in enumerate(files):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(files)} checked, {len(corrupt)} corrupt so far...", file=sys.stderr)

        reason = None
        try:
            size = f.stat().st_size
            if args.check_size and size < args.min_size_bytes:
                reason = f"too_small ({size} bytes)"
            else:
                with Image.open(f) as img:
                    img.verify()  # catches truncated files
        except UnidentifiedImageError:
            reason = "unidentified_format"
        except Exception as e:
            reason = str(e)[:100]

        if reason:
            corrupt.append({"path": str(f), "reason": reason, "size_bytes": f.stat().st_size})

    report = {
        "total_files": len(files),
        "corrupt_files": len(corrupt),
        "corrupt_pct": round(len(corrupt) / len(files) * 100, 2) if files else 0,
        "severity": "critical" if len(corrupt)/max(len(files),1) > 0.1 else "high" if len(corrupt)/max(len(files),1) > 0.02 else "low",
        "corrupt": corrupt[:200],
    }

    out = json.dumps(report, ensure_ascii=False, indent=2)
    print(out)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out)

    if args.save_list and corrupt:
        Path(args.save_list).write_text("\n".join(c["path"] for c in corrupt))
        print(f"Corrupt file list saved to {args.save_list}", file=sys.stderr)


if __name__ == "__main__":
    main()

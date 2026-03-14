"""
Detect duplicate images using MD5 (exact) or perceptual hash (near-duplicates).

Usage:
    # Exact duplicates by MD5
    uv run scripts/quality/vision/hash_duplicates.py --input data/images/ --method md5

    # Near-duplicates by perceptual hash
    uv run scripts/quality/vision/hash_duplicates.py --input data/images/ \
        --method phash --threshold 10

    # Save list of duplicates to remove
    uv run scripts/quality/vision/hash_duplicates.py --input data/images/ \
        --method dhash --output data/duplicate_images.json

Methods:
    md5     — exact byte-level duplicates (fast, no false positives)
    phash   — perceptual hash (catches resized/recompressed duplicates)
    dhash   — difference hash (good for near-duplicates, robust to brightness)

Requirements:
    uv add Pillow imagehash
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from collections import defaultdict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}


def md5_hash(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def perceptual_hash(path: Path, method: str, size: int = 8):
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        print("Pillow + imagehash required: uv add Pillow imagehash", file=sys.stderr)
        sys.exit(1)
    with Image.open(path) as img:
        img = img.convert("RGB")
        if method == "phash":
            return imagehash.phash(img, hash_size=size)
        elif method == "dhash":
            return imagehash.dhash(img, hash_size=size)
        elif method == "ahash":
            return imagehash.average_hash(img, hash_size=size)


def main():
    parser = argparse.ArgumentParser(description="Detect duplicate images by hash")
    parser.add_argument("--input", required=True, help="Path to image folder")
    parser.add_argument("--method", choices=["md5", "phash", "dhash", "ahash"], default="md5")
    parser.add_argument("--threshold", type=int, default=8,
                        help="Max hash distance for near-duplicates (perceptual methods, default: 8)")
    parser.add_argument("--hash-size", type=int, default=8,
                        help="Perceptual hash size (default: 8, higher = more sensitive)")
    parser.add_argument("--ext", default=",".join(IMAGE_EXTS),
                        help="Comma-separated image extensions to scan")
    parser.add_argument("--output", default="", help="Save duplicate report JSON")
    parser.add_argument("--save-remove-list", default="",
                        help="Save list of files to remove (keep first, remove rest)")
    args = parser.parse_args()

    root = Path(args.input)
    exts = {e.strip().lower() for e in args.ext.split(",")}
    files = sorted(f for f in root.rglob("*") if f.suffix.lower() in exts)

    print(f"Scanning {len(files)} images with method={args.method}...", file=sys.stderr)

    if args.method == "md5":
        hash_map = defaultdict(list)
        for i, f in enumerate(files):
            if i % 500 == 0:
                print(f"  {i}/{len(files)}...", file=sys.stderr)
            try:
                h = md5_hash(f)
                hash_map[h].append(str(f))
            except Exception as e:
                print(f"  Warning: {f}: {e}", file=sys.stderr)

        duplicate_groups = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
        to_remove = [p for paths in duplicate_groups.values() for p in paths[1:]]

        report = {
            "method": "md5",
            "total_files": len(files),
            "unique_files": len(hash_map) - len(duplicate_groups),
            "duplicate_groups": len(duplicate_groups),
            "duplicate_files": sum(len(v) - 1 for v in duplicate_groups.values()),
            "groups": {h: paths for h, paths in list(duplicate_groups.items())[:50]},
        }

    else:
        # Perceptual hash — compare all pairs within threshold
        hashes = []
        failed = []
        for i, f in enumerate(files):
            if i % 200 == 0:
                print(f"  Hashing {i}/{len(files)}...", file=sys.stderr)
            try:
                h = perceptual_hash(f, args.method, args.hash_size)
                hashes.append((str(f), h))
            except Exception as e:
                failed.append(str(f))

        print(f"  Finding pairs within threshold={args.threshold}...", file=sys.stderr)
        groups = []
        used = set()
        for i, (path_i, hash_i) in enumerate(hashes):
            if path_i in used:
                continue
            group = [path_i]
            for j, (path_j, hash_j) in enumerate(hashes):
                if i == j or path_j in used:
                    continue
                if (hash_i - hash_j) <= args.threshold:
                    group.append(path_j)
                    used.add(path_j)
            if len(group) > 1:
                used.add(path_i)
                groups.append(group)

        to_remove = [p for g in groups for p in g[1:]]
        report = {
            "method": args.method,
            "threshold": args.threshold,
            "total_files": len(files),
            "failed_to_hash": len(failed),
            "duplicate_groups": len(groups),
            "duplicate_files": len(to_remove),
            "groups": groups[:50],
        }

    print(f"\nFound {report['duplicate_groups']} duplicate groups, {report.get('duplicate_files', 0)} files to remove")

    out = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out)
        print(f"Report saved to {args.output}", file=sys.stderr)
    else:
        print(out)

    if args.save_remove_list and to_remove:
        Path(args.save_remove_list).write_text("\n".join(to_remove))
        print(f"Remove list saved to {args.save_remove_list} ({len(to_remove)} files)", file=sys.stderr)


if __name__ == "__main__":
    main()

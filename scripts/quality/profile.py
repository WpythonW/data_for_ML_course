"""
Quick dataset profile — shape, types, nulls, sample rows, file counts.
Works for tabular files (CSV/Parquet/JSON) and directories (image/text/audio folders).

Usage:
    uv run scripts/quality/profile.py --input data/train.csv
    uv run scripts/quality/profile.py --input data/images/ --modality image
    uv run scripts/quality/profile.py --input data/corpus.jsonl --modality text --text-col text
    uv run scripts/quality/profile.py --input data/train.csv --output data/profile.json
"""

import argparse
import json
import os
import sys
from pathlib import Path


def profile_tabular(path: Path, sep: str, text_col: str | None) -> dict:
    try:
        import pandas as pd
    except ImportError:
        print("pandas required: uv add pandas", file=sys.stderr); sys.exit(1)

    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, sep=sep, low_memory=False)
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".json",):
        df = pd.read_json(path)
    elif ext in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        print(f"Unsupported tabular format: {ext}", file=sys.stderr); sys.exit(1)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)

    profile = {
        "modality": "tabular",
        "path": str(path),
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": {
            col: {"count": int(null_counts[col]), "pct": float(null_pct[col])}
            for col in df.columns if null_counts[col] > 0
        },
        "total_missing_pct": round(df.isnull().sum().sum() / df.size * 100, 2),
        "duplicates_exact": int(df.duplicated().sum()),
        "numeric_summary": {},
        "sample": df.head(3).to_dict(orient="records"),
    }

    if numeric_cols:
        desc = df[numeric_cols].describe().to_dict()
        profile["numeric_summary"] = {
            col: {k: round(v, 4) for k, v in stats.items()}
            for col, stats in desc.items()
        }

    if text_col and text_col in df.columns:
        lengths = df[text_col].dropna().str.len()
        profile["text_col_stats"] = {
            "col": text_col,
            "empty": int((df[text_col] == "").sum()),
            "null": int(df[text_col].isnull().sum()),
            "len_mean": round(float(lengths.mean()), 1),
            "len_min": int(lengths.min()),
            "len_max": int(lengths.max()),
            "len_p50": int(lengths.quantile(0.5)),
            "len_p95": int(lengths.quantile(0.95)),
        }

    return profile


def profile_image(path: Path, extensions: list[str]) -> dict:
    exts = set(e.lower() for e in extensions)
    files = [f for f in path.rglob("*") if f.suffix.lower() in exts]
    sizes = [f.stat().st_size for f in files]

    # Try to get image dimensions for a sample
    dims = []
    try:
        from PIL import Image
        for f in files[:200]:
            try:
                with Image.open(f) as img:
                    dims.append(img.size)  # (W, H)
            except Exception:
                pass
    except ImportError:
        pass

    # Class distribution from subfolder structure
    class_dist = {}
    for f in files:
        cls = f.parent.name
        class_dist[cls] = class_dist.get(cls, 0) + 1

    widths = [d[0] for d in dims]
    heights = [d[1] for d in dims]

    profile = {
        "modality": "image",
        "path": str(path),
        "total_files": len(files),
        "extensions": {e: sum(1 for f in files if f.suffix.lower() == e) for e in exts if any(f.suffix.lower() == e for f in files)},
        "size_bytes": {
            "total": sum(sizes),
            "mean": round(sum(sizes) / len(sizes), 0) if sizes else 0,
            "min": min(sizes) if sizes else 0,
            "max": max(sizes) if sizes else 0,
        },
        "dimensions_sample": {
            "count_sampled": len(dims),
            "width": {"min": min(widths), "max": max(widths), "mean": round(sum(widths)/len(widths), 0)} if widths else {},
            "height": {"min": min(heights), "max": max(heights), "mean": round(sum(heights)/len(heights), 0)} if heights else {},
            "unique_sizes": len(set(dims)),
        } if dims else {},
        "class_distribution": dict(sorted(class_dist.items(), key=lambda x: -x[1])),
        "num_classes": len(class_dist),
    }
    return profile


def profile_text(path: Path, text_col: str | None, sep: str) -> dict:
    if path.is_file():
        return profile_tabular(path, sep, text_col or "text")

    # Directory of .txt files
    files = list(path.rglob("*.txt"))
    sizes = [f.stat().st_size for f in files]
    lengths = [len(f.read_text(errors="ignore").split()) for f in files[:500]]

    return {
        "modality": "text",
        "path": str(path),
        "total_files": len(files),
        "size_bytes": {"total": sum(sizes), "mean": round(sum(sizes)/len(sizes), 0) if sizes else 0},
        "word_count_sample": {
            "count_sampled": len(lengths),
            "mean": round(sum(lengths)/len(lengths), 1) if lengths else 0,
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
    }


def profile_audio(path: Path, extensions: list[str]) -> dict:
    exts = set(e.lower() for e in extensions)
    files = [f for f in path.rglob("*") if f.suffix.lower() in exts]
    sizes = [f.stat().st_size for f in files]

    durations = []
    try:
        import soundfile as sf
        for f in files[:200]:
            try:
                info = sf.info(str(f))
                durations.append(info.duration)
            except Exception:
                pass
    except ImportError:
        pass

    class_dist = {}
    for f in files:
        cls = f.parent.name
        class_dist[cls] = class_dist.get(cls, 0) + 1

    return {
        "modality": "audio",
        "path": str(path),
        "total_files": len(files),
        "extensions": {e: sum(1 for f in files if f.suffix.lower() == e) for e in exts},
        "size_bytes": {"total": sum(sizes), "mean": round(sum(sizes)/len(sizes), 0) if sizes else 0},
        "duration_seconds": {
            "count_sampled": len(durations),
            "mean": round(sum(durations)/len(durations), 2) if durations else 0,
            "min": round(min(durations), 2) if durations else 0,
            "max": round(max(durations), 2) if durations else 0,
        } if durations else {},
        "class_distribution": dict(sorted(class_dist.items(), key=lambda x: -x[1])),
    }


def main():
    parser = argparse.ArgumentParser(description="Quick dataset profile")
    parser.add_argument("--input", required=True, help="Path to file or directory")
    parser.add_argument("--modality", choices=["tabular", "image", "text", "audio", "auto"],
                        default="auto", help="Data modality (default: auto-detect)")
    parser.add_argument("--output", default="", help="Save profile JSON to this path")
    parser.add_argument("--sep", default=",", help="CSV separator (default: ,)")
    parser.add_argument("--text-col", default="", help="Column name containing text (for tabular text datasets)")
    parser.add_argument("--image-ext", default=".jpg,.jpeg,.png,.bmp,.webp,.tiff",
                        help="Image extensions to scan (default: common formats)")
    parser.add_argument("--audio-ext", default=".wav,.mp3,.flac,.ogg,.m4a",
                        help="Audio extensions to scan")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Error: path not found: {path}", file=sys.stderr); sys.exit(1)

    # Auto-detect modality
    modality = args.modality
    if modality == "auto":
        if path.is_dir():
            files = list(path.rglob("*"))
            exts = {f.suffix.lower() for f in files if f.is_file()}
            img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
            audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
            if exts & img_exts:
                modality = "image"
            elif exts & audio_exts:
                modality = "audio"
            else:
                modality = "text"
        else:
            ext = path.suffix.lower()
            if ext in (".csv", ".parquet", ".pq", ".json", ".jsonl", ".xlsx", ".xls", ".ndjson"):
                modality = "tabular"
            elif ext in (".jpg", ".jpeg", ".png"):
                modality = "image"
            elif ext in (".wav", ".mp3", ".flac"):
                modality = "audio"
            else:
                modality = "text"

    print(f"Profiling [{modality}]: {path}", file=sys.stderr)

    img_exts = [e.strip() for e in args.image_ext.split(",")]
    audio_exts = [e.strip() for e in args.audio_ext.split(",")]
    text_col = args.text_col or None

    if modality == "tabular":
        profile = profile_tabular(path, args.sep, text_col)
    elif modality == "image":
        profile = profile_image(path, img_exts)
    elif modality == "text":
        profile = profile_text(path, text_col, args.sep)
    elif modality == "audio":
        profile = profile_audio(path, audio_exts)
    else:
        profile = {"error": f"Unknown modality: {modality}"}

    out = json.dumps(profile, ensure_ascii=False, indent=2)
    print(out)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out, encoding="utf-8")
        print(f"\nSaved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

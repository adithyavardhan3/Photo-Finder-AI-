"""Streaming recursive scanner for edited and camera RAW images."""
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".pef", ".srw"}
RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".pef", ".srw"}

def scan_folder(folder: str):
    """Yield supported image paths recursively without materializing the full list."""
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Folder does not exist or is unavailable: {root}")
    for path in root.rglob("*"):
        try:
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield str(path)
        except (OSError, PermissionError):
            # One inaccessible entry should not abort the rest of the scan.
            continue

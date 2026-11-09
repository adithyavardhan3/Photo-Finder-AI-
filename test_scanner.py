from pathlib import Path
from scanner import scan_folder

def test_scan_folder_finds_supported_images(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"fake")
    (tmp_path / "b.PNG").write_bytes(b"fake")
    (tmp_path / "c.txt").write_text("not image")
    assert {Path(p).name for p in scan_folder(str(tmp_path))} == {"a.jpg", "b.PNG"}

def test_scan_folder_includes_camera_raw_formats(tmp_path: Path):
    for name in ("raw.CR3", "raw.NEF", "raw.ARW", "edited.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"fake")
    found = {Path(p).name for p in scan_folder(str(tmp_path))}
    assert found == {"raw.CR3", "raw.NEF", "raw.ARW", "edited.jpg"}

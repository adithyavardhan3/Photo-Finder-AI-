"""Local webcam helpers. The UI owns the live VideoCapture lifecycle."""
from pathlib import Path
from datetime import datetime
import cv2


def open_camera(camera_index: int = 0):
    """Open a local camera. Caller must release it when preview stops."""
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0
    cap = cv2.VideoCapture(camera_index, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "Could not open webcam. Check Windows camera permissions and close other camera apps."
        )
    return cap


def save_snapshot(frame):
    """Save the explicitly captured frame locally and return its path."""
    if frame is None:
        raise RuntimeError("No camera frame is available to capture.")
    out_dir = Path.home() / "PhotoFinderAI" / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"customer_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    if not cv2.imwrite(str(out_path), frame):
        raise RuntimeError("Could not save webcam snapshot.")
    return out_path

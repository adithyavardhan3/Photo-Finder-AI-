"""Offline YuNet + SFace face embedding engine. Model weights are fetched once on first use."""
from pathlib import Path
import urllib.request
import cv2
import numpy as np
from scanner import RAW_EXTENSIONS

MODEL_DIR = Path.home() / "PhotoFinderAI" / "models"
MODELS = {
    "yunet.onnx": "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface.onnx": "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

def _model_path(name):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / name
    if path.exists() and path.stat().st_size > 100_000:
        return str(path)
    url = MODELS[name]
    tmp = path.with_suffix(path.suffix + ".download")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PhotoFinderAI/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response, open(tmp, "wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk: break
                f.write(chunk)
        if tmp.stat().st_size < 100_000:
            raise RuntimeError("Downloaded model file is unexpectedly small.")
        tmp.replace(path)
    except Exception as exc:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass
        raise RuntimeError(f"Could not obtain {name}. Internet is required once to download the model; then processing is local. Details: {exc}") from exc
    return str(path)

class FaceEngine:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(_model_path("yunet.onnx"), "", (320, 320), 0.75, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(_model_path("sface.onnx"), "")

    def embeddings_from_image(self, image):
        if image is None: return []
        h, w = image.shape[:2]
        # Limit huge wedding images for face detection speed; coordinates are scaled back.
        max_side = 1600
        scale = min(1.0, max_side / max(h, w))
        work = cv2.resize(image, (max(1, int(w*scale)), max(1, int(h*scale)))) if scale < 1 else image
        self.detector.setInputSize((work.shape[1], work.shape[0]))
        _, faces = self.detector.detect(work)
        if faces is None: return []
        found = []
        for face in faces:
            aligned = self.recognizer.alignCrop(work, face)
            feat = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(feat))
            if norm > 0:
                feat /= norm
                found.append(feat)
        return found

    def read_image(self, path):
        """Read regular images with OpenCV and RAW images with optional rawpy."""
        p = Path(path)
        if p.suffix.lower() in RAW_EXTENSIONS:
            try:
                import rawpy
            except ImportError as exc:
                raise ValueError("RAW file requires rawpy. Install it with: python -m pip install rawpy") from exc
            with rawpy.imread(str(p)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, half_size=True, output_bps=8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image (unsupported format or damaged file).")
        return image

    def embeddings_from_path(self, path):
        image = self.read_image(path)
        return self.embeddings_from_image(image)

    def query_embeddings(self, path):
        return self.embeddings_from_path(path)

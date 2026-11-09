"""Face detection using OpenCV's bundled Haar cascade (not identity recognition)."""
from pathlib import Path
import cv2

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def detect_faces_in_image(image_path: str) -> int:
    image = cv2.imread(str(Path(image_path)))
    if image is None:
        raise ValueError("OpenCV could not read this image.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError("OpenCV face detector could not be loaded.")
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return len(faces)

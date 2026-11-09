# Photo Finder AI v6 — Faster incremental local face search

Windows desktop app for local photo indexing and face similarity search. Uses **YuNet** for face detection and **SFace** for face embeddings, then stores embeddings in SQLite and compares them locally. Search runs in a background worker.

## Setup
1. Extract this ZIP to a new folder (keep v3/v4 as backup).
2. Open PowerShell in `photo_finder_ai_v2`.
3. Run:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless
pip install -r requirements.txt
python main.py
```
4. On first use, model files are downloaded from the OpenCV Zoo GitHub repository into `%USERPROFILE%\PhotoFinderAI\models`. This requires internet once. After the model files are present, face processing and search are local and no photo is uploaded.
5. Create/select an event, select the folder, confirm permission, then index. Existing photos must be re-indexed once to create SFace embeddings. Then capture a face and search.

## Faster incremental indexing
- Re-indexing checks file size and modification time; unchanged photos with saved embeddings are skipped.
- Database writes are committed in batches of up to 100 changed photos.
- First-time indexing still needs to decode every photo and run face detection/embedding extraction.
- If you change or replace a photo while preserving both file size and modification timestamp, manually re-index that photo after changing its timestamp or moving it into a fresh event.

## Performance and limitations
- Designed as a CPU-first local prototype; actual indexing time depends on CPU, image sizes, and number of faces. It is not benchmarked or guaranteed for every 50,000-photo collection.
- Embeddings are stored in SQLite; query search uses a vectorized NumPy matrix comparison. Very large face counts may need FAISS or another approximate-nearest-neighbor index.
- The app may detect multiple faces per image. Search ranks photos by their best face match.
- Similarity is not a probability and does not prove identity. Evaluate false matches on representative wedding photos before relying on results.
- The v5 schema adds a local face_embeddings table to the existing SQLite database. Existing metadata is retained, but old photos need re-indexing.

## Model licensing notice
The code retrieves YuNet and SFace model files from OpenCV Zoo at first run. Before commercial redistribution, independently verify the license and provenance of the exact model weights and all dependencies. The application does not include the model weight files in this ZIP because the build environment could not download and verify them. Do not sell or redistribute pretrained weights until their rights are confirmed.

## Privacy
Use only photos you have permission to process. Face embeddings are sensitive biometric data; restrict access to the computer and database and delete data when no longer needed.

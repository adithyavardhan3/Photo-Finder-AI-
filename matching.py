"""Embedding-based local face search with YuNet detection and SFace recognition."""
from pathlib import Path
import cv2
import numpy as np
from face_engine import FaceEngine

def search_all(db_connection, query_image, limit=100):
    engine = FaceEngine()
    query_faces = engine.query_embeddings(query_image)
    if not query_faces:
        raise ValueError("No clear face was detected. Try a front-facing image with good lighting.")
    rows = db_connection.execute("""
        SELECT f.path, e.name, f.embedding FROM face_embeddings f
        JOIN events e ON e.event_id=f.event_id
        JOIN photos p ON p.event_id=f.event_id AND p.path=f.path
        WHERE p.status='indexed' ORDER BY f.embedding_id
    """).fetchall()
    if not rows:
        raise ValueError("No face embeddings found. Re-index your photo folders with v5 first.")
    matrix=[]; metadata=[]
    for path, event_name, blob in rows:
        vector=np.frombuffer(blob,dtype=np.float32)
        if vector.size:
            matrix.append(vector); metadata.append((path,event_name))
    if not matrix:
        raise ValueError("No valid face embeddings are stored. Re-index the photos.")
    matrix=np.vstack(matrix)
    # Similarity is cosine similarity of normalized SFace embeddings.
    best={}
    for q in query_faces:
        scores=matrix @ q
        # Only keep the best face match per photo.
        for idx in np.argpartition(scores, -min(200, len(scores)))[-min(200, len(scores)):]:
            path,event=metadata[int(idx)]; score=float(scores[int(idx)])
            if path not in best or score>best[path][1]: best[path]=(event,score)
    return [(path,event,score) for path,(event,score) in sorted(best.items(),key=lambda item:item[1][1],reverse=True)[:limit]]

"""SQLite storage for local event/photo metadata and face-detection counts."""
from pathlib import Path
from datetime import datetime, timezone
import sqlite3

class PhotoDatabase:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS events (
          event_id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL UNIQUE,
          created_at TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'active'
        );
        CREATE TABLE IF NOT EXISTS photos (
          photo_id INTEGER PRIMARY KEY AUTOINCREMENT,
          event_id INTEGER NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
          path TEXT NOT NULL, file_size INTEGER NOT NULL, modified_ns INTEGER NOT NULL,
          face_count INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'indexed',
          indexed_at TEXT NOT NULL, error TEXT,
          year TEXT, customer TEXT, folder_path TEXT, file_type TEXT, content_hash TEXT,
          UNIQUE(event_id, path)
        );
        CREATE INDEX IF NOT EXISTS idx_photos_event ON photos(event_id);
        CREATE TABLE IF NOT EXISTS face_embeddings (
          embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
          event_id INTEGER NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
          path TEXT NOT NULL, face_index INTEGER NOT NULL,
          embedding BLOB NOT NULL,
          UNIQUE(event_id, path, face_index)
        );
        CREATE INDEX IF NOT EXISTS idx_embeddings_event ON face_embeddings(event_id);
        """)
        # Migration for databases created by earlier prototype versions.
        columns = {r[1] for r in self.conn.execute("PRAGMA table_info(photos)")}
        if "face_count" not in columns:
            self.conn.execute("ALTER TABLE photos ADD COLUMN face_count INTEGER NOT NULL DEFAULT 0")
        if "error" not in columns:
            self.conn.execute("ALTER TABLE photos ADD COLUMN error TEXT")
        for col, decl in (("year", "TEXT"), ("customer", "TEXT"), ("folder_path", "TEXT"), ("file_type", "TEXT"), ("content_hash", "TEXT")):
            if col not in columns:
                self.conn.execute(f"ALTER TABLE photos ADD COLUMN {col} {decl}")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_year_customer ON photos(event_id, year, customer)")
        self.conn.commit()

    def create_event(self, name):
        name = name.strip()
        if not name:
            raise ValueError("Event name cannot be empty.")
        try:
            cur = self.conn.execute(
                "INSERT INTO events(name, created_at) VALUES (?, ?)",
                (name, datetime.now(timezone.utc).isoformat()))
            self.conn.commit()
            return int(cur.lastrowid)
        except sqlite3.IntegrityError:
            raise ValueError("An event with that name already exists.")

    def list_events(self):
        return self.conn.execute(
            "SELECT event_id, name FROM events WHERE status='active' ORDER BY name").fetchall()

    def upsert_photo(self, event_id, path, face_count=0, commit=True):
        p = Path(path).resolve()
        stat = p.stat()
        parts = p.parts
        year = next((part for part in parts if len(part) == 4 and part.isdigit() and part.startswith(("19", "20"))), None)
        customer = p.parent.name if year else None
        self.conn.execute("""
          INSERT INTO photos(event_id,path,file_size,modified_ns,face_count,indexed_at,error,year,customer,folder_path,file_type)
          VALUES(?,?,?,?,?,?,NULL,?,?,?,?)
          ON CONFLICT(event_id,path) DO UPDATE SET
            file_size=excluded.file_size, modified_ns=excluded.modified_ns,
            face_count=excluded.face_count, indexed_at=excluded.indexed_at,
            status='indexed', error=NULL, year=excluded.year, customer=excluded.customer,
            folder_path=excluded.folder_path, file_type=excluded.file_type
        """, (event_id, str(p), stat.st_size, stat.st_mtime_ns, int(face_count),
              datetime.now(timezone.utc).isoformat(), year, customer, str(p.parent), p.suffix.lower()))
        if commit:
            self.conn.commit()

    def record_error(self, event_id, path, message):
        p = Path(path)
        try:
            stat = p.stat()
            size, modified = stat.st_size, stat.st_mtime_ns
        except OSError:
            size, modified = 0, 0
        self.conn.execute("""
          INSERT INTO photos(event_id,path,file_size,modified_ns,face_count,status,indexed_at,error)
          VALUES(?,?,?,?,0,'error',?,?)
          ON CONFLICT(event_id,path) DO UPDATE SET status='error',error=excluded.error
        """, (event_id, str(p.resolve()), size, modified, datetime.now(timezone.utc).isoformat(), message))
        self.conn.commit()

    def list_photos(self, event_id):
        if event_id is None:
            return []
        rows = self.conn.execute("""
          SELECT photos.path, events.name, photos.face_count
          FROM photos JOIN events ON events.event_id=photos.event_id
          WHERE photos.event_id=? ORDER BY photos.path LIMIT 5000
        """, (event_id,)).fetchall()
        return [(Path(path).name, event, faces, path) for path, event, faces in rows]

    def save_face_embeddings(self, event_id, path, embeddings, commit=True):
        resolved = str(Path(path).resolve())
        self.conn.execute("DELETE FROM face_embeddings WHERE event_id=? AND path=?", (event_id, resolved))
        for i, vector in enumerate(embeddings):
            self.conn.execute("INSERT INTO face_embeddings(event_id,path,face_index,embedding) VALUES(?,?,?,?)",
                              (event_id, resolved, i, vector.astype("float32").tobytes()))
        if commit:
            self.conn.commit()

    def close(self):
        self.conn.close()

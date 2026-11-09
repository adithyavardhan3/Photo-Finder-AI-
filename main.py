"""Photo Finder AI v3: local photo indexing and experimental LBPH face search.

Search runs locally across all indexed events. LBPH is a baseline matcher, not a
modern deep embedding model; results may be inaccurate and require evaluation.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import cv2

from PySide6.QtCore import QObject, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QProgressBar, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QComboBox, QDialog, QHeaderView
)

from database import PhotoDatabase
from scanner import scan_folder
from face_detection import detect_faces_in_image
from webcam import open_camera, save_snapshot
from matching import search_all
from face_engine import FaceEngine


class IndexWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(int, int, int, str)
    failed = Signal(str)

    def __init__(self, folder: str, event_id: int, db_path: str):
        super().__init__()
        self.folder = folder
        self.event_id = event_id
        self.db_path = db_path
        self._cancelled = False

    @Slot()
    def run(self):
        db = None
        try:
            db = PhotoDatabase(self.db_path)
            # Avoid loading image paths more than once; unchanged indexed files are reused.
            # Count first for determinate progress; processing itself streams paths.
            paths = list(scan_folder(self.folder))
            engine = FaceEngine()
            total = max(1, len(paths))
            processed = 0
            skipped = 0
            face_count = 0
            errors = 0
            pending_writes = 0
            for i, path in enumerate(paths, start=1):
                if self._cancelled:
                    break
                try:
                    stat = os.stat(path)
                    resolved = str(Path(path).resolve())
                    existing = db.conn.execute(
                        "SELECT file_size, modified_ns, status FROM photos WHERE event_id=? AND path=?",
                        (self.event_id, resolved)).fetchone()
                    # Indexed files with zero faces are complete too; do not decode them on every scan.
                    if (existing and existing[0] == stat.st_size and existing[1] == stat.st_mtime_ns
                            and existing[2] == 'indexed'):
                        skipped += 1
                    else:
                        embeddings = engine.embeddings_from_path(path)
                        n_faces = len(embeddings)
                        db.upsert_photo(self.event_id, path, face_count=n_faces, commit=False)
                        db.save_face_embeddings(self.event_id, path, embeddings, commit=False)
                        face_count += n_faces
                        processed += 1
                        pending_writes += 1
                        if pending_writes >= 100:
                            db.conn.commit()
                            pending_writes = 0
                except Exception as exc:
                    errors += 1
                    db.record_error(self.event_id, path, str(exc))
                if i == 1 or i % 5 == 0 or i == total:
                    self.progress.emit(i, total, path)
            if pending_writes:
                db.conn.commit()
            message = ("Indexing cancelled. Completed files are saved; rerun to resume." if self._cancelled else
                       f"Scan complete: {processed:,} processed, {skipped:,} unchanged skipped, {errors:,} failed. RAW decoding uses rawpy when installed.")
            self.finished.emit(processed, face_count, errors, message)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if db:
                db.close()

    def cancel(self):
        self._cancelled = True


class SearchWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, db_path: str, query_image: str):
        super().__init__()
        self.db_path = db_path
        self.query_image = query_image

    @Slot()
    def run(self):
        db = None
        try:
            # Use a separate SQLite connection inside this worker thread.
            db = PhotoDatabase(self.db_path)
            results = search_all(db.conn, self.query_image, limit=100)
            self.finished.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if db is not None:
                db.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo Finder AI v6 — Faster Incremental Face Search")
        self.resize(1280, 860)
        app_data = Path.home() / "PhotoFinderAI"
        app_data.mkdir(parents=True, exist_ok=True)
        self.db_path = str(app_data / "photo_finder_v2.sqlite3")
        self.db = PhotoDatabase(self.db_path)
        self.selected_folder = ""
        self.camera = None
        self.current_frame = None
        self.captured_snapshot_path = None
        self.camera_timer = QTimer(self)
        self.camera_timer.setInterval(33)
        self.camera_timer.timeout.connect(self.update_camera_frame)

        self.event_name = QLineEdit()
        self.event_name.setPlaceholderText("e.g. Wedding - 2026-10-19")
        self.event_combo = QComboBox()
        self.event_combo.currentIndexChanged.connect(self.refresh_photos)
        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        self.folder_edit.setPlaceholderText("Choose a local photo folder")
        self.consent_check = QCheckBox(
            "I confirm I have permission to process photos in this event."
        )

        self.choose_folder_btn = QPushButton("Choose folder")
        self.choose_folder_btn.clicked.connect(self.choose_folder)
        self.create_event_btn = QPushButton("Create event")
        self.create_event_btn.clicked.connect(self.create_event)
        self.index_btn = QPushButton("Index photos + detect faces")
        self.index_btn.clicked.connect(self.start_indexing)
        self.scan_new_btn = QPushButton("Scan for New Photos")
        self.scan_new_btn.clicked.connect(self.start_indexing)
        self.index_btn.setEnabled(False)
        self.cancel_btn = QPushButton("Cancel indexing")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_indexing)
        self.open_camera_btn = QPushButton("Open Camera")
        self.open_camera_btn.clicked.connect(self.start_camera_preview)
        self.capture_btn = QPushButton("Capture")
        self.capture_btn.clicked.connect(self.capture_current_frame)
        self.capture_btn.setEnabled(False)
        self.retake_btn = QPushButton("Retake")
        self.retake_btn.clicked.connect(self.retake_photo)
        self.retake_btn.setEnabled(False)
        self.search_btn = QPushButton("Search Photos")
        self.search_btn.clicked.connect(self.search_photos)
        self.search_btn.setEnabled(False)
        self.webcam_preview = QLabel("Camera is off. Click Open Camera to start live preview.")
        self.webcam_preview.setMinimumHeight(240)
        self.webcam_preview.setMinimumWidth(320)
        self.webcam_preview.setAlignment(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.AlignmentFlag.AlignCenter)
        self.webcam_preview.setScaledContents(False)
        self.webcam_preview.setStyleSheet("background:#090d13;border:1px solid #354258;border-radius:10px;padding:8px;color:#8e9bb0;")
        self.webcam_status = QLabel("Camera is off. No snapshot has been captured.")
        self.status = QLabel("Ready · Local experimental face search enabled; results may be inaccurate.")
        self.progress = QProgressBar()

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["File name", "Event", "Detected faces", "Full local path"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setShowGrid(False)

        layout = QVBoxLayout()
        hero = QLabel("<h1 style='color:#f4f6ff;margin:4px 0'>Photo Finder AI <span style='color:#a69aff'>v5</span></h1><span style='color:#aab6c8'>LOCAL PHOTO WORKSPACE  ·  OFFLINE PROCESSING</span>")
        layout.addWidget(hero)
        layout.addWidget(QLabel(
            "Local-first • YuNet face detection + SFace embeddings • Similarity search"
        ))
        layout.addWidget(QLabel("Create event"))
        layout.addWidget(self.event_name)
        layout.addWidget(self.create_event_btn)
        layout.addWidget(QLabel("Selected event"))
        layout.addWidget(self.event_combo)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(self.choose_folder_btn)
        layout.addLayout(folder_row)
        layout.addWidget(self.consent_check)
        button_row = QHBoxLayout()
        button_row.addWidget(self.index_btn)
        button_row.addWidget(self.scan_new_btn)
        button_row.addWidget(self.cancel_btn)
        layout.addLayout(button_row)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(QLabel("Indexed photos"))
        layout.addWidget(self.table)
        layout.addWidget(QLabel("<h3>Webcam test</h3>"))
        camera_row = QHBoxLayout()
        camera_row.addWidget(self.webcam_preview, 1)
        camera_col = QVBoxLayout()
        camera_col.addWidget(self.open_camera_btn)
        camera_col.addWidget(self.capture_btn)
        camera_col.addWidget(self.retake_btn)
        camera_col.addWidget(self.search_btn)
        camera_col.addWidget(self.webcam_status)
        camera_col.addStretch(1)
        camera_row.addLayout(camera_col, 1)
        layout.addLayout(camera_row)

        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)
        self.thread = None
        self.worker = None
        self.search_thread = None
        self.search_worker = None
        self.refresh_events()
        self.refresh_photos()

    def refresh_events(self):
        current = self.event_combo.currentData()
        self.event_combo.blockSignals(True)
        self.event_combo.clear()
        for event_id, name in self.db.list_events():
            self.event_combo.addItem(name, event_id)
        if current is not None:
            idx = self.event_combo.findData(current)
            if idx >= 0:
                self.event_combo.setCurrentIndex(idx)
        self.event_combo.blockSignals(False)
        self.update_enabled()

    def refresh_photos(self):
        if not hasattr(self, "table"):
            return
        event_id = self.event_combo.currentData()
        rows = self.db.list_photos(event_id) if event_id is not None else []
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()
        self.update_enabled()

    def update_enabled(self):
        if hasattr(self, "index_btn"):
            can_index = (self.event_combo.currentData() is not None and bool(self.folder_edit.text())
                         and self.consent_check.isChecked() and self.worker is None)
            self.index_btn.setEnabled(can_index)
            self.scan_new_btn.setEnabled(can_index)

    def create_event(self):
        name = self.event_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Event name required", "Enter an event name.")
            return
        try:
            self.db.create_event(name)
            self.event_name.clear()
            self.refresh_events()
            self.status.setText(f"Created event: {name}")
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot create event", str(exc))

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose local photo folder")
        if folder:
            self.folder_edit.setText(folder)
            self.update_enabled()

    def start_indexing(self):
        event_id = self.event_combo.currentData()
        if event_id is None or not self.folder_edit.text():
            return
        if not self.consent_check.isChecked():
            QMessageBox.warning(self, "Permission required", "Confirm permission before indexing.")
            return
        self.thread = QThread()
        self.worker = IndexWorker(self.folder_edit.text(), int(event_id), self.db_path)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_worker)
        self.index_btn.setEnabled(False)
        self.scan_new_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status.setText("Scanning local photos and detecting faces… unchanged files will be skipped.")
        self.progress.setValue(0)
        self.thread.start()

    def cancel_indexing(self):
        if self.worker:
            self.worker.cancel()
            self.status.setText("Cancellation requested…")

    @Slot(int, int, str)
    def on_progress(self, current, total, path):
        self.progress.setMaximum(total)
        self.progress.setValue(current)
        self.status.setText(f"Processed {current:,}/{total:,}: {path}")

    @Slot(int, int, int, str)
    def on_finished(self, processed, faces, errors, message):
        self.status.setText(
            f"{message} Photos processed: {processed:,}; face detections: {faces:,}; errors: {errors:,}."
        )
        self.refresh_photos()

    @Slot(str)
    def on_failed(self, message):
        QMessageBox.critical(self, "Indexing error", message)
        self.status.setText("Indexing failed.")

    def cleanup_worker(self):
        self.worker = None
        self.thread = None
        self.cancel_btn.setEnabled(False)
        self.update_enabled()
        self.refresh_photos()

    def start_camera_preview(self):
        if not self.consent_check.isChecked():
            QMessageBox.warning(
                self, "Permission required",
                "Ask the person for permission before opening the camera or processing their photo."
            )
            return
        if self.camera is not None:
            return
        try:
            self.camera = open_camera()
            self.current_frame = None
            self.camera_timer.start()
            self.open_camera_btn.setEnabled(False)
            self.capture_btn.setEnabled(True)
            self.retake_btn.setEnabled(False)
            self.search_btn.setEnabled(False)
            self.webcam_status.setText("Live preview is on. Click Capture to freeze this frame and turn the camera off.")
        except Exception as exc:
            self.stop_camera_preview()
            QMessageBox.warning(self, "Webcam unavailable", str(exc))

    def update_camera_frame(self):
        if self.camera is None:
            return
        ok, frame = self.camera.read()
        if not ok or frame is None:
            self.stop_camera_preview()
            self.webcam_status.setText("Could not read a camera frame. Reopen the camera and try again.")
            return
        self.current_frame = frame.copy()
        self.show_frame(frame)

    def show_frame(self, frame):
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        self.webcam_preview.setPixmap(pixmap.scaled(
            self.webcam_preview.size(),
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.AspectRatioMode.KeepAspectRatio,
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.TransformationMode.SmoothTransformation
        ))

    def stop_camera_preview(self):
        self.camera_timer.stop()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.open_camera_btn.setEnabled(True)
        self.capture_btn.setEnabled(False)

    def capture_current_frame(self):
        if self.camera is None or self.current_frame is None:
            QMessageBox.information(self, "No frame", "Wait for the live preview, then click Capture.")
            return
        frozen_frame = self.current_frame.copy()
        self.stop_camera_preview()  # release webcam immediately after capture
        try:
            new_path = save_snapshot(frozen_frame)
            old_path = self.captured_snapshot_path
            self.captured_snapshot_path = new_path
            if old_path and old_path != new_path:
                try:
                    old_path.unlink(missing_ok=True)
                except OSError:
                    pass
            self.current_frame = frozen_frame
            self.show_frame(frozen_frame)
            self.retake_btn.setEnabled(True)
            self.search_btn.setEnabled(self.event_combo.currentData() is not None)
            self.webcam_status.setText(
                f"Captured locally: {new_path.name}. Camera is off. Review the photo, Retake, or Search Photos."
            )
        except Exception as exc:
            self.webcam_status.setText("Capture failed. The camera is off.")
            QMessageBox.warning(self, "Capture failed", str(exc))

    def retake_photo(self):
        # The old snapshot remains only until a replacement is successfully captured.
        self.start_camera_preview()

    def search_photos(self):
        if not self.captured_snapshot_path or not self.captured_snapshot_path.exists():
            QMessageBox.information(self, "Capture required", "Open the camera and capture a photo first.")
            return
        if not self.consent_check.isChecked():
            QMessageBox.warning(self, "Permission required", "Confirm permission before searching indexed photos.")
            return
        if self.search_thread is not None:
            return

        # Keep the UI responsive: face detection and matching can take time on CPU.
        self.search_btn.setEnabled(False)
        self.index_btn.setEnabled(False)
        self.webcam_status.setText("Searching indexed photos in the background… The app may take time, but should remain responsive.")
        self.search_thread = QThread(self)
        self.search_worker = SearchWorker(self.db_path, str(self.captured_snapshot_path))
        self.search_worker.moveToThread(self.search_thread)
        self.search_thread.started.connect(self.search_worker.run)
        self.search_worker.finished.connect(self.show_search_results)
        self.search_worker.failed.connect(self.on_search_failed)
        self.search_worker.finished.connect(self.search_thread.quit)
        self.search_worker.failed.connect(self.search_thread.quit)
        self.search_thread.finished.connect(self.cleanup_search_worker)
        self.search_thread.start()

    @Slot(object)
    def show_search_results(self, results):
        self.webcam_status.setText(f"Search complete: {len(results)} candidate photos. Double-click a result to open it.")
        dialog = QDialog(self)
        dialog.setWindowTitle("Photo Finder AI — Search Results")
        dialog.resize(900, 560)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"Top {len(results)} results across all indexed events. Higher cosine similarity means closer SFace embeddings; this is NOT a probability or identity guarantee."))
        table = QTableWidget(len(results), 3)
        table.setHorizontalHeaderLabels(["Photo", "Event", "SFace cosine similarity (higher is closer)"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        for i, (path, event_name, distance) in enumerate(results):
            table.setItem(i, 0, QTableWidgetItem(Path(path).name))
            table.setItem(i, 1, QTableWidgetItem(event_name))
            table.setItem(i, 2, QTableWidgetItem(f"{distance:.4f}"))
            table.item(i, 0).setData(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.ItemDataRole.UserRole, path)
        table.cellDoubleClicked.connect(lambda r, c: os.startfile(table.item(r, 0).data(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.ItemDataRole.UserRole)) if table.item(r, 0) else None)
        layout.addWidget(table)
        note = QLabel("SFace embedding similarity is a ranking signal, not proof of identity. False matches can occur, especially for small, blurry, side-angle, or occluded faces. All processing is local.")
        note.setWordWrap(True)
        layout.addWidget(note)
        dialog.exec()

    @Slot(str)
    def on_search_failed(self, message):
        self.webcam_status.setText("Search failed. You can try again.")
        QMessageBox.warning(self, "Search could not complete", message)

    def cleanup_search_worker(self):
        self.search_worker = None
        self.search_thread = None
        self.search_btn.setEnabled(self.captured_snapshot_path is not None and self.captured_snapshot_path.exists())
        self.update_enabled()

    def closeEvent(self, event):
        if self.search_thread is not None:
            QMessageBox.information(self, "Search in progress", "Wait for the photo search to finish before closing the app.")
            event.ignore()
            return
        if self.worker:
            QMessageBox.warning(self, "Indexing in progress", "Cancel indexing and wait before closing.")
            event.ignore()
            return
        self.stop_camera_preview()
        self.db.close()
        event.accept()


def main():
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QWidget {
            background-color: #11151d;
            color: #e8edf5;
            font-family: 'Segoe UI';
            font-size: 10pt;
        }
        QMainWindow { background-color: #0b0f16; }
        QLabel { color: #dce4f0; }
        QLineEdit, QComboBox {
            background: #1b2230;
            border: 1px solid #303b4d;
            border-radius: 8px;
            padding: 9px 11px;
            selection-background-color: #7867f5;
        }
        QLineEdit:focus, QComboBox:focus { border: 1px solid #9b8cff; }
        QPushButton {
            background: #252e3e;
            color: #f4f6fb;
            border: 1px solid #354258;
            border-radius: 8px;
            padding: 9px 14px;
            font-weight: 600;
        }
        QPushButton:hover { background: #333e54; border-color: #8a7cff; }
        QPushButton:pressed { background: #4c42a4; }
        QPushButton:disabled { color: #727d8e; background: #1a202a; border-color: #252d39; }
        QCheckBox { spacing: 9px; padding: 5px; }
        QCheckBox::indicator { width: 17px; height: 17px; }
        QProgressBar {
            background: #1b2230; border: 1px solid #303b4d;
            border-radius: 6px; text-align: center; min-height: 15px;
        }
        QProgressBar::chunk { background: #8b7cff; border-radius: 5px; }
        QTableWidget {
            background: #151b25; alternate-background-color: #1a2230;
            gridline-color: #293344; border: 1px solid #303b4d;
            border-radius: 8px; selection-background-color: #403b75;
        }
        QHeaderView::section {
            background: #222b3a; color: #dfe6f3; padding: 9px;
            border: none; border-bottom: 1px solid #384459; font-weight: 700;
        }
        QScrollBar:vertical { background: #151b25; width: 12px; margin: 0; }
        QScrollBar::handle:vertical { background: #3b465a; border-radius: 5px; min-height: 24px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QDialog { background: #11151d; }
        QMessageBox { background: #11151d; }
    """)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

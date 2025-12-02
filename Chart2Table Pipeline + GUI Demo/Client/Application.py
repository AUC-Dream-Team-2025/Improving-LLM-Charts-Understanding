# compare_app.py
import sys
import time
import json
from typing import Optional
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QTextEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QGroupBox, QLineEdit, QSplitter,
    QMessageBox, QProgressBar, QSizePolicy, QDialog, QDialogButtonBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QByteArray, QBuffer

# Worker thread that sends image+question to a model endpoint
class ModelWorker(QThread):
    finished_signal = pyqtSignal(dict, str)   # (json_result, model_name)
    error_signal = pyqtSignal(str, str)       # (error_message, model_name)

    def __init__(self, url: str, question: str, image_bytes: Optional[bytes], model_name: str, timeout: int = 60):
        super().__init__()
        self.url = url
        self.question = question
        self.image_bytes = image_bytes
        self.model_name = model_name
        self.timeout = timeout

    def run(self):
        import requests
        t0 = time.time()
        try:
            files = {}
            data = {"question": self.question}
            if self.image_bytes:
                files["image"] = ("image.jpg", self.image_bytes, "image/jpeg")
            resp = requests.post(self.url, files=files or None, data=data, timeout=self.timeout)
            resp.raise_for_status()
            # always keep the original response text for more flexible display
            raw_text = resp.text
            json_resp = {}
            try:
                json_resp = resp.json()
            except Exception:
                # if not JSON, return text under key 'answer'
                json_resp = {"answer": raw_text}
            # attach the raw text so callers can show the un-escaped original
            try:
                json_resp["_raw_text"] = raw_text
            except Exception:
                pass
            json_resp["_client_elapsed_ms"] = int((time.time() - t0) * 1000)
            self.finished_signal.emit(json_resp, self.model_name)
        except Exception as e:
            self.error_signal.emit(str(e), self.model_name)


class CompareApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Comparator")
        self.resize(1000, 650)

        # central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        # Left side: inputs
        left = QVBoxLayout()
        main_layout.addLayout(left, 1)

        # Endpoint inputs (editable)
        self.model1_url_input = QLineEdit("http://10.40.32.8:8001/predict")
        self.model2_url_input = QLineEdit("http://10.40.32.12:8001/predict")
        left.addWidget(QLabel("Model A endpoint (POST image+question):"))
        left.addWidget(self.model1_url_input)
        left.addWidget(QLabel("Model B endpoint (POST image+question):"))
        left.addWidget(self.model2_url_input)

        # Question input
        left.addWidget(QLabel("Question:"))
        self.question_edit = QTextEdit()
        self.question_edit.setPlaceholderText("Type the user's question here...")
        left.addWidget(self.question_edit, 1)

        # Image attach + preview
        attach_row = QHBoxLayout()
        self.attach_btn = QPushButton("Attach image")
        self.attach_btn.clicked.connect(self.on_attach)
        attach_row.addWidget(self.attach_btn)

        self.clear_img_btn = QPushButton("Clear image")
        self.clear_img_btn.clicked.connect(self.on_clear_image)
        attach_row.addWidget(self.clear_img_btn)

        left.addLayout(attach_row)

        self.image_label = QLabel("No image")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(360, 270)
        self.image_label.setStyleSheet("border: 1px solid #aaa;")
        self.image_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        left.addWidget(self.image_label)

        # Send button and progress
        send_row = QHBoxLayout()
        self.send_btn = QPushButton("Send to both models")
        self.send_btn.clicked.connect(self.on_send)
        send_row.addWidget(self.send_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 2)
        self.progress.setValue(0)
        send_row.addWidget(self.progress)
        left.addLayout(send_row)

        # Right side: two result panels in a splitter (resizable)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 2)

        # Model A panel
        self.model_a_box = self._make_result_box("Model A")
        splitter.addWidget(self.model_a_box)

        # Model B panel
        self.model_b_box = self._make_result_box("Model B")
        splitter.addWidget(self.model_b_box)

        # State
        self.current_pixmap: Optional[QPixmap] = None
        self.current_image_bytes: Optional[bytes] = None
        self.workers = {}  # model_name -> worker

    def _make_result_box(self, title):
        box = QGroupBox(title)
        layout = QVBoxLayout()
        box.setLayout(layout)
        info = QLabel("No response yet")
        info.setWordWrap(True)
        answer = QTextEdit()
        answer.setReadOnly(True)
        raw_btn = QPushButton("Show raw JSON")
        raw_btn.clicked.connect(lambda _, b=box: self._show_raw_json(b))
        save_btn = QPushButton("Save text")
        save_btn.clicked.connect(lambda _, b=box: self._save_answer_text(b))
        save_json_btn = QPushButton("Save raw JSON")
        save_json_btn.clicked.connect(lambda _, b=box: self._save_raw_json(b))
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(raw_btn)
        bottom_row.addWidget(save_btn)
        bottom_row.addWidget(save_json_btn)
        layout.addWidget(info)
        layout.addWidget(answer, 1)
        layout.addLayout(bottom_row)
        # store references for quick access
        box._info_label = info
        box._answer_widget = answer
        box._last_json = None
        return box

    def _show_raw_json(self, box):
        if box._last_json is None:
            QMessageBox.information(self, "Raw JSON", "No JSON available yet for this model.")
            return
        # Prefer showing the original raw response text if available so
        # escape sequences (like "\\n", "\\t", or backslash-escaped
        # symbols) can be rendered more readably.
        raw = box._last_json.get("_raw_text") if isinstance(box._last_json, dict) else None
        pretty = None
        if raw:
            try:
                # attempt to make escape sequences human-friendly
                pretty = bytes(raw, "utf-8").decode("unicode_escape")
            except Exception:
                pretty = raw
        if pretty is None:
            # fallback: pretty-print the parsed JSON
            pretty = json.dumps(box._last_json, indent=2, ensure_ascii=False)
        dlg = QDialog(self)
        dlg.setWindowTitle("Raw JSON")
        dlg.resize(700, 500)
        layout = QVBoxLayout(dlg)
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText(pretty)
        layout.addWidget(txt)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close | QDialogButtonBox.StandardButton.Save)
        def on_close():
            dlg.accept()
        def on_save():
            path, _ = QFileDialog.getSaveFileName(self, "Save raw JSON", "", "JSON files (*.json);;All files (*.*)")
            if path:
                try:
                    with open(path, "w", encoding="utf8") as f:
                        f.write(pretty)
                    QMessageBox.information(self, "Saved", f"Saved JSON to {path}")
                except Exception as e:
                    QMessageBox.warning(self, "Save failed", f"Could not save file: {e}")
        buttons.accepted.connect(on_save)
        buttons.rejected.connect(on_close)
        # Map the Close button (rejected) and Save (accepted) behavior
        layout.addWidget(buttons)
        dlg.exec()

    def _save_answer_text(self, box):
        txt = box._answer_widget.toPlainText()
        if not txt:
            QMessageBox.information(self, "Save", "No answer to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save answer", "", "Text files (*.txt);;All files (*.*)")
        if path:
            with open(path, "w", encoding="utf8") as f:
                f.write(txt)
            QMessageBox.information(self, "Saved", f"Saved to {path}")

    def _save_raw_json(self, box):
        if box._last_json is None:
            QMessageBox.information(self, "Save JSON", "No JSON available yet for this model.")
            return
        pretty = json.dumps(box._last_json, indent=2, ensure_ascii=False)
        path, _ = QFileDialog.getSaveFileName(self, "Save raw JSON", "", "JSON files (*.json);;All files (*.*)")
        if path:
            try:
                with open(path, "w", encoding="utf8") as f:
                    f.write(pretty)
                QMessageBox.information(self, "Saved", f"Saved JSON to {path}")
            except Exception as e:
                QMessageBox.warning(self, "Save failed", f"Could not save file: {e}")

    def on_attach(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All files (*)")
        if not path:
            return
        pix = QPixmap(path)
        if pix.isNull():
            QMessageBox.warning(self, "Invalid image", "Could not load image.")
            return
        # scale to preview size while maintaining aspect ratio
        scaled = pix.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.current_pixmap = pix
        # also store raw bytes (JPEG)
        self.current_image_bytes = self._pixmap_to_jpeg_bytes(self.current_pixmap)

    def on_clear_image(self):
        self.image_label.clear()
        self.image_label.setText("No image")
        self.current_pixmap = None
        self.current_image_bytes = None

    def _pixmap_to_jpeg_bytes(self, pixmap: QPixmap) -> bytes:
        ba = QByteArray()
        buffer = QBuffer(ba)
        buffer.open(QBuffer.OpenModeFlag.WriteOnly)
        # save as JPEG, quality default
        pixmap.save(buffer, "JPEG")
        buffer.close()
        return bytes(ba)

    def on_send(self):
        q = self.question_edit.toPlainText().strip()
        if not q:
            QMessageBox.warning(self, "Empty question", "Please type a question before sending.")
            return

        url_a = self.model1_url_input.text().strip()
        url_b = self.model2_url_input.text().strip()
        if not url_a or not url_b:
            QMessageBox.warning(self, "Empty endpoint", "Please provide both model endpoints.")
            return

        # disable UI elements to avoid repeated sends
        self.send_btn.setEnabled(False)
        self.progress.setValue(0)
        for box in (self.model_a_box, self.model_b_box):
            box._info_label.setText("Waiting for response...")
            box._answer_widget.clear()
            box._last_json = None

        # start two worker threads
        w_a = ModelWorker(url_a, q, self.current_image_bytes, "Model A", timeout=60)
        w_a.finished_signal.connect(self.on_worker_finished)
        w_a.error_signal.connect(self.on_worker_error)
        w_b = ModelWorker(url_b, q, self.current_image_bytes, "Model B", timeout=60)
        w_b.finished_signal.connect(self.on_worker_finished)
        w_b.error_signal.connect(self.on_worker_error)

        self.workers = {"Model A": w_a, "Model B": w_b}
        w_a.start()
        w_b.start()

    def on_worker_finished(self, json_result: dict, model_name: str):
        # map to correct box
        box = self.model_a_box if model_name == "Model A" else self.model_b_box
        answer = json_result.get("answer") or json_result.get("text") or str(json_result)
        confidence = json_result.get("confidence")
        elapsed = json_result.get("_client_elapsed_ms")
        # if the answer contains backslash-escapes (e.g. "\\*", "\\n"),
        # try to unescape them for nicer display
        if isinstance(answer, str) and "\\" in answer:
            try:
                unescaped = bytes(answer, "utf-8").decode("unicode_escape")
                answer = unescaped
            except Exception:
                # leave original if unescape fails
                pass
        # remove backslashes and asterisks from the displayed answer
        if isinstance(answer, str):
            try:
                answer = answer.replace("\\", "").replace("*", "")
            except Exception:
                pass
        # format info
        info_lines = []
        if confidence is not None:
            info_lines.append(f"confidence: {confidence}")
        if elapsed is not None:
            info_lines.append(f"round-trip: {elapsed} ms")
        box._info_label.setText(" | ".join(info_lines) if info_lines else "Response received")
        box._answer_widget.setPlainText(answer)
        box._last_json = json_result
        # update progress
        v = self.progress.value() + 1
        self.progress.setValue(v)
        if v >= 2:
            self.send_btn.setEnabled(True)

    def on_worker_error(self, error_msg: str, model_name: str):
        box = self.model_a_box if model_name == "Model A" else self.model_b_box
        box._info_label.setText(f"Error: {error_msg}")
        box._answer_widget.setPlainText("")
        box._last_json = {"error": error_msg}
        v = self.progress.value() + 1
        self.progress.setValue(v)
        if v >= 2:
            self.send_btn.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    w = CompareApp()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

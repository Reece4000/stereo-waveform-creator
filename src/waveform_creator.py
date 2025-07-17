import numpy as np
from PySide6.QtWidgets import (QWidget, QPushButton, QVBoxLayout, 
                               QHBoxLayout, QLabel, QSpinBox, QFrame,
                               QComboBox, QSizePolicy, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QIcon
import wave
import os
import struct



def show_error_dialog(parent, message, title="Error"):
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.setWindowTitle(title)
    msg_box.setText(str(message))
    msg_box.exec()
    


class WaveformCanvas(QFrame):
    def __init__(self, samples, callback=None, color=(30, 30, 30)):
        super().__init__()
        self.setMinimumHeight(150)
        self.samples = samples
        self.callback = callback
        self.line_color = color
        self.setMouseTracking(True)
        self.drawing = False
        self.last_idx = None
        self.last_val = None

    def set_samples(self, samples):
        self.samples = samples
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Dimensions and margins
        w = self.width()
        h = self.height()

        margin_left = 45
        margin_bottom = 20
        margin_top = 10
        plot_width = w - margin_left
        plot_height = h - margin_bottom - margin_top
        mid_y = margin_top + plot_height // 2

        # Background
        painter.fillRect(self.rect(), QColor(120, 120, 120, 127))

        # --- Gridlines and Ticks ---
        grid_pen = QPen(QColor(80, 80, 80, 100), 1, Qt.DashLine)
        painter.setPen(grid_pen)

        # Y gridlines at -1.0, -0.5, 0.0, 0.5, 1.0
        y_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        for val in y_values:
            y = int(mid_y - val * (plot_height / 2))
            painter.drawLine(margin_left, y, w, y)

        # X gridlines at fixed intervals
        x_ticks = np.linspace(0, len(self.samples) - 1, 8, dtype=int)
        for tick in x_ticks:
            x = int(tick * plot_width / len(self.samples)) + margin_left
            painter.drawLine(x, margin_top, x, h - margin_bottom)

        # --- Axes ---
        axis_pen = QPen(QColor(40, 40, 40), 2)
        painter.setPen(axis_pen)
        painter.drawLine(margin_left, margin_top, margin_left, h - margin_bottom)  # Y-axis
        painter.drawLine(margin_left, mid_y, w, mid_y)  # X-axis (centre)

        # --- Labels ---
        painter.setPen(QColor(20, 20, 20))

        right_padding = 5  # pixels padding from right edge
        font_metrics = painter.fontMetrics()
        for val in y_values:
            y = int(mid_y - val * (plot_height / 2))
            label = f"{val:.1f}"
            painter.drawText(5, y + font_metrics.ascent() // 2, label)

        for i, tick in enumerate(x_ticks):
            x = int(tick * (plot_width - 1) / (len(self.samples) - 1)) + margin_left
            label = str(tick) if tick < len(self.samples) - 1 else str(tick + 1)
            label_width = font_metrics.horizontalAdvance(label)

            if i == 0:
                # Left-align first label
                painter.drawText(x, h - 5, label)
            elif i == len(x_ticks) - 1:
                # Clamp x so text doesn't go out of right boundary
                x_clamped = min(x - label_width, w - label_width - right_padding)
                painter.drawText(x_clamped, h - 5, label)
            else:
                # Centre-align others
                painter.drawText(x - label_width // 2, h - 5, label)

        # --- Waveform ---
        waveform_pen = QPen(QColor(*self.line_color), 2)
        painter.setPen(waveform_pen)

        points = []
        for i, val in enumerate(self.samples):
            x = int(i * plot_width / len(self.samples)) + margin_left
            y = int(mid_y - val * (plot_height / 2))
            points.append(QPoint(x, y))

        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i + 1])

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_idx = None
            self.last_val = None
            self.draw_at(event.pos())

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.draw_at(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            if self.callback:
                self.callback()

    def draw_at(self, pos):
        w = self.width()
        h = self.height()

        margin_left = 45
        margin_bottom = 20
        margin_top = 10
        plot_width = w - margin_left
        plot_height = h - margin_bottom - margin_top
        mid_y = margin_top + plot_height // 2

        idx = int((pos.x() - margin_left) * len(self.samples) / plot_width)
        if idx < 0 or idx >= len(self.samples):
            return

        val = (mid_y - pos.y()) / (plot_height / 2)
        val = max(-1.0, min(1.0, val))

        if self.last_idx is None:
            self.samples[idx] = val
        else:
            start_idx, end_idx = sorted([self.last_idx, idx])
            start_val, end_val = (self.last_val, val) if self.last_idx <= idx else (val, self.last_val)
            length = end_idx - start_idx
            if length == 0:
                self.samples[idx] = val
            else:
                for i in range(start_idx, end_idx + 1):
                    t = (i - start_idx) / length
                    self.samples[i] = start_val * (1 - t) + end_val * t

        self.last_idx = idx
        self.last_val = val
        self.update()


class StereoWaveformDrawer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Stereo Waveform Creator")
        self.setWindowIcon(QIcon("resources/icon.png"))
        with open("resources/stylesheet.txt") as f:
            self.setStyleSheet(f.read())

        self.num_samples = 674  # C2
        self.left_samples = np.zeros(self.num_samples, dtype=np.float32)
        self.right_samples = np.zeros(self.num_samples, dtype=np.float32)

        self.top_canvas = WaveformCanvas(self.left_samples, self.top_changed, color=(150, 80, 80))
        self.bottom_canvas = WaveformCanvas(self.right_samples, self.bottom_changed, color=(70, 140, 140))

        # --- Controls between waveforms ---
        mid_controls_layout = QHBoxLayout()

        def add_btn(label, callback):
            btn = QPushButton(label)
            btn.clicked.connect(callback)
            btn.setMinimumHeight(32)
            mid_controls_layout.addWidget(btn)

        add_btn("Copy L -> R", self.copy_left_to_right)
        add_btn("Copy R -> L", self.copy_right_to_left)
        add_btn("Clear Left", self.reset_left)
        add_btn("Clear Right", self.reset_right)
        add_btn("Load from .wav", self.load_waveform)
        add_btn("Export .wav", self.export_wav)
        
        # Functions Dropdown
        self.functions_combo = QComboBox()
        self.functions_combo.addItems(["Functions", "Normalise", "Rectify", "Remove DC", "Invert Left", "Invert Right", "Fuzz Left", "Fuzz Right", "Randomise Left", "Randomise Right"])
        self.functions_combo.currentIndexChanged.connect(self.apply_func)
        mid_controls_layout.addWidget(self.functions_combo)

        # Preset Dropdown
        self.presets_combo = QComboBox()
        self.presets_combo.addItems(["Presets","Sine", "Triangle", "Saw", "Square", "White Noise"])
        self.presets_combo.currentIndexChanged.connect(self.apply_preset)
        mid_controls_layout.addWidget(self.presets_combo)

        # --- Layout ---
        wave_layout = QVBoxLayout()
        wave_layout.addWidget(self.top_canvas, stretch=1)
        wave_layout.addLayout(mid_controls_layout)
        wave_layout.addWidget(self.bottom_canvas, stretch=1)

        self.setLayout(wave_layout)
        self.resize(800, 500)

    def top_changed(self):
        self.bottom_canvas.update()

    def bottom_changed(self):
        self.top_canvas.update()

    # --- Button Methods ---
    def copy_left_to_right(self):
        self.right_samples[:] = self.left_samples
        self.bottom_canvas.update()

    def copy_right_to_left(self):
        self.left_samples[:] = self.right_samples
        self.top_canvas.update()

    def reset_left(self):
        self.left_samples[:] = 0
        self.top_canvas.update()

    def reset_right(self):
        self.right_samples[:] = 0
        self.bottom_canvas.update()

    def normalise(self):
        for channel in [self.left_samples, self.right_samples]:
            peak = np.max(np.abs(channel))
            if peak > 0:
                channel[:] = channel / peak
        self.top_canvas.update()
        self.bottom_canvas.update()

    def rectify(self):
        self.left_samples[:] = np.abs(self.left_samples)
        self.right_samples[:] = np.abs(self.right_samples)
        self.top_canvas.update()
        self.bottom_canvas.update()

    def remove_dc(self):
        self.left_samples[:] = self.left_samples - np.mean(self.left_samples)
        self.right_samples[:] = self.right_samples - np.mean(self.right_samples)
        self.top_canvas.update()
        self.bottom_canvas.update()

    def invert(self):
        self.left_samples[:] *= -1
        self.right_samples[:] *= -1
        self.top_canvas.update()
        self.bottom_canvas.update()
        
    def add_local_noise(self):
        # Apply small random noise to existing samples to perturb the waveform slightly
        noise_amplitude = 0.02  # Adjust this for how much randomness you want
        self.left_samples += np.random.uniform(-noise_amplitude, noise_amplitude, self.num_samples)
        self.right_samples += np.random.uniform(-noise_amplitude, noise_amplitude, self.num_samples)
        
        # Clip to keep within [-1, 1]
        np.clip(self.left_samples, -1, 1, out=self.left_samples)
        np.clip(self.right_samples, -1, 1, out=self.right_samples)

        self.top_canvas.update()
        self.bottom_canvas.update()
        

    def randomise(self):
        n = self.num_samples
        
        # Generate smooth random perturbation by interpolating fewer random points
        downsample_factor = 16
        base_noise = np.random.normal(0, 1, n // downsample_factor + 2)
        smooth_noise = np.interp(np.linspace(0, len(base_noise) - 1, n), np.arange(len(base_noise)), base_noise)
        
        # Normalize smooth noise to max abs 1, scale to ~0.3 amplitude
        smooth_noise /= np.max(np.abs(smooth_noise))
        smooth_noise *= 0.3
        
        # Add perturbation in-place without replacing the arrays (preserving references)
        self.left_samples += smooth_noise
        self.right_samples += smooth_noise
        
        # Clip samples in-place
        np.clip(self.left_samples, -1, 1, out=self.left_samples)
        np.clip(self.right_samples, -1, 1, out=self.right_samples)
        
        # Trigger canvas updates
        self.top_canvas.update()
        self.bottom_canvas.update()
        
    def load_waveform_from_wav(self, path):
        try:
            with wave.open(path, 'rb') as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()

                raw_data = wf.readframes(n_frames)

            # Number of samples per channel
            total_samples = n_frames
            max_channels = min(n_channels, 2)

            # Convert raw bytes to integers
            if sample_width == 1:
                fmt = f'{n_frames * n_channels}B'
                data = np.array(struct.unpack(fmt, raw_data), dtype=np.uint8)
                data = (data - 128) / 128.0
            elif sample_width == 2:
                fmt = f'<{n_frames * n_channels}h'
                data = np.array(struct.unpack(fmt, raw_data), dtype=np.int16)
                data = data / 32768.0
            elif sample_width == 3:
                # 24-bit PCM handling
                data = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, sample_width * n_channels)
                samples = []
                for frame in data:
                    frame_samples = []
                    for ch in range(max_channels):
                        i = ch * 3
                        sample_bytes = frame[i:i+3]
                        as_int = int.from_bytes(sample_bytes, byteorder='little', signed=True)
                        frame_samples.append(as_int / (2**23))
                    samples.append(frame_samples)
                data = np.array(samples)
            elif sample_width == 4:
                fmt = f'<{n_frames * n_channels}i'
                data = np.array(struct.unpack(fmt, raw_data), dtype=np.int32)
                data = data / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            if sample_width in (1, 2, 4):
                data = data.reshape(-1, n_channels)[:, :max_channels]

            # Resample or stretch each channel
            def resample(channel_data):
                old_indices = np.linspace(0, len(channel_data) - 1, num=len(channel_data))
                new_indices = np.linspace(0, len(channel_data) - 1, num=self.num_samples)
                return np.interp(new_indices, old_indices, channel_data)

            left = resample(data[:, 0])
            right = resample(data[:, 1]) if max_channels == 2 else np.zeros_like(left)

            self.left_samples[:] = left
            self.right_samples[:] = right

            self.top_canvas.update()
            self.bottom_canvas.update()

        except Exception as e:
            show_error_dialog(self, f"Could not load waveform:\n{e}")
        
    def load_waveform(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Waveform", "", "WAV files (*.wav)")
        if file_path:
            self.load_waveform_from_wav(file_path)
            

    def export_wav(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export as WAV", "", "WAV files (*.wav)")
        if file_path:
            samplerate = 44100
            sampwidth = 2  # 16-bit PCM

            # Check if right channel has valid data
            is_stereo = (
                self.right_samples is not None and
                np.any(self.right_samples)
            )

            if is_stereo:
                frames = np.stack([self.left_samples, self.right_samples], axis=-1)
                nchannels = 2
            else:
                frames = self.left_samples
                nchannels = 1

            int_data = np.int16(frames * 32767)

            with wave.open(file_path, "wb") as f:
                f.setnchannels(nchannels)
                f.setsampwidth(sampwidth)
                f.setframerate(samplerate)
                f.writeframes(int_data.tobytes())

            print("Exported WAV to", file_path)

    def apply_preset(self, index):
        if index == 0:
            return  # "Presets" label

        t = np.linspace(0, 1, self.num_samples, endpoint=False)
        waveform = np.zeros_like(t)

        text = self.presets_combo.currentText()
        if text == "Sine":
            waveform = np.sin(2 * np.pi * t)
        elif text == "Triangle":
            waveform = 2 * np.abs(2 * (t % 1) - 1) - 1
        elif text == "Saw":
            waveform = 2 * (t % 1) - 1
        elif text == "Square":
            waveform = np.sign(np.sin(2 * np.pi * t))
        elif text == "White Noise":
            waveform = np.random.uniform(-1, 1, self.num_samples)

        self.left_samples[:] = waveform
        self.right_samples[:] = waveform
        self.top_canvas.update()
        self.bottom_canvas.update()
        
    def apply_func(self, index):
        if index == 0:
            return  # "Presets" label

        text = self.functions_combo.currentText()
        # ["Normalise", "Rectify", "Remove DC", "Invert Left", "Invert Right", "Fuzz Left", "Fuzz Right", "Randomise Left", "Randomise Right"]
        if text == "Normalise":
            self.normalise()



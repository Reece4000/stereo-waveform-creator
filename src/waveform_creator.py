import numpy as np
from PySide6.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QHBoxLayout, 
                               QFileDialog, QMessageBox, QMenu)
from PySide6.QtGui import QIcon, QAction
from src.waveform_canvas import WaveformCanvas
import wave
import struct

# stereo_waveform_creator
# A simple tool for creating stereo single cycle waveforms. Supports exporting to wav, loading of existing wav files from disk, and a few additional functions


def show_error_dialog(parent, message, title="Error"):
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.setWindowTitle(title)
    msg_box.setText(str(message))
    msg_box.exec()
    

class WaveformCreator(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Stereo Waveform Creator")
        self.setWindowIcon(QIcon("resources/icon.png"))
        with open("styles/standard.qss") as f:
            self.stylesheet = f.read()
            self.setStyleSheet(self.stylesheet)

        self.num_samples = 674  # C2
        self.left_samples = np.zeros(self.num_samples, dtype=np.float32)
        self.right_samples = np.zeros(self.num_samples, dtype=np.float32)

        self.top_canvas = WaveformCanvas(self.left_samples, "left", self.left_changed, color=(150, 80, 80))
        self.bottom_canvas = WaveformCanvas(self.right_samples, "right", self.right_changed, color=(70, 140, 140))

        # --- Controls between waveforms ---
        mid_controls_layout = QHBoxLayout()

        def add_btn(label, callback):
            btn = QPushButton(label)
            btn.clicked.connect(callback)
            btn.setMinimumHeight(32)
            mid_controls_layout.addWidget(btn)

        add_btn("Load from .wav", self.load_waveform)
        add_btn("Export .wav", self.export_wav)

        self.presets_button = QPushButton("Presets")
        self.presets_menu = QMenu()

        # List of presets
        presets = ["Sine", "Triangle", "Saw Up", "Saw Down", "Square", "White Noise", "Sinc",
                   "Pyramid", "PWM", "Ellipsoid", "Tangent", "Exponential"]

        for preset in presets:
            action = QAction(preset, self)
            action.triggered.connect(lambda checked, p=preset: self.apply_preset(p))
            self.presets_menu.addAction(action)
        self.presets_menu.setStyleSheet(self.stylesheet)

        self.presets_button.setMenu(self.presets_menu)
        mid_controls_layout.addWidget(self.presets_button)

        # --- Layout ---
        wave_layout = QVBoxLayout()
        wave_layout.addWidget(self.top_canvas, stretch=1)
        wave_layout.addLayout(mid_controls_layout)
        wave_layout.addWidget(self.bottom_canvas, stretch=1)

        self.setLayout(wave_layout)
        self.resize(800, 500)

    def left_changed(self):
        self.top_canvas.update()

    def right_changed(self):
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

    def apply_preset(self, preset):

        t = np.linspace(0, 1, self.num_samples, endpoint=False)
        waveform = np.zeros_like(t)

        if preset == "Sine":
            waveform = np.sin(2 * np.pi * t)

        elif preset == "Triangle":
            # Full cycle triangle wave
            waveform = 2 * np.abs(2 * (t % 1) - 1) - 1

        elif preset == "Saw Up":
            waveform = 2 * (t % 1) - 1

        elif preset == "Saw Down":
            waveform = 1 - 2 * (t % 1)

        elif preset == "Square":
            waveform = np.sign(np.sin(2 * np.pi * t))

        elif preset == "White Noise":
            waveform = np.random.uniform(-1, 1, self.num_samples)

        elif preset == "Sinc":
            # Windowed to one cycle by 8-lobe width
            waveform = np.sinc(8 * (t - 0.5))

        elif preset == "Pyramid":
            steps = 10  # Must be even
            half = steps // 2
            up = np.linspace(-1, 1, half + 1)
            down = np.linspace(1, -1, half + 1)[1:]  # remove duplicate peak
            stair = np.concatenate((up, down))  # total = steps
            indices = np.floor(t * steps).astype(int) % steps
            waveform = stair[indices]

        elif preset == "PWM":
            duty_cycle = 0.2  # range: 0 < dc < 1
            waveform = np.where((t % 1) < duty_cycle, 1.0, -1.0)

        elif preset == "Ellipsoid":
            # Symmetric ellipsoid arc covering full cycle
            x = 2 * (t - 0.5)  # range [-1, 1]
            waveform = np.sqrt(np.clip(1 - x ** 2, 0, 1)) * 2 - 1  # scale to [-1, 1]

        elif preset == "Tangent":
            waveform = np.tan(2 * np.pi * t)
            waveform = np.clip(waveform, -1, 1)

        elif preset == "Exponential":
            # Rising then falling exponential over full cycle
            half = self.num_samples // 2
            rise = np.exp(np.linspace(0, 4, half, endpoint=False))
            fall = np.exp(np.linspace(4, 0, self.num_samples - half))
            waveform = np.concatenate((rise, fall))
            waveform = waveform / np.max(waveform)  # normalise to [0, 1]
            waveform = waveform * 2 - 1  # scale to [-1, 1]

        self.left_samples[:] = waveform
        self.right_samples[:] = waveform
        self.top_canvas.update()
        self.bottom_canvas.update()





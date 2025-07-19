import numpy as np
from PySide6.QtWidgets import QFrame, QMenu
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QAction 


CANVAS_COLOR = QColor(0, 0, 0, 255)
GRIDLINES_COLOR = QColor(255, 255, 255, 50)
AXIS_COLOR = QColor(255, 255, 255, 50)
LABELS_COLOR = QColor(255, 255, 255, 255)


class WaveformCanvas(QFrame):   

    def __init__(self, samples, channel, callback=None, color=(30, 30, 30)):
        super().__init__()
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.channel = channel
        self.setMinimumHeight(150)
        self.samples = samples
        self.callback = callback
        self.line_color = color
        self.setMouseTracking(True)
        self.drawing = False
        self.last_idx = None
        self.last_val = None
        
    def show_context_menu(self, pos):
        menu = QMenu(self)

        options = {
            "Copy to other channel": self.copy_to_other_channel,
            "Clear": self.clear_channel,
            "Rectify": self.rectify_channel,
            "Invert": self.invert_channel,
            "Normalise": self.normalise_channel,
            "Remove DC offset": self.remove_dc_offset,
            "Fuzz": self.fuzz_channel,
            "Randomise": self.randomise_channel,
        }

        for text, callback in options.items():
            action = QAction(text, self)
            action.triggered.connect(callback)
            menu.addAction(action)

        menu.exec(self.mapToGlobal(pos))

    def copy_to_other_channel(self):
        if self.channel == "left":
            self.parent().right_samples[:] = self.samples[:]
            self.parent().right_changed()
        else:
            self.parent().left_samples[:] = self.samples[:]
            self.parent().left_changed()

    def clear_channel(self):
        self.samples[:] = 0
        self.update()

    def rectify_channel(self):
        self.samples[:] = np.abs(self.samples)
        self.update()

    def invert_channel(self):
        self.samples[:] *= -1
        self.update()

    def normalise_channel(self):
        peak = np.max(np.abs(self.samples))
        if peak > 0:
            self.samples[:] = self.samples / peak
            self.update()

    def remove_dc_offset(self):
        self.samples -= np.mean(self.samples)
        peak = np.max(np.abs(self.samples))
        if peak > 1.0:
            self.samples /= peak 
        self.update()

    def fuzz_channel(self):
        noise = 0.02
        self.samples += np.random.uniform(-noise, noise, len(self.samples))
        np.clip(self.samples, -1, 1, out=self.samples)
        self.update()

    def randomise_channel(self):
        n = len(self.samples)
        
        # Generate smooth random perturbation by interpolating fewer random points
        downsample_factor = 16
        base_noise = np.random.normal(0, 1, n // downsample_factor + 2)
        smooth_noise = np.interp(np.linspace(0, len(base_noise) - 1, n), np.arange(len(base_noise)), base_noise)
        
        # Normalize smooth noise to max abs 1, scale to ~0.3 amplitude
        smooth_noise /= np.max(np.abs(smooth_noise))
        smooth_noise *= 0.3
        
        # Add perturbation in-place 
        self.samples += smooth_noise
        np.clip(self.samples, -1, 1, out=self.samples)
        self.update()

    def set_samples(self, samples):
        self.samples = samples
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()

        margin_left = 45
        margin_bottom = 20
        margin_top = 10
        plot_width = w - margin_left
        plot_height = h - margin_bottom - margin_top
        mid_y = margin_top + plot_height // 2

        painter.fillRect(self.rect(), CANVAS_COLOR)  # bg

        grid_pen = QPen(GRIDLINES_COLOR, 1, Qt.DashLine)
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
        axis_pen = QPen(AXIS_COLOR)
        painter.setPen(axis_pen)
        painter.drawLine(margin_left, margin_top, margin_left, h - margin_bottom)  # Y-axis
        painter.drawLine(margin_left, mid_y, w, mid_y)  # X-axis (centre)

        # --- Labels ---
        painter.setPen(LABELS_COLOR)

        right_padding = 5  
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
        
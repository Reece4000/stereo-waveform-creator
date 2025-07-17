import sys
from PySide6.QtWidgets import QApplication
from src.waveform_creator import StereoWaveformDrawer


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StereoWaveformDrawer()
    window.show()
    sys.exit(app.exec())

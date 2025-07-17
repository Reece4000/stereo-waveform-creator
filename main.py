import sys
from PySide6.QtWidgets import QApplication
from src.waveform_creator import WaveformCreator


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WaveformCreator()
    window.show()
    sys.exit(app.exec())

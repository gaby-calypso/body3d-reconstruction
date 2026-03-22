"""
app.py
------
Punto de entrada de la aplicación GUI.

Uso:
    python3 app.py
"""

import sys
from PyQt5.QtWidgets import QApplication
from src.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Body3D Reconstruction")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
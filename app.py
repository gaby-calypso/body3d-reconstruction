"""
app.py
------
Punto de entrada de la aplicación GUI.

Uso:
    python3 app.py
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from src.gui.main_window import MainWindow, FONT_FAMILY


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Body3D Reconstruction")
    app.setStyle("Fusion")  # base consistente entre plataformas para que la QSS se vea igual en todos lados
    app.setFont(QFont(FONT_FAMILY.split(",")[0].strip(' "'), 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
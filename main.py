import sys

import app

if __name__ == "__main__":
    qt_app = app.QApplication(sys.argv)
    app.apply_dark_theme(qt_app)
    window = app.MainWindow()
    window.showMaximized()
    sys.exit(qt_app.exec())

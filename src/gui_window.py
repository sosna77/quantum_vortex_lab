import numpy as np
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from PySide6.QtGui import QTransform, QGuiApplication
from PySide6.QtCore import Qt
import pyqtgraph as pg
from engine import GPEWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Quantum Vortex Lab')
        self.resize(800, 800)

        # robimy layout i widget do wizualizacji
        layout = QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # ustawiamy główny widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        self.plot_widget.setLabel('bottom','X')
        self.plot_widget.setLabel('left','Y')

        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)

        L = 10
        N = 256
        DX = L / N
        self.image_item.setPos(-L/2, -L/2)
        transform = QTransform()
        transform.scale(DX, DX)
        self.image_item.setTransform(transform)

        # ładniejszy pyqtgraph
        cmap = pg.colormap.get('viridis')
        self.image_item.setColorMap(cmap)

        self.plot_widget.setXRange(-5, 5, padding=0)
        self.plot_widget.setYRange(-5, 5, padding=0)
        self.plot_widget.getViewBox().disableAutoRange()

        self.first_frame = True
        self.max_val = 1

        # ==== SETUP MYSZKI ====
        # mysz nie przesuwa wykresu
        self.plot_widget.setMouseEnabled(x=False, y=False)

        # połączenie sygnału ruchu myszy
        self.plot_widget.scene().sigMouseMoved.connect(self.mouse_moved)


        # worker setup
        self.worker = GPEWorker()

        # połączenie sygnału do slotu (metody update_screen)
        self.worker.frame_ready.connect(self.update_screen)
        self.worker.start()

    def mouse_moved(self, pos):
        view_box = self.plot_widget.getViewBox()

        if view_box.sceneBoundingRect().contains(pos):
            mouse_point = view_box.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()

            # sprawdzamy czy kliknięta
            is_pressed = bool(QGuiApplication.mouseButtons() & Qt.MouseButton.LeftButton)

            self.worker.update_spoon(x, y, active=is_pressed)

        else:
            self.worker.update_spoon(0, 0, active=False)


    def update_screen(self, ksi):
        rho = np.abs(ksi.T)**2

        if self.first_frame:
            self.max_val = np.max(rho)
            self.threshold = self.max_val * 0.005
            self.first_frame = False
        rho[rho < self.threshold] = 0

        self.image_item.setImage(rho, autoLevels=False, levels=(0, self.max_val))
            

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()
        
import numpy as np
from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QSlider, QLabel
from PySide6.QtGui import QTransform, QGuiApplication
from PySide6.QtCore import Qt
import pyqtgraph as pg
from .engine import GPEWorker
from .config import SimConfig

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Quantum Vortex Lab')
        self.resize(1200, 650)

        # robimy layout i widget do wizualizacji

        plots_layout = QHBoxLayout()        # layout na ploty (poziom)
        plots_layout.setSpacing(30)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom','X')
        self.plot_widget.setLabel('left','Y')
        self.plot_widget.setTitle("Gęstość Prawdopodobieństwa |ψ|²", color='#EEE', size='14pt')
        self.plot_widget.getViewBox().disableAutoRange()
        self.plot_widget.getViewBox().setAspectLocked(True)
        plots_layout.addWidget(self.plot_widget)

        self.phase_widget = pg.PlotWidget()
        self.phase_widget.setLabel('bottom','X')
        self.phase_widget.setLabel('left','Y')
        self.phase_widget.setTitle("Faza Funkcji Falowej arg(ψ)", color='#EEE', size='14pt')
        self.phase_widget.getViewBox().disableAutoRange()
        self.phase_widget.getViewBox().setAspectLocked(True)
        plots_layout.addWidget(self.phase_widget)

        # --- PRZYCISK RESET ---
        self.btn_reset = QPushButton('Reset symulacji (ponowne ITE)')
        self.btn_reset.clicked.connect(self.on_reset_clicked)

        slider_layout = QHBoxLayout()
        self.gamma_label = QLabel('Temperatura: 0.01')
        self.gamma_label.setFixedWidth(150)

        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(0, 100)
        self.gamma_slider.setValue(10)
        self.gamma_slider.valueChanged.connect(self.on_gamma_changed)

        slider_layout.addWidget(self.gamma_label)
        slider_layout.addWidget(self.gamma_slider)

        main_layout = QVBoxLayout()         # główny layout (pion)
        main_layout.setContentsMargins(30, 50, 30, 30)
        main_layout.setSpacing(20)
        main_layout.addLayout(plots_layout)
        main_layout.addLayout(slider_layout)
        main_layout.addWidget(self.btn_reset)

        # ustawiamy główny widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    

        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)

        self.phase_item = pg.ImageItem()
        self.phase_widget.addItem(self.phase_item)

        L = 10
        N = 256
        DX = L / N
        self.image_item.setPos(-L/2, -L/2)
        transform = QTransform()
        transform.scale(DX, DX)
        self.image_item.setTransform(transform)

        self.phase_item.setPos(-L/2, -L/2)
        self.phase_item.setTransform(transform)

        # ładniejszy pyqtgraph
        cmap = pg.colormap.get('viridis')
        self.image_item.setColorMap(cmap)

        self.plot_widget.setXRange(-5, 5, padding=0)
        self.plot_widget.setYRange(-5, 5, padding=0)
        self.plot_widget.getViewBox().disableAutoRange()

        cmap2 = pg.colormap.getFromMatplotlib('hsv')
        self.phase_item.setColorMap(cmap2)

        self.phase_widget.setXRange(-5, 5, padding=0)
        self.phase_widget.setYRange(-5, 5, padding=0)
        self.phase_widget.getViewBox().disableAutoRange()


        self.first_frame = True
        self.max_val = 1

        # ==== SETUP MYSZKI ====
        # mysz nie przesuwa wykresu
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.phase_widget.setMouseEnabled(x=False, y=False)

        # połączenie sygnału ruchu myszy
        self.plot_widget.scene().sigMouseMoved.connect(self.mouse_moved)
        self.phase_widget.scene().sigMouseMoved.connect(self.mouse_moved)


        # worker setup
        cfg = SimConfig()
        self.worker = GPEWorker(cfg)

        # połączenie sygnału do slotu (metody update_screen)
        self.worker.frame_ready.connect(self.update_screen)
        self.worker.start()
    def on_gamma_changed(self, value):
        real_gamma = value / 1000.0
        self.gamma_label.setText(f'Temperatura: {real_gamma:.3f}')

        if hasattr(self, 'worker'):
            self.worker.set_gamma(real_gamma)

    def on_reset_clicked(self):
        self.worker.trigger_reset()

    def mouse_moved(self, pos):
        vb_plot = self.plot_widget.getViewBox()
        vb_phase = self.phase_widget.getViewBox()
        is_pressed = bool(QGuiApplication.mouseButtons() & Qt.MouseButton.LeftButton)

        if vb_plot.sceneBoundingRect().contains(pos):
            mouse_point = vb_plot.mapSceneToView(pos)
            self.worker.update_spoon(mouse_point.x(), mouse_point.y(), active=is_pressed)
        elif vb_phase.sceneBoundingRect().contains(pos):
            mouse_point = vb_phase.mapSceneToView(pos)
            self.worker.update_spoon(mouse_point.x(), mouse_point.y(), active=is_pressed)
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

        phase = np.angle(ksi.T)
        phase[rho == 0] = 0
        self.phase_item.setImage(phase, autoLevels=False, levels=(-np.pi, np.pi))
            

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()
        
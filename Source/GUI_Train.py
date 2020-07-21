import sys
import Source.AI_Backend as AIB
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QLineEdit, QStatusBar, QSizePolicy
import pyqtgraph as pg

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt

class img_plot(QWidget):

    def create_matplot(self, layout, ker, titles, figsize = (2,2)):
        self.bcol = (1, 1, 1)
        self.fig, self.axs = plt.subplots(1, 5, figsize=figsize)
        self.fig.patch.set_facecolor(self.bcol)
        self.im = []

        #self.fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        for j in range(5):
            img = ker[:, :, :, j]
            img = img[..., ::-1]
            img = np.clip(img,0.0,1.0)
            self.im.append(self.axs[j].imshow(img))
            self.axs[j].title.set_text(titles[j])
        self.plot = FigureCanvas(self.fig)
        layout.addWidget(self.plot)

    def update_matplot(self, ker, titles):
        for j in range(5):
            img = ker[:, :, :, j]
            img = img[..., ::-1]
            img = np.clip(img,0.0,1.0)
            self.im[j].set_data(img)
            self.axs[j].title.set_text(titles[j])
            self.axs[j].axis("off")
        self.plot.draw()

class Window(QWidget):
    """Main Window."""
    def __init__(self, data_path = r"\Data", parent=None):
        """Initializer."""
        self.sidebar_width = 300

        super().__init__(parent)

        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)

        self.addAI(data_path=data_path)


        self.setWindowTitle('VarroAI')
        self.top_layout = QVBoxLayout()
        self.layout = QHBoxLayout()
        self.sidebar = QVBoxLayout()
        self.create_plot(self.layout)


        self.create_train_button(self.sidebar)
        self.create_save_button(self.sidebar)
        self.sidebar.addStretch(1)
        self.create_load_button(self.sidebar)
        self.layout.addLayout(self.sidebar)
        self.top_layout.addLayout(self.layout)
        self.plot = img_plot()
        self.plot.create_matplot(self.top_layout, self.x.getKernels()[0],[1,2,3,4,5])
        self.plot2 = img_plot()
        self.plot2.create_matplot(self.top_layout, np.moveaxis(self.x.ValData[:5],0,-1), self.x.make_prediction_ValData()[:5])
        self.setLayout(self.top_layout)

    def create_plot(self, layout):
        self.my_plot = pg.PlotWidget()
        layout.addWidget(self.my_plot)
        self.my_plot.addLegend()
        self.my_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.my_plot.plot(self.ax, self.acc, pen=(0, 2), name="accuracy", clear=True)
        self.my_plot.plot(self.ax, self.loss, pen=(1, 2), name="loss", clear=True)

    def update_plot(self):
        l, a = self.x.evaluateModel()
        self.loss.append(l)
        self.acc.append(a)
        self.ax.append(self.loss.__len__())
        self.my_plot.plot(self.ax, self.acc, pen=(0, 2), clear=True)
        self.my_plot.plot(self.ax, self.loss, pen=(1, 2))
        QApplication.processEvents()
    def create_train_button(self, layout):
        self.input = QLineEdit()
        self.input.setMaximumWidth(self.sidebar_width)
        layout.addWidget(self.input)
        plotButton = QPushButton("Train")
        plotButton.setMaximumWidth(self.sidebar_width)
        layout.addWidget(plotButton)
        plotButton.clicked.connect(self.train)

    def create_save_button(self, layout):
        self.save_input = QLineEdit()
        self.save_input.setMaximumWidth(self.sidebar_width)
        layout.addWidget(self.save_input)
        self.saveButton = QPushButton("Save")
        self.saveButton.setMaximumWidth(self.sidebar_width)
        layout.addWidget(self.saveButton)
        self.saveButton.clicked.connect(self.save)
    def create_load_button(self, layout):
        self.load_input = QLineEdit()
        self.load_input.setMaximumWidth(self.sidebar_width)
        layout.addWidget(self.load_input)
        self.loadButton = QPushButton("Load")
        self.loadButton.setMaximumWidth(self.sidebar_width)
        layout.addWidget(self.loadButton)
        self.loadButton.clicked.connect(self.load)
    def save(self):
        dir= self.save_input.text()
        self.x.saveModel(dir)
    def load(self):
        dir = self.load_input.text()
        self.x.loadModel(dir)
        self.acc = []
        self.loss = []
        self.ax = []
        self.plot2.update_matplot(np.moveaxis(self.x.ValData[:5], 0, -1),
                                  self.x.make_prediction_ValData()[:5])
        self.plot.update_matplot(self.x.getKernels()[0], [1, 2, 3, 4, 5])
        self.update_plot()
    def train(self):
        for i in range(int(self.input.text())):
            self.x.trainDataGen(1)
            self.update_plot()
            QApplication.processEvents()
            self.plot2.update_matplot(np.moveaxis(self.x.ValData[:5], 0, -1),
                                      self.x.make_prediction_ValData()[:5])
            self.plot.update_matplot(self.x.getKernels()[0], [1, 2, 3, 4, 5])

        self.x.saveModel("Autosave.hdf5")

    def addAI(self, data_path = r"\Data"):
        self.acc = []
        self.loss = []
        self.ax =[]
        self.x = AIB.AI()
        self.x.LoadData(data_path)
        self.x.addModel((100,100,3))

def start_training_GUI(data_path= r"\Data"):
    app = QApplication(sys.argv)
    win = Window(data_path=data_path)
    win.show()
    sys.exit(app.exec_())
#  Copyright (c) 2023. Salim Janji.
#   All rights reserved.

import sys
import time
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg


class Plotter(QtGui.QMainWindow):
    def __init__(self, parent=None):
        self.app = QtGui.QApplication([])
        super().__init__(parent)
        self.start_time = time.time()
        self.users = []
        self.users_plotter = []
        self.shadow_xs = []
        self.shadow_ys = []
        self.shadows_state = []
        self.shadows_plotter = []
        self.shadows_update_flag = True
        self.shadows_flag = False
        self.base_stations = None
        self.base_stations_plotter = []

        # Configurations
        pg.setConfigOption('background', 'w')

        # Create GUI elements
        self.mainbox = QtGui.QWidget()
        self.setCentralWidget(self.mainbox)
        self.mainbox.setLayout(QtGui.QVBoxLayout())

        self.canvas = pg.GraphicsLayoutWidget()
        self.mainbox.layout().addWidget(self.canvas)

        self.label = QtGui.QLabel()
        self.mainbox.layout().addWidget(self.label)

        # self.view = self.canvas.addViewBox()
        # self.view.setAspectLocked(True)
        # self.view.setRange(QtCore.QRectF(0, 0, 100, 100))

        #  image plot
        self.plot_widget = self.canvas.addPlot()
        # self.plot_widget.hideAxis('left')
        # self.plot_widget.hideAxis('bottom')

        #####################
        self.counter = 0
        self.fps = 0.
        self.last_update = time.time()

    def _update(self):
        for _user, _plotter in zip(self.users, self.users_plotter):
            _plotter.setData(x=[_user[0]], y=[_user[1]])
        if self.shadows_flag:
            self.update_shadows()
        if self.base_stations:
            for _bs, _plotter in zip(self.base_stations, self.base_stations_plotter):
                _plotter.setData(x=[_bs[0]], y=[_bs[1]])

        # now = time.time()
        # dt = (now - self.last_update)
        # if dt <= 0:
        #     dt = 0.000000000001
        # fps2 = 1.0 / dt
        # self.last_update = now
        # self.fps = self.fps * 0.9 + fps2 * 0.1
        # tx = 'Mean Frame Rate:  {fps:.3f} FPS'.format(fps=self.fps)
        time_diff = int(time.time() - self.start_time)
        tx = f'Time passed = {int(time_diff / 60)}:{time_diff % 60}'
        self.label.setText(tx)
        QtCore.QTimer.singleShot(100, self._update)

    def set_users(self, users_list):
        self.users = users_list
        for _user in users_list:
            self.users_plotter.append(self.plot_widget.plot(x=[_user[0]], y=[_user[1]],
                                                            symbol='o', pen=None, color='b'))

    def set_base_stations(self, base_stations):
        self.base_stations = base_stations
        for _bs in base_stations:
            self.base_stations_plotter.append(self.plot_widget.plot(x=[_bs[0]], y=[_bs[1]],
                                                                    symbol='x', pen=None, color='r'))

    def set_shadows(self, shadow_xs, shadow_ys, shadows_state, shadows_update_flag):
        self.shadows_flag = True
        self.shadow_xs = shadow_xs
        self.shadow_ys = shadow_ys
        self.shadows_state = shadows_state
        self.shadows_update_flag = shadows_update_flag
        self.shadows_plotter = pg.ScatterPlotItem(symbol='s', pxMode=False)
        self.plot_widget.addItem(self.shadows_plotter)
        self.update_shadows()

    def update_shadows(self):
        if not self.shadows_update_flag:
            return
        self.shadows_plotter.clear()
        spots = [None] * self.shadow_xs.size * self.shadow_ys.size
        gray_color_brush = pg.mkBrush(color=pg.hsvColor(0, 0, 0.5, 0.4))
        yellow_color_brush = pg.mkBrush(color=pg.hsvColor(60 / 360, 1, 1, 0.1))
        with self.shadows_state.get_lock():
            shadow_state_array = to_numpy_array(self.shadows_state, self.shadow_ys.size)
            for idx_x in range(self.shadow_xs.size):
                for idx_y in range(self.shadow_ys.size):
                    spot_dic = {'pos': (self.shadow_xs[idx_x], self.shadow_ys[idx_y]), 'size': 0.5,
                                'pen': None,
                                'brush': gray_color_brush if shadow_state_array[idx_x, idx_y] else yellow_color_brush}
                    spots[idx_x * self.shadow_ys.size + idx_y] = spot_dic
        self.shadows_plotter.addPoints(spots)
        self.shadows_update_flag.clear()

    def set_fixed_background(self, background_objects, color=None, width=None, style=None):

        if style == 'dotted':
            _style = QtCore.Qt.DotLine
        elif style == 'dashed':
            _style = QtCore.Qt.DashLine
        else:
            _style = None
        _pen = pg.mkPen(color=color, width=width, style=_style)
        for stroke in background_objects:
            self.plot_widget.plot(stroke[0], stroke[1],
                                  pen=_pen)

    def start_plotter(self):
        self._update()
        self.show()
        QtGui.QApplication.instance().exec_()


def to_numpy_array(mp_arr, y_size):
    _array = np.frombuffer(mp_arr.get_obj(), dtype=bool)
    return np.reshape(_array, (-1, y_size))


if __name__ == '__main__':
    plotter = Plotter()
    plotter.start_plotter()

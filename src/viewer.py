import sys, time, threading, queue
import numpy as np
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

class SimpleViewer(QtWidgets.QMainWindow):
    def __init__(self, history=1000, video_path="output.mp4", fps=30):
        super().__init__()
        self.resize(1280, 720)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(splitter)

        # image pane
        self.img_widget = pg.GraphicsLayoutWidget(); splitter.addWidget(self.img_widget)
        self.img_item = pg.ImageItem(); vb = self.img_widget.addViewBox(lockAspect=True)
        vb.addItem(self.img_item)

        # 3‑D pane
        self.gl = gl.GLViewWidget(); self.gl.opts["distance"] = 6; splitter.addWidget(self.gl)
        axis = gl.GLAxisItem(size=QtGui.QVector3D(1,1,1)); self.gl.addItem(axis)
        self.est_line = gl.GLLinePlotItem(color=(1,0,0,1), width=2, antialias=True)
        self.pts_item = gl.GLScatterPlotItem(size=4, color=(1,1,0,1))
        self.gl.addItem(self.est_line); self.gl.addItem(self.pts_item)

        self.history = history
        self.est_buf = np.empty((0,3))
        self._img_q = queue.Queue(1); self._est_q = queue.Queue(); self._pts_q = queue.Queue()
        self._running = True

        # status bar
        self.status = QtWidgets.QLabel(); self.statusBar().addPermanentWidget(self.status)
        self.statusBar().addPermanentWidget(QtWidgets.QLabel("<font color='red'>■</font> траектория  <font color='yellow'>■</font> точки"))
        self._cnt = 0; self._t0 = time.time()

        # video writer (30 сек)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video = cv2.VideoWriter(video_path, fourcc, fps, (self.width(), self.height()))
        self._record_start = time.time(); self._record_len = 50.0

        # refresh timer
        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self._update_gui); self.timer.start(int(1000/fps))

    def update_image(self, img):
        if self._running:
            if self._img_q.full():
                self._img_q.get_nowait()
            self._img_q.put(img.copy())

    def update_pose(self, T):
        if self._running:
            self._est_q.put(np.asarray(T.t))

    def update_points(self, pts):
        if self._running:
            self._pts_q.put(np.asarray(pts))

    def _update_gui(self):
        try:
            img = self._img_q.get_nowait()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.rot90(img, -1)
            self.img_item.setImage(img, autoLevels=False)
        except queue.Empty:
            pass

        while not self._est_q.empty():
            p = self._est_q.get(); self.est_buf = np.vstack([self.est_buf, p])
        if self.est_buf.shape[0] >= 2:
            self.est_line.setData(pos=self.est_buf)
            cx, cy, cz = self.est_buf[-1]
            self.gl.setCameraPosition(pos=QtGui.QVector3D(cx, cy, cz+3))

        try:
            pts = self._pts_q.get_nowait(); self.pts_item.setData(pos=pts)
        except queue.Empty:
            pass

        self._cnt += 1; now = time.time()
        if now - self._t0 > 0.5:
            fps_val = self._cnt / (now - self._t0); self.status.setText(f"FPS {fps_val:.1f}"); self._cnt = 0; self._t0 = now

        # record video (first 30 s)
        if self.video.isOpened() and (now - self._record_start) < self._record_len:
            qimg = self.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGB888)
            w, h = qimg.width(), qimg.height()
            ptr = qimg.bits(); ptr.setsize(h * w * 3)
            frame = np.frombuffer(ptr, np.uint8).reshape(h, w, 3)
            self.video.write(frame[..., ::-1])

    def closeEvent(self, e):
        self._running = False
        if self.video.isOpened():
            self.video.release()
        e.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv); v = SimpleViewer(); v.show()
    def demo():
        cap = cv2.VideoCapture(0); t = 0.0
        while v._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            v.update_image(frame)
            v.update_pose(type('P', (), {'t': np.array([np.cos(t), np.sin(t), 0.1*t])})())
            v.update_points(np.random.randn(500, 3)*0.1 + [np.cos(t), np.sin(t), 0.1*t])
            t += 0.05; time.sleep(1/30)
        cap.release()
    threading.Thread(target=demo, daemon=True).start(); sys.exit(app.exec_())

import numpy as np

class MovingAverage:

    def __init__(self, window_size):

        self.window_size_ = window_size
        self.markers_ = []
        self.average_ = 0


    def addMarkers(self, points):

        if len(self.markers_) < self.window_size_:
            self.markers_.append(points)

        else:
            self.markers_.pop(0)
            self.markers_.append(points)

    def getAverage(self):
        markers = self.markers_
        markers = np.array(markers)
        sum = np.sum(markers, axis = 0)
        self.average_ = (sum / len(self.markers_))
        if len(self.markers_) < self.window_size_:
            return markers[-1]
        else:
            return self.average_

    def getListLength(self):
        l = len(self.markers_)
        return l






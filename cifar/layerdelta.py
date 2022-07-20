import tensorflow as tf
import numpy

class Conv2dLayerDelta:
    type = tf.keras.layers.Conv2D
    lname = "default"
    layerindex = 0
    height = 0
    width = 0
    channels = 0
    filters = 0
    deltaarray = []

    def __init__(self, layername, kheight, kwidth, channel, filter):
        self.lname = layername
        self.height = kheight
        self.width = kwidth
        self.channels = channel
        self.filters = filter

    def AddArray(self, data):
        self.deltaarray.append(data)

    @property
    def name(self):
        return self.lname

class floatdelta:
    valueone = 0.0
    valuetwo = 0.0
    deltaval = 0.0
    
    def __init__(self, valone, valtwo, delval):
        self.valueone = valone
        self.valuetwo = valtwo
        self.deltaval = delval
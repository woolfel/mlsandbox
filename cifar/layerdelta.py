import tensorflow as tf
import numpy
import floatdelta

class Conv2dLayerDelta:
    type = tf.keras.layers.Conv2D
    layername = "default"
    layerindex = 0
    height = 0
    width = 0
    channels = 0
    filters = 0
    deltaarray = []
    diffcount = 0
    paramcount = 0
    deltasum = 0.0

    def __init__(self, layername, kheight, kwidth, channel, filter):
        self.layername = layername
        self.height = kheight
        self.width = kwidth
        self.channels = channel
        self.filters = filter

    def AddArray(self, data):
        self.deltaarray.append(data)

    def incrementDeltaCount(self):
        self.diffcount +=1
    
    def incrementParamCount(self):
        self.paramcount +=1

    def AddDelta(self, dval):
        self.deltasum += dval

    @property
    def name(self):
        return self.layername


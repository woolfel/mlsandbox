import tensorflow as tf
import numpy
import floatdelta
from marshmallow import Schema, fields

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
    biasarray = []
    biasdiffcount = 0
    biasdeltasum = 0.0

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
    
    def incrementBiasDeltaCount(self):
        self.biasdiffcount +=1
    
    def AddBiasDelta(self, dval):
        self.biasdeltasum += dval

    @property
    def name(self):
        return self.layername

class Conv2dLayerDeltaSchema(Schema):
    type = fields.Str()
    layername = fields.Str()
    layerindex = fields.Integer()
    height = fields.Integer()
    width = fields.Integer()
    channels = fields.Integer()
    filters = fields.Integer()
    deltaarray = fields.List(fields.List(fields.List(fields.List(fields.List(fields.Nested(floatdelta.FloatDeltaSchema))))))
    diffcount = fields.Integer()
    paramcount = fields.Integer()
    deltasum = fields.Float()
    biasarray = fields.List(fields.Nested(floatdelta.FloatDeltaSchema))
    biasdiffcount = fields.Integer()
    biasdeltasum = fields.Float()
import tensorflow as tf
import numpy
import floatdelta
from marshmallow import Schema, fields

class Conv2dLayerDelta:

    def __init__(self, layerindex, layername, kheight, kwidth, channel, filter):
        self.index = layerindex
        self.layername = layername
        self.height = kheight
        self.width = kwidth
        self.channels = channel
        self.filters = filter
        self.type = 'tf.keras.layers.Conv2D'
        self.layerindex = 0
        self.deltaarray = []
        self.diffcount = 0
        self.paramcount = 0
        self.deltasum = 0.0
        self.deltamax = 0.0
        self.biasarray = []
        self.biasparamcount = 0
        self.biasdiffcount = 0
        self.biasdeltasum = 0.0
        self.biasdeltamax = 0.0

    def AddArray(self, data):
        self.deltaarray.append(data)
        #print('delta array len: ', len(self.deltaarray))

    def incrementDeltaCount(self):
        self.diffcount +=1
    
    def incrementParamCount(self):
        self.paramcount +=1

    def AddDelta(self, dval):
        self.deltasum += dval
        if dval > self.deltamax:
            self.deltamax = dval
    
    def incrementBiasDeltaCount(self):
        self.biasdiffcount +=1
    
    def AddBiasDelta(self, dval):
        self.biasdeltasum += dval
        if dval > self.biasdeltamax:
            self.biasdeltamax = dval

    def incrementBiasParamCount(self):
        self.biasparamcount +=1

    @property
    def name(self):
        return self.layername

class Conv2dLayerDeltaSchema(Schema):
    type = fields.Str()
    index = fields.Integer()
    layername = fields.Str()
    height = fields.Integer()
    width = fields.Integer()
    channels = fields.Integer()
    filters = fields.Integer()
    deltaarray = fields.List(fields.List(fields.List(fields.List(fields.Nested(floatdelta.FloatDeltaSchema)))))
    diffcount = fields.Integer()
    paramcount = fields.Integer()
    deltasum = fields.Float()
    deltamax = fields.Float()
    biasarray = fields.List(fields.Nested(floatdelta.FloatDeltaSchema))
    biasdiffcount = fields.Integer()
    biasdeltasum = fields.Float()
    biasparamcount = fields.Integer()
    biasdeltamax = fields.Float()

class DenseLayerDelta:
    def __init__(self, layerindex, name) -> None:
        self.layername = name
        self.index = layerindex
        self.type = 'tf.keras.layers.Dense'
        self.deltaarray = []
        self.diffcount = 0
        self.paramcount = 0
        self.deltasum = 0.0
        self.deltamax = 0.0
        self.biasarray = []
        self.biasdiffcount = 0
        self.biasdeltasum = 0.0
        self.biasparamcount = 0
        self.biasdeltamax = 0.0

    def AddArray(self, data):
        self.deltaarray.append(data)
        #print('delta array len: ', len(self.deltaarray)) v

    def incrementParamCount(self):
        self.paramcount +=1
    
    def incrementDeltaCount(self):
        self.diffcount +=1

    def AddDelta(self, dval):
        self.deltasum += dval
        if dval > self.deltamax:
            self.deltamax = dval

    def AddBiasDelta(self, dval):
        self.biasdeltasum += dval
        if dval > self.biasdeltamax:
            self.biasdeltamax = dval

    def incrementBiasDeltaCount(self):
        self.biasdiffcount +=1

    def incrementBiasParamCount(self):
        self.biasparamcount +=1


class DenseLayerDeltaSchema(Schema):
    index = fields.Integer()
    layername = fields.Str()
    index = fields.Integer()
    diffcount = fields.Integer()
    paramcount = fields.Integer()
    biasdiffcount = fields.Integer()
    biasdeltasum = fields.Integer()
    deltaarray = fields.List(fields.List(fields.List(fields.Nested(floatdelta.FloatDeltaSchema))))
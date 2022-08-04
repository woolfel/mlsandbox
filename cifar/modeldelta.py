import tensorflow as tf
import layerdelta
from marshmallow import Schema, fields

class ModelDelta:

    def __init__(self, name, filename1, filename2):
        self.modelname = name
        self.modelfile1 = filename1
        self.modelfile2 = filename2
        self.layerdeltas = []

    @property
    def name(self):
        return self.modelname
    
    def addLayerDelta(self, delta):
        self.layerdeltas.append(delta)

class ModelDeltaSchema(Schema):
    modelname = fields.Str()
    modelfile1 = fields.Str()
    modelfile2 = fields.Str()
    layerdeltas = fields.List(fields.Nested(layerdelta.Conv2dLayerDeltaSchema))
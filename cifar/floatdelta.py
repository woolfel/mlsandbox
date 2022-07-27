import tensorflow as tf
from marshmallow import Schema, fields

class FloatDelta:
    valueone = 0.0
    valuetwo = 0.0
    deltaval = 0.0
    
    def __init__(self, valone, valtwo, delval):
        self.valueone = valone
        self.valuetwo = valtwo
        self.deltaval = delval

class FloatDeltaSchema(Schema):
    valueone = fields.Float()
    valuetwo = fields.Float()
    deltaval = fields.Float()
import tensorflow as tf
from marshmallow import Schema, fields

class FloatDelta:
    
    def __init__(self, valone, valtwo, delval):
        self.valueone = valone
        self.valuetwo = valtwo
        self.deltaval = delval

class FloatDeltaSchema(Schema):
    valueone = fields.Float()
    valuetwo = fields.Float()
    deltaval = fields.Float()
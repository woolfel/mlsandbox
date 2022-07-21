import tensorflow as tf

class FloatDelta:
    valueone = 0.0
    valuetwo = 0.0
    deltaval = 0.0
    
    def __init__(self, valone, valtwo, delval):
        self.valueone = valone
        self.valuetwo = valtwo
        self.deltaval = delval
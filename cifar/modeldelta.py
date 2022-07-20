import tensorflow as tf
import layerdelta

class ModelDelta:

    def __init__(self, name, filename1, filename2):
        self.modelname = name
        self.modelfile1 = filename1
        self.modelfile2 = filename2
        self.layerDeltas = []

    @property
    def name(self):
        return self.modelname
    
    def addLayerDelta(self, delta):
        self.layerDeltas.append(delta)
        
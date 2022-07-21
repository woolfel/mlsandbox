from datetime import datetime
import json
import re
import arrow
import modeldelta
import layerdelta

from datetime import datetime
from decimal import Decimal, ROUND_UP

""" EncoderExtension  for writing the model delta data to JSON format """
class EncoderExtension(json.encoder):
    # overload the method default
    def default(self, obj):

        if isinstance(obj, datetime):
            return arrow.get(obj).isoformat()

        elif isinstance(obj, modeldelta.ModelDelta):
            return json.JSONEncoder().encode({
                "modelname":obj.modelname,
                "modelfile1":obj.modelfile1,
                "modelfile2":obj.modelfile2,
                "layerdeltas":obj.layerdeltas})
        elif isinstance(obj, layerdelta.Conv2dLayerDelta):
            return json.JSONEncoder().encode({})
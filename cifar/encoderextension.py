from datetime import datetime
import json
import re
import arrow
import modeldelta
import layerdelta

from datetime import datetime
from decimal import Decimal, ROUND_UP

""" EncoderExtension  for writing the model delta data to JSON format """
class EncoderExtension(json.JSONEncoder):
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
            return json.JSONEncoder().encode({
                "type":obj.type,
                "layername":obj.layername,
                "layerindex":obj.layerindex,
                "height":obj.height,
                "width":obj.width,
                "channels":obj.channels,
                "filters":obj.filters,
                "deltaarray":obj.deltaarray,
                "diffcount":obj.diffcount,
                "paramcount":obj.paramcount,
                "deltasum":obj.deltasum})

        return json.JSONEncoder.default(self,obj)

def json_encode(data):
    return EncoderExtension().encode(data)
# Marshmallow doesn't handle an array of objects, so I'm forced to write my own
# utility. If python 3 supported better type system and generics, I wouldn't
# need to write my own utility to write objects to JSON

import modeldelta
import layerdelta

# entry function for saving modelDelta as JSON
def writeDiffModel(modeldelta):
    print('  Write the Model Delta to JSON format')
    modelstring = '{"modelname":"' + modeldelta.modelname 
    modelstring += '","modelfile1":"' + modeldelta.modelfile1 
    modelstring += '","modelfile2":"' + modeldelta.modelfile2
    modelstring += '","layerdeltas":['
    # iterate over the layer diffs
    for l in range(len(modeldelta.layerdeltas)):
        ldelta = modeldelta.layerdeltas[l]
        if isinstance(ldelta, layerdelta.Conv2dLayerDelta):
            modelstring += convertConv2D(ldelta)
        elif isinstance(ldelta, layerdelta.DenseLayerDelta):
            modelstring += convertDense(ldelta)

    modelstring += "]}"

    return modelstring

# function for converting Conv2dLayerDelta object
def convertConv2D(conv2ddelta):
    print('    convert Conv2D layer delta')
    cvdstring = '{"index":' + str(conv2ddelta.index)
    cvdstring += ',"layername":"' + conv2ddelta.layername
    cvdstring += '","type":"' + conv2ddelta.type
    cvdstring += '","height":"' + str(conv2ddelta.height)
    cvdstring += '","width":"' + str(conv2ddelta.width)
    cvdstring += '","channels":"' + str(conv2ddelta.channels)
    cvdstring += '","filters":"' + str(conv2ddelta.filters)
    cvdstring += '","diffcount":"' + str(conv2ddelta.diffcount)
    cvdstring += '","paramcount":"' + str(conv2ddelta.paramcount)
    cvdstring += '","deltasum":"' + str(conv2ddelta.deltasum)
    cvdstring += '","deltamax":"' + str(conv2ddelta.deltamax)
    cvdstring += '","biasdiffcount":"' + str(conv2ddelta.biasdiffcount)
    cvdstring += '","biasdeltasum":"' + str(conv2ddelta.biasdeltasum)
    cvdstring += '","biasparamcount":"' + str(conv2ddelta.biasparamcount)
    cvdstring += '","biasdeltamax":"' + str(conv2ddelta.biasdeltamax)
    cvdstring += '","deltaarray":['
    # the delta array is nested array of arrays

    cvdstring += ']'
    cvdstring += '}'
    return cvdstring

# function for converting DenseLayerDelta object
def convertDense(densedelta):
    print('    convert Dense layer delta')
    dsdstring = '{"index":' + str(densedelta.index)
    dsdstring += ',"layername":"' + densedelta.layername
    dsdstring += '","type":"' + densedelta.type
    dsdstring += '","diffcount":"' + str(densedelta.diffcount)
    dsdstring += '","paramcount":"' + str(densedelta.paramcount)
    dsdstring += '","deltasum":"' + str(densedelta.deltasum)
    dsdstring += '","deltamax":"' + str(densedelta.deltamax)
    dsdstring += '","biasdiffcount":"' + str(densedelta.biasdiffcount)
    dsdstring += '","biasdeltasum":"' + str(densedelta.biasdeltasum)
    dsdstring += '","biasparamcount":"' + str(densedelta.biasparamcount)
    dsdstring += '","biasdeltamax":"' + str(densedelta.biasdeltamax)
    dsdstring += '","deltaarray":['
    dsdstring += ']'
    dsdstring += '}'
    return dsdstring

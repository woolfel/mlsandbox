from textwrap import indent
from marshmallow import Schema
from numpy import ndarray
from sqlalchemy import null
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import sys
import floatdelta
import layerdelta
import modeldelta as md
import layerdelta
import json

print(tf.__version__)

model_diff = null

# main function to 
def main():
    args = sys.argv[0:]

    if len(sys.argv) == 1:
        print('Example usage:')
        print('               python checkpoint_diff.py ./mymodel1.hdf5 ./mymodel2.hdf5 result_output.json')
    else:
        print('Loading with args:  ', args)
        model1 = tf.keras.models.load_model(args[1])
        model2 = tf.keras.models.load_model(args[2])
        outputfile = args[3]
        model_diff = md.ModelDelta(model1.name,args[1], args[2])
        print(model_diff.name)
        print(model_diff.modelfile1)
        print(model_diff.modelfile2)
        print(' deltas=', model_diff.layerdeltas)
        compare(model_diff, model1, model2)
        print(' saving diff to file: ', outputfile)
        modelschema = md.ModelDeltaSchema()
        jsonresult = modelschema.dumps(model_diff)
        #print(jsonresult)
        diffout = open(outputfile,"x")
        diffout.write(jsonresult)
        diffout.close()

""" diff is the entry point for comparing the weights of two checkpoint models
 For now diff will ignore the layer if it's the Input for the model. The reason
 for skipping the input layer is to reduce noise. The assumption might be
 wrong and the filters in the input layer could be significant.
"""
def compare(diff, model1, model2):
    print(' ---------- comparing the checkpoints ----------')
    # iterate over a sequential model and do diff
    for index, item in enumerate(model1.layers):
        m1layer = item
        m2layer = model2.layers[index]
        # switch statement to handle each layer type properly
        if isinstance(item, tf.keras.layers.Conv2D):
            print('Conv2D layer')
            diffConv2D(diff, index, m1layer.weights, m2layer.weights)
        elif isinstance(item, tf.keras.layers.MaxPooling2D):
            print('MaxPooling2D layer')
            diffMaxPool(diff, index, m1layer, m2layer)
        elif isinstance(item, tf.keras.layers.Flatten):
            diffFlatten(diff,index, m1layer, m2layer)
        elif isinstance(item, tf.keras.layers.Dropout):
            diffDropout(diff, index, m1layer, m2layer)
        elif isinstance(item, tf.keras.layers.Dense):
            diffDense(diff, index, m1layer, m2layer)
        elif isinstance(item, tf.keras.layers.Conv3D):
            print('Conv3D layer')
        else:
            print(item.__class__.__name__)

        #print(diff)
    print(' --- done with diff')


""" If the layer is not input layer, we compare the weights.
 Layer definition: Conv2D(256, (2, 2), strides=(1,1), activation='relu', name='L2_conv2d')
 Weight shape: shape(2, 2, 256, 256)
 The first 2 number is the kernel (2,2), the third number is channels (aka previous layer filter size),
 forth number is the layer filters. Keras source for third number has input_channel // self.groups
 https://github.com/keras-team/keras/blob/master/keras/layers/convolutional/base_conv.py line 212. 
 The // operator is floor division, which means most of the time the value is divided by default group 1.

 The input is equal to the output from the previous layer
 the last is the output filter. Note the kernel may be different, so the function has to look
 at the shape.

 TODO - for now it's a bunch of nested for loops. Needs to be refactored and clean it up
"""
def diffConv2D(diff, index, weights1, weights2):
    if index > 0:
        # We should always have weights for Conv2D, but check to be safe
        if len(weights1) > 0:
            kheight = weights1[0].shape[0]
            kwidth = weights1[0].shape[1]
            prevchannels = weights1[0].shape[2]
            filters = weights1[0].shape[3]
            print(' kernel height/width=', kheight, kwidth)
            print(' channels=', prevchannels)
            print(' filter =', filters)
            # Conv2D layers weights have kernel and bias. By default bias is true. It is optional
            lydelta = layerdelta.Conv2dLayerDelta(index, weights1[0].name, kheight, kwidth, prevchannels, filters)
            diff.addLayerDelta(lydelta)
            #print(weights1)
            for h in range(1):
                h1 = weights1[h].numpy()
                h2 = weights2[h].numpy()
                # the height array for deltas based on kernel height
                wharray = []
                lydelta.AddArray(wharray)
                wdarray1 = h1[0]
                wdarray2 = h2[0]
                # defensive code to make sure it is an array with len attribute
                if hasattr(wdarray1, "__len__"):
                    wlen = len(wdarray1)
                    # the width array for deltas based on kernel width
                    #wwarray = []
                    #wharray.append(wwarray)
                    for nw in range(wlen):
                        """ this should ndarray of channels """
                        #print(' width iterate: ', nw)
                        chlen1 = len(wdarray1[nw])
                        carray1 = wdarray1[nw]
                        carray2 = wdarray2[nw]
                        charray = []
                        wharray.append(charray)
                        for nc in range(chlen1):
                            # print(' channel iterate: ', nc)
                            farray1 = carray1[nc]
                            farray2 = carray2[nc]
                            wtarray = []
                            charray.append(wtarray)
                            #print(' filter len: ', len(farray1), end=' ')
                            for nf in range(len(farray1)):
                                # the actual weights
                                lydelta.incrementParamCount()
                                wt1 = farray1[nf]
                                wt2 = farray2[nf]
                                delta = abs(wt2 - wt1)
                                lydelta.AddDelta(delta)
                                float_diff = floatdelta.FloatDelta(wt1, wt2, delta)
                                wtarray.append(float_diff)
                                #print(' diff : ', wt1, wt2, delta, end=' ')
                                if delta > 0:
                                    lydelta.incrementDeltaCount()
                    else:
                        #print(wdarray1)
                        print('')
            print(' layer diff count: ', lydelta.diffcount, " - total: ", lydelta.paramcount, " deltaSum: ", lydelta.deltasum)

            # for x in range(len(weights1)):
            #     print('  shape=', weights1[x].shape, '\n')
            #     nw1 = weights1[x].numpy()
            #     for y in range(len(nw1)):
            #         yarr = nw1[y]
            #         inspectArray(yarr,'  ')

            if len(weights1) == 2:
                # bias is just 1 array of floats
                arraylen = weights1[1].shape[0]
                print('  shape =', arraylen)
                bw1 = weights1[1].numpy()
                bw2 = weights2[1].numpy()
                deltas = []
                lydelta.biasarray = deltas
                for ix in range(arraylen):
                    w1 = bw1[ix]
                    w2 = bw2[ix]
                    delta = abs(w1 - w2)
                    float_diff = floatdelta.FloatDelta(w1, w2, delta)
                    deltas.append(float_diff)
                    lydelta.AddBiasDelta(delta)
                    lydelta.incrementBiasParamCount()
                    if delta > 0:
                        lydelta.incrementBiasDeltaCount()
            print(' bias diff count: ', lydelta.biasdiffcount, " - total: ", lydelta.biasparamcount, " deltaSum: ", lydelta.biasdeltasum)

    else:
        print('input layer - no need to diff')

def diffMaxPool(diff, index, layer1, layer2):
    print(" - maxpool size: ", layer1.pool_size)

def inspectArray(narrayobj, sep):
    if hasattr(narrayobj, "__len__"):
        print('[',end='')
        print(len(narrayobj),sep,end='')
        for z in range(len(narrayobj)):
            charray = narrayobj[z]
            inspectArray(charray,'')
        print('] ',end='')

""" Keras dense layer has weights and bias. Depending on the model configuration, the layer
might not have bias.
"""
def diffDense(diff, index, layer1, layer2):
    print(' - calculate diff for dense layer')
    print(layer1.name)
    # #print(layer1.weights)
    shape = layer1.weights[0].shape
    print('  - dense shape: ', shape)
    weights1 = layer1.weights
    weights2 = layer2.weights
    wlen = len(weights1)
    print('  weights len: ', wlen)
    denseDelta = layerdelta.DenseLayerDelta(index, layer1.name)
    diff.addLayerDelta(denseDelta)
    # # dense layer weights has kernel and bias
    # kshape = weights1[0].shape
    # dimen = kshape[0]
    # weights = kshape[1]
    # knarray1 = weights1[0]
    # knarray2 = weights2[0]
    # deltaarray = []
    # denseDelta.AddArray(deltaarray)
    # #print('  weights: ', weights1)
    # print('  kernarray: ', knarray1)
    # for x in range(dimen):
    #     #print(' x: ', x, end=' ')
    #     dimarray1 = knarray1[x]
    #     dimarray2 = knarray2[x]
    #     dimensions = []
    #     deltaarray.append(dimensions)
    #     # defensive code to make sure it's an array
    #     if hasattr(dimarray1, "__len__"):
    #         nestlen = len(dimarray1)
    #         #print(' weights length: ', nestlen)
    #         for y in range(nestlen):
    #             wt1 = dimarray1[y]
    #             wt2 = dimarray2[y]
    #             dval = abs(wt1 - wt2)
    #             fldelta = floatdelta.FloatDelta(wt1, wt2, dval)
    #             dimensions.append(fldelta)
    #             denseDelta.incrementParamCount()
    #             if dval > 0.0:
    #                 denseDelta.incrementDeltaCount()
    # # the bias
    # if len(weights1) > 1:
    #     bsarray1 = weights1[1].numpy
    #     bsarray2 = weights2[1].numpy
    #     print('  bias array: ', bsarray1)

    # for x in range(wlen):
    #     print('  shape=', weights1[x].shape, '\n')
    #     nw1 = weights1[x].numpy()
    #     for y in range(len(nw1)):
    #         yarr = nw1[y]
    #         inspectArray(yarr,'  ')
    #print('  dense delta: ', len(denseDelta.deltaarray), ' diffcount: ', denseDelta.diffcount)

def diffDropout(diff, index, layer1, layer2):
    print(' - calculate diff for dropout')
    print(' layer name: ', layer1.name)

def diffFlatten(diff, index, layer1, layer2):
    print(' - calculate diff for Flatten')
    print('  layer name: ', layer1.name)
    print('  flat weights: ', layer1.weights)
    

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()

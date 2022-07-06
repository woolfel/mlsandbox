from numpy import ndarray
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import sys
import difflib
from deepdiff import DeepDiff

print(tf.__version__)

# main function to 
def main():
    args = sys.argv[0:]

    if len(sys.argv) == 1:
        print('Example usage:')
        print('               python checkpoint_diff.py ./mymodel1.hdf5 ./mymodel2.hdf5')
    else:
        print('Loading with args:  ', args)
        model1 = tf.keras.models.load_model(args[1])
        model2 = tf.keras.models.load_model(args[2])
        #print(model1.summary())
        compare(model1, model2)

def compare(model1, model2):
    print(' ---------- comparing the checkpoints ----------')
    # iterate over a sequential model and do diff
    for index, item in enumerate(model1.layers):
        m1layer = item
        m2layer = model2.layers[index]
        # switch statement to handle each layer type properly
        if isinstance(item, tf.keras.layers.Conv2D):
            print('Conv2D layer')
            diffConv2D(index, m1layer.weights, m2layer.weights)
        elif isinstance(item, tf.keras.layers.MaxPooling2D):
            print('MaxPooling2D layer')
            diffMaxPool(m1layer, m2layer)
        elif isinstance(item, tf.keras.layers.Flatten):
            print('Flatten layer')
        elif isinstance(item, tf.keras.layers.Dropout):
            print('Dropout layer')
        elif isinstance(item, tf.keras.layers.Dense):
            print('Dense layer')
        elif isinstance(item, tf.keras.layers.Conv3D):
            print('Conv3D layer')
        else:
            print(item.__class__.__name__)

        #print(diff)
    print(' --- done with diff')

# diff is the entry point for comparing the weights of two checkpoint models
# For now diff will ignore the layer if it's the Input for the model. The reason
# for skipping the input layer is to reduce noise. The assumption might be
# wrong and the filters in the input layer makes a significant difference.
#
# If the layer is a hidden layer (ie not input)
# Conv2D(256, (2, 2), strides=(1,1), activation='relu', name='L2_conv2d')
# shape(2, 2, 256, 256)
# The first 2 number is the kernel (2,2), the input is equal to the output from the previous layer
# the last is the output filter
# 
def diffConv2D(index, weights1, weights2):
    if index > 0:
        # We should always have weights for Conv2D, but check to be safe
        if len(weights1) > 0:
            for x in range(len(weights1)):
                print('  shape=', weights1[x].shape, '\n')
                nw1 = weights1[x].numpy()
                for y in range(len(nw1)):
                    yarr = nw1[y]
                    inspectArray(yarr,'  ')
                    
                print('\n')
            #print(' ----- weights 2 ----- ')
            #print(weights2[0].numpy)
            #print(weights2.__class__.__name__)
            for x in range(len(weights2)):
                #print('  ', weights2[x], '\n')
                nw2 = weights2[x].numpy()
    else:
        print('input layer - no need to diff')

def diffMaxPool(layer1, layer2):
    print(layer1)

def inspectArray(narrayobj, sep):
    if hasattr(narrayobj, "__len__"):
        print('[',end='')
        print(len(narrayobj),sep,end='')
        for z in range(len(narrayobj)):
            charray = narrayobj[z]
            inspectArray(charray,'')
        print('] ',end='')

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()

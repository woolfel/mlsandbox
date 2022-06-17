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
        print(m1layer.__class__.__name__, m1layer.name)
        diff(m1layer.weights, m2layer.weights)
        #print(diff)
    print(' --- done with diff')

def diff(weights1, weights2):
    if len(weights1) > 0:
        #print(' ----- weights 1 ----- ')
        #print(weights1.__class__.__name__)
        #print(weights1[0].numpy)
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
        print(' - no weights')

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

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
        diff(m1layer.weights, m2layer.weights)
        #print(diff)
    print(' --- done with diff')

def diff(weights1, weights2):
    print(' ----- weights ----- ')
    if len(weights1) > 0:
        deepdiff = DeepDiff(weights1[0].numpy, weights2[0].numpy)
        #print(deepdiff.items)
        print(weights1[0].numpy)
        print(weights2[0].numpy)

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()

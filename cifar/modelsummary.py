import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import sys

print(tf.__version__)

# main function to 
def main():
    args = sys.argv[0:]

    if len(sys.argv) == 1:
        print('Example usage:')
        print('               python modelsummary.py ./mymodel.hdf5')
    else:
        print('Loading with args:  ', args)
        model = tf.keras.models.load_model(args[1])
        print(model.summary())

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()

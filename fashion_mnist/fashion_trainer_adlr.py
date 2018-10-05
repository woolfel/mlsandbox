# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import pydot
import graphviz
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing.spawn import prepare

batch_size=500
num_classes = 10
epochs = 15
epoch2 = 20
steps = 60000 / batch_size
second_lr = 0.00044444
    
# input image dimensions
img_rows, img_cols = 28, 28
# input shape is the rows, cols and 1 channel for fasion MNIST
# if the images were color, the channel would be 3
input_shape = (img_rows, img_cols, 1)

def createDataGenerator():
    # Create instance of DataGenerator and use shift, h flip and zoom
    datagen = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=.115,
        height_shift_range=.115,
        horizontal_flip=True,
        zoom_range=.021)
    return datagen

def createDataGenerator2():
    # Create instance of DataGenerator that shifts the image side ways 1 pixel
    # note for larger images, you'd want a higher value
    datagen2 = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=.05)
    return datagen2


def prepareTrainingData():
    # the data, split between train and test sets
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # Prepare the training and test data with reshape
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)

def createModel():
    # Setup the Keras model
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(196, kernel_size=(2, 2), strides=(1,1), activation='relu', input_shape=input_shape))
    model.add(keras.layers.Conv2D(256, (2, 2), strides=(1,1), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(keras.layers.Conv2D(256, (1, 1), activation='relu'))
    model.add(keras.layers.Conv2D(512, (2, 2), activation='relu'))
    model.add(keras.layers.Dropout(0.1589))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5683))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def calculateLR(epoch):
    drop = 0.000005
    lr = second_lr - (epoch * drop)
    print('adjusted rate: ', lr)
    return lr
    
def main():

    print(tf.__version__)

    datagen = createDataGenerator()
    datagen2 = createDataGenerator2()
    (x_train, y_train),(x_test, y_test) = prepareTrainingData()
    model = createModel()
    
    datagen.fit(x_train)
    print('Learning rate: ', keras.backend.eval(model.optimizer.lr))
    start_time = time.time()
    # calling fit_geneator will use the DataGenerator to modify the images in real-time
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=steps,
                        epochs=epochs,
                        validation_data=(x_test, y_test))
    # Set the learning rate to reduce the chance of over fitting
    model.optimizer = keras.optimizers.Adam(second_lr)
    #model.loss = keras.losses.sparse_categorical_crossentropy
    model.loss = keras.losses.mean_squared_logarithmic_error
    print('Learning rate: ', keras.backend.eval(model.optimizer.lr))
    print('Loss: ', model.loss)
    
    clr = keras.callbacks.LearningRateScheduler(calculateLR)
    # Retrain on the original image set to increase test accuracy
    # Note: if the batch size is too small the loss starts to increase 
    #       and the test accuracy degrades. This suggests the model is
    #       over fitting. On a Gigabyte 1070 card with 8G, the max
    #       batch size is about 1000. 
    history2 = model.fit(x_train, 
              y_train,
              batch_size=800,
              epochs=epoch2,
              verbose=1,
              callbacks=[clr],
              validation_data=(x_test, y_test))

    end_time = time.time()
    # Merge the results of both history
    for k, v in history2.history.items():
    	for item in v:
    		history.history[k].append(item)
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


    # A final test to evaluate the model
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Elapsed Time: %0.4f seconds' % (end_time - start_time))
    print('Elapsed Time: %0.4f minutes' % ((end_time - start_time)/60))
    print(model.summary())
    # save the model
    model.save('./saved_model.h5')

if __name__ == '__main__':
    main()
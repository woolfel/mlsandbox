# TensorFlow and tf.keras
from turtle import shape
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os

print(tf.__version__)

# The filename format has the epoch number + accuracy + loss in HDF5 format
# the reason for using HDF5 format is cross platform compatibility and make it easier to load in other languages
checkpoint_path = "training/weights.{epoch:02d}-{val_accuracy:.2f}-{val_loss:.2f}.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# the benchmark loads the MNIST dataset from tensorflow datasets
# a possible alternative is fashion MNIST, which should require more power
(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

# you can change the batch size to see how it performs. Larger batch size will stress GPU more
batch_size = 256
epoch_count = 10

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(batch_size)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

InputShape = tf.keras.Input(shape=(32,32,3))

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(256, kernel_size=(2, 2), strides=(1,1), activation='relu', input_shape=(32,32,3)),
  tf.keras.layers.Conv2D(256, (2, 2), strides=(1,1), activation='relu', name='L2_conv2d'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='L3_MaxP'),
  tf.keras.layers.Conv2D(256, (1, 1), activation='relu', name='L4_conv2d'),
  tf.keras.layers.Conv2D(512, (2, 2), activation='relu', name='L5_conv2d'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, name='L6_MaxP'),
  tf.keras.layers.Conv2D(256, (1, 1), activation='relu', name='L7_conv2d'),
  tf.keras.layers.Conv2D(512, (2, 2), activation='relu', name='L8_conv2d'),
  tf.keras.layers.Dropout(0.1589, name='L9_Drop'),
  tf.keras.layers.Flatten(name='L10_flat'),
  tf.keras.layers.Dense(128, activation='relu', name='L11_Dense'),
  tf.keras.layers.Dropout(0.5683, name='L12_Drop'),
  tf.keras.layers.Dense(10, activation='softmax', name='Dense_output')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=1,
                                                 monitor='accuracy',
                                                 save_freq='epoch')

#model.save_weights(checkpoint_path.format(epoch=0))

start_time = time.time()

model.fit(
    ds_train,
    epochs=epoch_count,
    validation_data=ds_test,
    callbacks=[cp_callback]
)
end_time = time.time()

# A final test to evaluate the model
print('Test loss:', model.loss)
print('Elapsed Time: %0.4f seconds' % (end_time - start_time))
print('Elapsed Time: %0.4f minutes' % ((end_time - start_time)/60))
print(model.summary())
# TensorFlow and tf.keras
from turtle import shape
import tensorflow as tf
import tensorflow_datasets as tfds
import time

print(tf.__version__)

# the benchmark loads the MNIST dataset from tensorflow datasets
# a possible alternative is fashion MNIST, which should require more power
(ds_train, ds_test), ds_info = tfds.load(
    'cifar100',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

# you can change the batch size to see how it performs. Larger batch size will stress GPU more
batch_size = 32

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

model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=InputShape, classes=100)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

start_time = time.time()
 # changing the epochs count doesn't affect total memory used, but it does improve accuracy
model.fit(
    ds_train,
    epochs=15,
    validation_data=ds_test,
)
end_time = time.time()

# A final test to evaluate the model
print('Test loss:', model.loss)
print('Elapsed Time: %0.4f seconds' % (end_time - start_time))
print('Elapsed Time: %0.4f minutes' % ((end_time - start_time)/60))
print(model.summary())
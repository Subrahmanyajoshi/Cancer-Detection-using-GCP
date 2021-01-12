import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def keras_estimator(imag_shape=(None, 300, 300, 3)):
    model = Sequential([
        layers.Conv2D(filters=32, kernel_size=((3, 3)), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(filters=64, kernel_size=((3, 3)), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(filters=128, kernel_size=((3, 3)), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(filters=256, kernel_size=((3, 3)), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.build(input_shape=imag_shape)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # return tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir,
    #                                              config=config)
    return model

"""
def input_fn(train_dir, val_dir, batch_size, mode):

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    image_dataset = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        image_dataset = train_datagen.flow_from_directory(
                        train_dir,
                        target_size=(300, 300),
                        batch_size=10,
                        class_mode='binary')
    if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
        image_dataset = test_datagen.flow_from_directory(
                        val_dir,
                        target_size=(300, 300),
                        batch_size=10,
                        class_mode='binary')

    return image_dataset.next()
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.logging.set_verbosity(tf.logging.INFO)


def keras_estimator():
    model = Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

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

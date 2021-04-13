from abc import ABC, abstractmethod
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


class Model(ABC):
    """ An abstract class which outlines the structure of model classes. All classes which inherit from this class
        should implement the abstract methods """

    @abstractmethod
    def build(self):
        """ This method does not require implementation inside abstract class"""
        ...


class CNNModel(Model):

    def __init__(self, img_shape: Optional[Tuple] = (None, 300, 300, 3)):
        """ Init method
        Args:
            img_shape (Optional[Tuple]): shape of input image
        """
        self.img_shape = img_shape

    def build(self):
        """ Creates a cnn model, compiles it and returns it
        Args:
        Returns:
            Built model
        """
        model = Sequential([
            layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),  # Dropout for regularization
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        model.build(input_shape=self.img_shape)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

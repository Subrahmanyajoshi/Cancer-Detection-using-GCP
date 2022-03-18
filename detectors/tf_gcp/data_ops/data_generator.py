import os

import numpy as np
from cv2 import imread, resize
from google.cloud.storage import Bucket
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):

    def __init__(self, image_filenames: np.ndarray, labels: np.ndarray, batch_size: int, dest_dir: str,
                 bucket: Bucket, image_shape: tuple):
        """ Init Method
        Args:
            image_filenames (np.array): List of image filenames
            labels (np.array): labels associated with filenames
            batch_size (int): batch size of model
            dest_dir (str): directory where actual images are located
            bucket (Bucket): Gcs bucket name
        """
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.dest_dir = dest_dir
        self.bucket = bucket
        self.img_width = image_shape[0]
        self.img_height = image_shape[1]
        self.img_channels = image_shape[2]

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx: int):
        """ Creates a batch of images and associated labels and returns it
        Args:
            idx (int): index from which to start the batch
        Returns:
            A batch of images and labels associated with it
        """
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        # Read Images and crate a list from it
        images = [imread(os.path.join(self.dest_dir + str(file_name))) for file_name in batch_x]

        # the following code downloads images one by one from google storage every time this generator
        # is called. it works but is a bit slow.
        """
        images = []
        for file_name in batch_x:
            image_path = os.path.join(self.dest_dir + str(file_name))
            image_blob = self.bucket.get_blob(image_path)
            image_blob.download_to_filename(file_name)
            images.append(imread(file_name))
            os.remove(file_name)
        """
        images = np.array([resize(img, (self.img_width, self.img_height))for img in images]) / 255.0
        labels = np.array(batch_y)
        return images, labels

import os
from google.cloud import storage

import numpy as np
from cv2 import imread, resize
from tensorflow import keras


class MyCustomGenerator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size, dest_dir, bucket):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.dest_dir = dest_dir
        
        client = storage.Client()
        self.bucket = client.get_bucket(bucket)

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

#         print(self.dest_dir)
#         print(os.path.join(self.dest_dir + str(batch_x[0])))
#         images = [imread(os.path.join(self.dest_dir + str(file_name))) for file_name in batch_x]
#         print(images)
        
        images = []
        for file_name in batch_x:
            image_path = os.path.join(self.dest_dir + str(file_name))
            image_blob = self.bucket.get_blob(image_path)
            image_blob.download_to_filename(file_name)
            images.append(imread(file_name))
            os.remove(file_name)
            
#         raise ValueError()
#         print(f'index: {idx}, batch size: {self.batch_size}, batch_x: {batch_x}')

        images = np.array([resize(img, (300, 300))for img in images]) / 255.0
        labels = np.array(batch_y)
        return images, labels

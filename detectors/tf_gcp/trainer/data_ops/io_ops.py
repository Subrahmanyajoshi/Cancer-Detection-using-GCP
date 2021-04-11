import abc
import os

import numpy as np
from io import BytesIO
from tensorflow.python.lib.io import file_io

from detectors.common import SystemOps
from detectors.tf_gcp.trainer.task import BucketOps


class IO(abc.ABC):
    X_TRAIN = 'X_train_filenames.npy'
    Y_TRAIN = 'y_train.npy'
    X_VAL = 'X_val_filenames.npy'
    Y_VAL = 'y_val.npy'

    def __init__(self, input_dir: str, bucket=None):
        self.X_train_filenames = os.path.join(input_dir, 'train', IO.X_TRAIN)
        self.y_train = os.path.join(input_dir, 'train', IO.Y_TRAIN)
        self.X_val_filenames = os.path.join(input_dir, 'val', IO.X_VAL)
        self.y_val = os.path.join(input_dir, 'val', IO.Y_VAL)
        self.bucket = bucket

    @abc.abstractmethod
    def load(self):
        ...

    @abc.abstractmethod
    def write(self, src_path: str, dest_path: str):
        ...


class LocalIO(IO):

    def __init__(self, input_dir: str):
        super(LocalIO, self).__init__(input_dir=input_dir)

    def load(self):
        X_train_filenames = np.load(self.X_train_filenames)
        y_train = np.load(self.y_train)
        X_val_filenames = np.load(self.X_val_filenames)
        y_val = np.load(self.y_val)
        return X_train_filenames, y_train, X_val_filenames, y_val

    def write(self, src_path: str, dest_path: str):
        SystemOps.move(src_path=src_path, dst_path=dest_path)


class CloudIO(IO):

    def __init__(self, input_dir: str, bucket):
        super(CloudIO, self).__init__(input_dir=input_dir, bucket=bucket)

    @staticmethod
    def load_npy(file_name: str):
        file = BytesIO(file_io.read_file_to_string(file_name, binary_mode=True))
        np_data = np.load(file)
        return np_data

    def upload_file_to_gcs(self, src_path: str, dest_path: str):
        blob = self.bucket.blob(dest_path)
        blob.upload_from_filename(src_path)

    def copy_directory_to_gcs(self, local_path: str, gcs_path: str):
        for local_file in os.listdir(local_path):
            l_file = os.path.join(local_path, local_file)
            if not os.path.isfile(l_file):
                continue
            remote_path = os.path.join(gcs_path, local_file)
            self.upload_file_to_gcs(src_path=l_file, dest_path=remote_path)

    def load(self):
        X_train_filenames = CloudIO.load_npy(file_name=self.X_train_filenames)
        y_train = CloudIO.load_npy(file_name=self.y_train)
        X_val_filenames = CloudIO.load_npy(file_name=self.X_val_filenames)
        y_val = CloudIO.load_npy(file_name=self.y_val)
        return X_train_filenames, y_train, X_val_filenames, y_val

    def write(self, src_path: str, dest_path: str):
        if self.bucket is None:
            raise ValueError('Please provide the bucket object to copy file to GCS')
        if not dest_path.startswith('gs://'):
            dest_path = BucketOps.get_gcs_path(bucket_name=self.bucket.name, path=dest_path)
        if os.path.isfile(src_path):
            self.upload_file_to_gcs(src_path=src_path, dest_path=dest_path)
        else:
            self.copy_directory_to_gcs(local_path=src_path, gcs_path=dest_path)

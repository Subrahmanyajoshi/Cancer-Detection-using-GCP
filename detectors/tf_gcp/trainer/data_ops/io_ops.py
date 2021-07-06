import abc
import os
from typing import Optional

import numpy as np
from io import BytesIO

from google.cloud.storage import Bucket
from tensorflow.python.lib.io import file_io

from detectors.common import SystemOps


class IO(abc.ABC):
    """ An abstract class which outlines the structure of Io classes. All classes which inherit from this class
    should implement all of the abstract methods """

    # standard file names
    X_TRAIN = 'X_train_filenames.npy'
    Y_TRAIN = 'y_train.npy'
    X_VAL = 'X_val_filenames.npy'
    Y_VAL = 'y_val.npy'

    def __init__(self, input_dir: str, bucket: Optional[Bucket] = None):
        """ Init method
        Args:
            input_dir (str): Input directory where all data needed to train a model is located
            bucket (Optional[Bucket]): Google Cloud Storage bucket name
        """
        self.X_train_filenames = os.path.join(input_dir, 'train', IO.X_TRAIN)
        self.y_train = os.path.join(input_dir, 'train', IO.Y_TRAIN)
        self.X_val_filenames = os.path.join(input_dir, 'val', IO.X_VAL)
        self.y_val = os.path.join(input_dir, 'val', IO.Y_VAL)
        self.bucket = bucket

    @abc.abstractmethod
    def load(self):
        """ This method does not require implementation inside abstract class"""
        ...

    @abc.abstractmethod
    def write(self, src_path: str, dest_path: str):
        """ This method does not require implementation inside abstract class"""
        ...


class LocalIO(IO):
    """ To perform IO operations in local file system"""

    def __init__(self, input_dir: str):
        """ Init method
        Args:
            input_dir (str): Input directory where all data needed to train a model is located
        """
        super(LocalIO, self).__init__(input_dir=input_dir)

    def load(self):
        """ Load numpy arrays which contain information about file names and labels"""
        X_train_filenames = np.load(self.X_train_filenames)
        y_train = np.load(self.y_train)
        X_val_filenames = np.load(self.X_val_filenames)
        y_val = np.load(self.y_val)
        return X_train_filenames, y_train, X_val_filenames, y_val

    def write(self, src_path: str, dest_path: str):
        SystemOps.move(src_path=src_path, dst_path=dest_path)


class CloudIO(IO):
    """ To perform IO operations from/to Google Cloud Storage"""

    def __init__(self, input_dir: str, bucket: Bucket):
        """ Init method
        Args:
            input_dir (str): GCS directory where all data needed to train a model is located
            bucket (Optional[Bucket]): Google Cloud Storage bucket name
        """
        super(CloudIO, self).__init__(input_dir=input_dir, bucket=bucket)

    @staticmethod
    def load_npy(file_name: str):
        """ loads npy file from Google CLoud Storage
        Args:
            file_name (str): Name of npy file in Google Cloud Storage Bucket
        Returns:
            numpy array containing data loaded from npy file
        """
        file = BytesIO(file_io.read_file_to_string(file_name, binary_mode=True))
        np_data = np.load(file)
        return np_data

    @staticmethod
    def copy_from_gcs(src_path: str, dest_path: str):
        SystemOps.run_command(f'gsutil -m cp -r {src_path} {dest_path}')

    def upload_file_to_gcs(self, src_path: str, dest_path: str):
        """ Uploads file to Google Cloud Storage
        Args:
            src_path (str): path in local file system
            dest_path (str): path in Google Cloud Storage Bucket
        Returns:
        """
        blob = self.bucket.blob(dest_path)
        blob.upload_from_filename(src_path)

    def copy_directory_to_gcs(self, local_path: str, gcs_path: str):
        """ Copies a directory's contents to Google Cloud Storage
        Args:
            local_path (str): path in local file system
            gcs_path (str): path in Google Cloud Storage Bucket
        Returns:
        """
        for local_file in os.listdir(local_path):
            l_file = os.path.join(local_path, local_file)
            if not os.path.isfile(l_file):
                continue

            # Create path to file system inside GCS
            remote_path = os.path.join(gcs_path, local_file)
            self.upload_file_to_gcs(src_path=l_file, dest_path=remote_path)

    def load(self):
        """ Creates filenames and labels numpy arrays"""
        X_train_filenames = CloudIO.load_npy(file_name=self.X_train_filenames)
        y_train = CloudIO.load_npy(file_name=self.y_train)
        X_val_filenames = CloudIO.load_npy(file_name=self.X_val_filenames)
        y_val = CloudIO.load_npy(file_name=self.y_val)
        return X_train_filenames, y_train, X_val_filenames, y_val

    def write(self, src_path: str, dest_path: str, use_system_cmd: bool = True):
        """ Writes files/folders to Google Cloud Storage
        Args:
            src_path (str): path in local file system
            dest_path (str): path in Google Cloud Storage
            use_system_cmd (bool): a boolean switch to inform whether to use GCS modules or system commands to write
                                   to GCS
        Returns:
        """
        if use_system_cmd:
            SystemOps.run_command(f"gsutil mv -r {src_path} {dest_path}")
            return

        if self.bucket is None:
            raise ValueError('Please provide the bucket object to copy file to GCS')

        # Get relative path inside bucket from absolute path of file/directory in Google Cloud Storage
        if dest_path.startswith('gs://'):
            dest_path = dest_path.split(f"{self.bucket.name}/")[1]

        dest_path = os.path.join(dest_path, os.path.basename(src_path))
        if os.path.isfile(src_path):
            self.upload_file_to_gcs(src_path=src_path, dest_path=dest_path)
        else:
            self.copy_directory_to_gcs(local_path=src_path, gcs_path=dest_path)

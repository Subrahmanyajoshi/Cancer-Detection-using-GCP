import os
import shutil

import yaml
from google.cloud import storage


class SystemOps(object):
    """ Performs system operations which are usually achieved using shell commands"""

    @staticmethod
    def check_and_delete(path: str):
        """ Checks if path exists, if it exists then deletes it
        Args:
            path (str): path to be deleted
        """
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path=path)
            elif os.path.isfile(path):
                os.remove(path=path)

    @staticmethod
    def create_dir(path: str):
        """ Creates directory
        Args:
            path (str): path to directory to be created
        Returns:
        """
        os.mkdir(path=path)

    @staticmethod
    def clean_dir(path: str):
        """ Checks if path exists, if it exists then deletes it and re-creates it
        Args:
            path (str): path to directory to be cleaned
        Returns:
        """
        SystemOps.check_and_delete(path=path)
        SystemOps.create_dir(path=path)

    @staticmethod
    def run_command(command: str):
        """ Executes input shell command
        Args:
            command (str): shell command to be executed
        Returns:
        """
        os.system(command)

    @staticmethod
    def move(src_path: str, dst_path: str):
        """ Moves file/folders from source to destination path
        Args:
            src_path (str): source path
            dst_path (str): destination path
        """
        shutil.move(src=src_path, dst=dst_path)


class BucketOps(object):
    """ Class which implements operations associated with Google Cloud Storage buckets"""

    @staticmethod
    def extract_bucket_name(gcs_path: str):
        """ Extracts bucket name from given Google cloud storage path
        Args:
            gcs_path (str): Google Cloud Storage path
        Returns:
            name of the bucket
        """
        if not gcs_path.startswith('gs://'):
            raise ValueError("Input path doesn't seem to be a GCS path")
        path = gcs_path.split('gs://')[0]
        bucket_name = path.split('/')[0]
        return bucket_name

    @staticmethod
    def get_bucket(bucket_name: str):
        """" Creates Bucket object from given bucket name
        Args:
            bucket_name (str): name of bucket
        Returns:
            Gcs bucket object
        """
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        return bucket

    @staticmethod
    def get_gcs_path(bucket_name: str, path: str):
        """ Create absolute GCS path from given path and bucket names
        Args:
            bucket_name (str): Name of bucket
            path (str): Relative path inside Google Cloud Storage Bucket
        Returns:
            Absolute path inside Google Cloud storage"""
        return os.path.join(f"gs://{bucket_name}", path)


class YamlConfig(object):

    @staticmethod
    def load(filepath: str):
        """ Loads yaml file in the specified path
        Args:
            filepath (str): path of yaml file to be loaded
        """
        with open(filepath) as filestream:
            config = yaml.safe_load(filestream)
        return config

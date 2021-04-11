import os
import shutil
from google.cloud import storage


class SystemOps(object):

    @staticmethod
    def check_and_delete(path: str):
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path=path)
            elif os.path.isfile(path):
                os.remove(path=path)

    @staticmethod
    def create_dir(path: str):
        os.mkdir(path=path)

    @staticmethod
    def clean_dir(path: str):
        SystemOps.check_and_delete(path=path)
        SystemOps.create_dir(path=path)

    @staticmethod
    def run_command(command: str):
        os.system(command)

    @staticmethod
    def move(src_path: str, dst_path: str):
        shutil.move(src=src_path, dst=dst_path)


class BucketOps(object):

    @staticmethod
    def extract_bucket_name(gcs_path: str):
        if not gcs_path.startswith('gs://'):
            raise ValueError("Input path doesn't seem to be a GCS path")
        path = gcs_path.split('gs://')[0]
        bucket_name = path.split('/')[0]
        return bucket_name

    @staticmethod
    def get_bucket(bucket_name: str):
        client = storage.Client()
        bucket = client.get_bucket(bucket_name=bucket_name)
        return bucket

    @staticmethod
    def get_gcs_path(bucket_name: str, path: str):
        return os.path.join(f"gs://{bucket_name}", path)

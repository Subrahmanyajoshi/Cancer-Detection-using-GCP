import importlib
import os
import zipfile
from datetime import datetime

import tensorflow as tf

from argparse import Namespace

from detectors.common import BucketOps, SystemOps, YamlConfig
from detectors.tf_gcp.trainer.data_ops.data_generator import DataGenerator
from detectors.tf_gcp.trainer.data_ops.io_ops import CloudIO, LocalIO
from detectors.tf_gcp.trainer.models.models import CNNModel, VGG19Model


class Trainer(object):

    def __init__(self, config: dict):
        """ Init method
        Args:
            config (dict): Dictionary containing configurations
        """
        self.run_type = config.get('train_type', 'unk').strip()
        self.train_params = Namespace(**config.get('train_params'))
        self.model_params = Namespace(**config.get('model_params'))
        self.cp_path = None
        self.csv_path = None
        self.callbacks = self.init_callbacks()
        self.bucket = None
        bucket_name = 'unk'
        if self.train_params.data_dir.startswith('gs://'):
            bucket_name = self.train_params.data_dir.split('gs://')[1].split('/')[0]
        elif self.train_params.output_dir.startswith('gs://'):
            bucket_name = self.train_params.data_dir.split('gs://')[1].split('/')[0]
        if bucket_name != 'unk':
            self.bucket = BucketOps.get_bucket(bucket_name)

    def init_callbacks(self):
        """ Creates callback objects mentioned in configurations """
        callbacks = []
        module = importlib.import_module('tensorflow.keras.callbacks')
        for cb in self.train_params.callbacks:
            if cb == 'CSVLogger':
                self.csv_path, filename = os.path.split(self.train_params.callbacks[cb]['filename'])
                self.train_params.callbacks[cb]['filename'] = filename

            obj = getattr(module, cb)
            callbacks.append(obj(**self.train_params.callbacks[cb]))
        return callbacks

    def clean_up(self):
        """ Deletes temporary directories created while training"""
        SystemOps.check_and_delete('all_images')
        SystemOps.check_and_delete('train_logs.csv')
        SystemOps.check_and_delete('config.yaml')

    def train(self):
        if self.model_params.model == 'CNN':
            model = CNNModel(img_shape=(None,) + eval(self.train_params.image_shape)).build(self.model_params)
        elif self.model_params.model == 'VGG19':
            model = VGG19Model(eval(self.train_params.image_shape)).build(self.model_params)
        else:
            raise NotImplementedError(f"{self.model_params.model} model is currently not supported. "
                                      f"Please choose between CNN and VGG19")
        model.summary()

        if self.bucket is not None:
            io_operator = CloudIO(input_dir=self.train_params.data_dir, bucket=self.bucket)
            SystemOps.run_command(f"gsutil -m cp -r {os.path.join(self.train_params.data_dir, 'all_images.zip')} ./")
            with zipfile.ZipFile('all_images.zip', 'r') as zip_ref:
                zip_ref.extractall('./all_images')
            SystemOps.check_and_delete('all_images.zip')
        else:
            io_operator = LocalIO(input_dir=self.train_params.data_dir)

        X_train_files, y_train, X_val_files, y_val = io_operator.load()
        train_generator = DataGenerator(image_filenames=X_train_files,
                                        labels=y_train,
                                        batch_size=self.train_params.batch_size,
                                        dest_dir='./all_images/',
                                        bucket=self.bucket,
                                        image_shape=eval(self.train_params.image_shape))
        validation_generator = DataGenerator(image_filenames=X_val_files,
                                             labels=y_val,
                                             batch_size=self.train_params.batch_size,
                                             dest_dir='./all_images/',
                                             bucket=self.bucket,
                                             image_shape=eval(self.train_params.image_shape))

        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=self.train_params.num_epochs,
            callbacks=self.callbacks,
            steps_per_epoch=self.train_params.steps_per_epoch,
            workers=self.train_params.workers,
            use_multiprocessing=self.train_params.use_multiprocessing
        )

        # save model as pb file
        EXPORT_PATH = os.path.join(
            self.train_params.output_dir, 'trained_model', datetime.now().strftime("%Y_%m_%d-%H:%M:%S"))

        tf.saved_model.save(
            obj=model, export_dir=EXPORT_PATH)  # with default serving function

        print("Exported trained model to {}".format(EXPORT_PATH))
        io_operator.write('train_logs.csv', self.csv_path)

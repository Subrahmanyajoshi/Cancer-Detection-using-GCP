import argparse
import importlib
import os
import zipfile
from argparse import Namespace

from detectors.common import BucketOps, SystemOps, YamlConfig
from detectors.tf_gcp.trainer.data_ops.data_generator import DataGenerator
from detectors.tf_gcp.trainer.data_ops.io_ops import CloudIO, LocalIO
from detectors.tf_gcp.trainer.models.models import CNNModel, VGG19Model


class Trainer(object):
    MODEL_NAME = 'Breast_cancer_detector.h5'

    def __init__(self, config: dict, bucket: str = None):
        """ Init method
        Args:
            config (dict): Dictionary containing configurations
            bucket (str): Bucket name
        """
        self.run_type = config.get('run_type', '').strip()
        self.train_params = Namespace(**config.get('train_params'))
        self.model_params = Namespace(**config.get('model_params'))
        self.bucket = BucketOps.get_bucket(bucket)
        self.callbacks = self.init_callbacks()

    def init_callbacks(self):
        """ Creates callback objects mentioned in configurations """
        callbacks = []
        module = importlib.import_module('tensorflow.keras.callbacks')
        for cb in self.train_params.callbacks:
            if cb == 'ModelCheckpoint':
                _, filename = os.path.split(self.train_params.callbacks[cb]['filepath'])
                self.train_params.callbacks[cb]['filepath'] = os.path.join('checkpoints', filename)
            obj = getattr(module, cb)
            callbacks.append(obj(**self.train_params.callbacks[cb]))
        return callbacks

    @staticmethod
    def cleanup():
        """ Deletes temporary directories created while training"""
        SystemOps.check_and_delete('all_images')
        SystemOps.check_and_delete('checkpoints')
        SystemOps.check_and_delete('trained_model')

    def train(self):
        if self.model_params.model == 'CNN':
            Model = CNNModel(img_shape=(None,) + self.model_params.image_shape).build(self.model_params)
        elif self.model_params.model == 'VGG19':
            Model = VGG19Model(img_shape=(None,) + self.model_params.image_shape).build(self.model_params)
        else:
            raise NotImplementedError("Specified model is currently not supported")
        Model.summary()

        if self.run_type == 'ai_platform':
            io_operator = CloudIO(input_dir=self.train_params.input_dir, bucket=self.bucket)
            os.system(f"gsutil -m cp -r {os.path.join(self.train_params.input_dir, 'all_images.zip')} ./")
            with zipfile.ZipFile('all_images.zip', 'r') as zip_ref:
                zip_ref.extractall('./all_images')
            SystemOps.check_and_delete('all_images.zip')
        elif self.run_type == 'local':
            io_operator = LocalIO(input_dir=self.train_params.input_dir)
        else:
            raise RuntimeError(f"run_type must be either 'local' or 'ai_platform', it can't be {self.run_type}")

        X_train_files, y_train, X_val_files, y_val = io_operator.load()
        train_generator = DataGenerator(image_filenames=X_train_files,
                                        labels=y_train,
                                        batch_size=self.train_params.batch_size,
                                        dest_dir='./all_images/',
                                        bucket=self.bucket)
        validation_generator = DataGenerator(image_filenames=X_val_files,
                                             labels=y_val,
                                             batch_size=self.train_params.batch_size,
                                             dest_dir='./all_images/',
                                             bucket=self.bucket)

        SystemOps.clean_dir('checkpoints')

        history = Model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=self.train_params.num_epochs,
            callbacks=self.callbacks,
            steps_per_epoch=self.train_params.steps_per_epoch,
            workers=self.train_params.workers,
            use_multiprocessing=self.train_params.use_multiprocessing
        )

        # save model as hdf5 file
        Model.save(Trainer.MODEL_NAME)

        SystemOps.create_dir('trained_model')
        SystemOps.move(Trainer.MODEL_NAME, 'trained_model')
        # send saved model to 'trained_model' directory
        io_operator.write('trained_model', self.train_params.output_dir)
        io_operator.write('checkpoints', self.train_params.output_dir)

        # Delete unwanted directories used while training
        Trainer.cleanup()


def main():
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument('--package-path', help='GCS or local path to training data',
                        required=False)
    parser.add_argument('--bucket', help='Name of the GCS bucket in which data is stored',
                        required=False)
    parser.add_argument('--job-dir', type=str, help='GCS location to write checkpoints and export models',
                        required=False)
    parser.add_argument('--config', type=str, required=False, default='../config/config.yaml',
                        help='Yaml configuration file path')

    args = parser.parse_args()
    config = YamlConfig.load(filepath=args.config)

    trainer = Trainer(config=config, bucket=args.bucket)
    trainer.train()


if __name__ == "__main__":
    main()

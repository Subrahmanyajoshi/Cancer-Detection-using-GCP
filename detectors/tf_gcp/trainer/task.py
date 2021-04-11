import argparse
import os
import zipfile
from argparse import Namespace

import tensorflow as tf

from detectors.common import BucketOps, SystemOps
from detectors.tf_gcp.trainer.data_ops.data_generator import DataGenerator
from detectors.tf_gcp.trainer.data_ops.io_ops import CloudIO, LocalIO
from detectors.tf_gcp.trainer.models import cnn


class Trainer(object):
    CLASSIFICATION_MODEL = 'Breast_cancer_detector.hdf5'

    def __init__(self, args: Namespace):
        self.args = args

    @staticmethod
    def cleanup():
        SystemOps.check_and_delete('all_images')
        SystemOps.check_and_delete('all_images.zip')
        SystemOps.check_and_delete('checkpoints')

    def train(self):
        Model = cnn.keras_estimator()
        Model.summary()

        bucket = BucketOps.get_bucket(self.args.bucket)
        
        if self.args.input_dir.startswith('gs://'):
            io_operator = CloudIO(input_dir=self.args.input_dir, bucket=bucket)
            os.system(f"gsutil cp -r {os.path.join(self.args.input_dir, 'all_images.zip')} ./")
            with zipfile.ZipFile('all_images.zip', 'r') as zip_ref:
                zip_ref.extractall('./')
        else:
            io_operator = LocalIO(input_dir=self.args.input_dir)

        X_train_files, y_train, X_val_files, y_val = io_operator.load()
        train_generator = DataGenerator(X_train_files, y_train, self.args.batch_size,
                                        './all_images/', bucket)
        validation_generator = DataGenerator(X_val_files, y_val, self.args.batch_size,
                                             './all_images/', bucket)

        SystemOps.clean_dir('checkpoints')
        cp_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints/model.{epoch:02d}-{'
                                                                    'val_loss:.2f}.hdf5',
                                                           monitor='val_loss',
                                                           save_freq='epoch',
                                                           save_best_only=False)
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.args.output_dir, 'tensorboard'))

        epochs = self.args.num_epochs
        history = Model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=[cp_checkpoint, tensorboard],
            steps_per_epoch=self.args.steps_per_epoch
        )

        Model.save(Trainer.CLASSIFICATION_MODEL)
        io_operator.write(Trainer.CLASSIFICATION_MODEL, self.args.output_dir)
        io_operator.write('checkpoints', self.args.output_dir)
        Trainer.cleanup()


def main():
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument('--package-path', help='GCS or local path to training data',
                        required=False)
    parser.add_argument('--bucket', help='Name of the GCS bucket in which data is stored',
                        required=False)
    parser.add_argument('--batch_size', help='Number of examples to compute gradient over.', type=int,
                        default=32, required=False)
    parser.add_argument('--output-dir', help='GCS location to write checkpoints and export models',
                        required=False)
    parser.add_argument('--job-dir', type=str, help='GCS location to write checkpoints and export models',
                        required=False)
    parser.add_argument('--input-dir', type=str, help='path to train data directory',
                        required=True)
    parser.add_argument('--num-epochs', type=int, help='number of epochs',
                        required=True)
    parser.add_argument('--steps-per-epoch', type=int, help='number of steps per epoch', default=2,
                        required=False)
    args = parser.parse_args()
    trainer = Trainer(args=args)
    trainer.train()


if __name__ == "__main__":
    main()

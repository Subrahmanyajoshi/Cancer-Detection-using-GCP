import argparse
import os
from tensorflow.python.lib.io import file_io
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from . import model

CLASSIFICATION_MODEL = 'cl_model.hdf5'


def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='r') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def get_args():
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--package-path',
        help='GCS or local path to training data',
    )
    parser.add_argument(
        '--job-dir',
        type=str,
        help='GCS location to write checkpoints and export models'
    )
    parser.add_argument(
        '--train-dir',
        help='GCS or local path to training data',
        required=True
    )
    args, _ = parser.parse_known_args()
    return args


def train_and_evaluate(args):
    Model = model.keras_estimator()
    Model.summary()

    DATA_PATH = args.train_dir
    train_dir = os.path.join(DATA_PATH, 'train')
    val_dir = os.path.join(DATA_PATH, 'val')

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(300, 300),
        batch_size=10,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(300, 300),
        batch_size=10,
        class_mode='binary')

    epochs = 10
    history = Model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs
    )

    Model.save('cl_model.hdf5')

    job_dir = args.job_dir + '/export'

    if job_dir.startswith("gs://"):
        Model.save(CLASSIFICATION_MODEL)
        copy_file_to_gcs(job_dir, CLASSIFICATION_MODEL)
    else:
        Model.save(os.path.join(job_dir, CLASSIFICATION_MODEL))


# Running the app
if __name__ == "__main__":
    args = get_args()
    arguments = args.__dict__
    train_and_evaluate(args)

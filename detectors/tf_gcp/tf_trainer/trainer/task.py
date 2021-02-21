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
        '--bucket',
        help='GCS path to data. We assume that data is in gs://BUCKET/babyweight/preproc/',
        required=True
    )
    parser.add_argument(
        '--output_dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--job-dir',
        type=str,
        help='GCS location to write checkpoints and export models'
    )

    args, _ = parser.parse_known_args()
    return args


def train_and_evaluate(args):
    Model = model.keras_estimator()
    Model.summary()

    DATA_PATH = os.path.join(args.bucket, 'test_cnn_data')
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

    cp_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/model.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                       monitor='val_loss',
                                                       save_freq='epoch',
                                                       save_best_only=False)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='tensorboard')

    epochs = 10
    history = Model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[cp_checkpoint, tensorboard]
    )

    Model.save(os.path.join(args.output_dir, 'cl_model.hdf5'))
#     job_dir = args.job_dir + '/export'

#     if job_dir.startswith("gs://"):
#         Model.save(CLASSIFICATION_MODEL)
#         copy_file_to_gcs(job_dir, CLASSIFICATION_MODEL)
#     else:
#         Model.save(os.path.join(job_dir, CLASSIFICATION_MODEL))


# Running the app
if __name__ == "__main__":
    args = get_args()
    arguments = args.__dict__
    train_and_evaluate(args)
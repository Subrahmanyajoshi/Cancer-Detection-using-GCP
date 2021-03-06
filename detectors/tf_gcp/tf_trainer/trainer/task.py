import argparse
import os
import glob
import shutil
from google.cloud import storage

from tensorflow.python.lib.io import file_io
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from trainer.data_generator import MyCustomGenerator
from trainer import model

from io import StringIO, BytesIO
from tensorflow.python.lib.io import file_io

CLASSIFICATION_MODEL = 'cl_model.hdf5'


def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='wb+') as output_f:
            output_f.write(input_f.read())


def copy_directory_to_gcs(local_path, bucket, gcs_path):
    """Recursively copy a directory of files to GCS.

    local_path should be a directory and not have a trailing slash.
    """
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
            continue
        remote_path = os.path.join(gcs_path, local_file)
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)

def get_args():
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--package-path',
        help='GCS or local path to training data',
        required=False
    )
    parser.add_argument(
        '--bucket',
        help='GCS path to data. We assume that data is in gs://BUCKET/babyweight/preproc/',
        required=False
    )
    parser.add_argument(
        '--batch_size',
        help='Number of examples to compute gradient over.',
        type=int,
        default=512
    )
    parser.add_argument(
        '--output-dir',
        help='GCS location to write checkpoints and export models',
        required=False
    )
    parser.add_argument(
        '--job-dir',
        type=str,
        help='GCS location to write checkpoints and export models',
        required=False
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        help='path to train directory',
        required=True
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        help='number of epochs',
        required=True
    )

    args_, _ = parser.parse_known_args()
    return args_

def load_npy_from_gcs(file_path):
    _file = BytesIO(file_io.read_file_to_string(file_path, binary_mode=True))
    np_data = np.load(_file)
    return np_data

def train_and_evaluate(args_):
    Model = model.keras_estimator()
    Model.summary()
    
    client = storage.Client()
    bucket = client.get_bucket(args_.bucket)
    
    X_train_filenames = load_npy_from_gcs(os.path.join(('gs://'+args_.bucket), args_.input_dir, 'train', 'X_train_filenames.npy'))
    y_train = load_npy_from_gcs(os.path.join(('gs://'+args_.bucket), args_.input_dir, 'train', 'y_train.npy'))
    X_val_filenames = load_npy_from_gcs(os.path.join(('gs://'+args_.bucket), args_.input_dir, 'val', 'X_val_filenames.npy'))
    y_val = load_npy_from_gcs(os.path.join(('gs://'+args_.bucket), args_.input_dir, 'val', 'y_val.npy'))
    train_dir = args_.input_dir

    """
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
    """

    train_generator = MyCustomGenerator(X_train_filenames, y_train, args_.batch_size, 
                                        os.path.join(train_dir, 'all_images/'), bucket)
    validation_generator = MyCustomGenerator(X_val_filenames, y_val, args_.batch_size, 
                                             os.path.join(train_dir, 'all_images/'), bucket)

    if os.path.exists('checkpoints'):
        shutil.rmtree('checkpoints')
    os.mkdir('checkpoints')
    
    cp_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints/model.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                       monitor='val_loss',
                                                       save_freq='epoch',
                                                       save_best_only=False)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    epochs = args_.num_epochs
    history = Model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[cp_checkpoint, tensorboard]
    )
    

    if args.output_dir.startswith("gs://"):
        Model.save(CLASSIFICATION_MODEL)
        copy_file_to_gcs(args.output_dir, CLASSIFICATION_MODEL)
        copy_directory_to_gcs('./checkpoints', bucket, os.path.join(args.output_dir, 'checkpoints'))
    else:
        Model.save(os.path.join(job_dir, CLASSIFICATION_MODEL))


# Running the app
if __name__ == "__main__":
    args = get_args()
    arguments = args.__dict__
    train_and_evaluate(args)

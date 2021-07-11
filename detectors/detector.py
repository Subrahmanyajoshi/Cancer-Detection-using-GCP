import argparse
import os
from argparse import Namespace

from cv2 import imread, resize

from detectors.common import YamlConfig, SystemOps
from detectors.tf_gcp.trainer.models.models import CNNModel, VGG19Model
from detectors.tf_gcp.trainer.task import Trainer


class Predictor(object):

    def __init__(self, config: dict):
        self.config = config.get('predict_params', {})
        self.data_path = self.config.get('data_path', None)
        self.img_shape = eval(config.get('train_params').get('image_shape'))
        self.model_params = Namespace(**config.get('model_params'))
        self.model = self.load_model(self.config.get('model_path', None))

    def load_model(self, model_path: str):
        if self.model_params.model == 'CNN':
            model = CNNModel(img_shape=(None,) + self.img_shape).build(self.model_params)
        elif self.model_params.model == 'VGG19':
            model = VGG19Model(eval(self.img_shape)).build(self.model_params)
        else:
            raise NotImplementedError(f"{self.model_params.model} model is currently not supported. "
                                      f"Please choose between CNN and VGG19")

        if model_path.startswith('gs://'):
            SystemOps.run_command(f"gsutil -m cp -r {model_path} ./")
            model_path = os.path.basename(model_path)

        model.load_weights(model_path)
        return model

    def predict(self, img_path: str):
        image = imread(img_path)
        image = resize(image, self.img_shape) / 255
        image = image[None, :, :, :]
        result = self.model.predict(image)
        if result > 0.5:
            print(f'Image: {os.path.relpath(img_path, self.data_path)}, Prediction: Cancerous, '
                  f'Confidence: {result * 100}%')
        else:
            print(f'Image: {os.path.relpath(img_path, self.data_path)}, Prediction: Benign, '
                  f'Confidence: {(result - 1) * 100}%')

    def run(self):
        if self.data_path.startswith('gs://'):
            SystemOps.run_command(f"gsutil -m cp -r {self.data_path} ./")
            self.data_path = os.path.basename(self.data_path)

        if os.path.isfile(self.data_path):
            self.predict(self.data_path)
            return

        print(f'Reading images from {self.data_path}')
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                abs_path = os.path.join(root, file)
                self.predict(abs_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true', required=False,
                        help='A boolean switch to tell the script to run the predictions')
    parser.add_argument('--train', action='store_true', required=False,
                        help='A boolean switch to tell the script to run training')
    parser.add_argument('--train_type', type=str, required=False, choices=['local', 'ai_platform'],
                        help='to specify whether to train locally or to submit train job to AI platform')
    parser.add_argument('--config', type=str, required=True,
                        help='Yaml configuration file path')

    args = parser.parse_args()

    if not args.train and not args.predict:
        raise ValueError('Please specify either --train or --predict command line argument while running')

    if args.train and args.train_type is None:
        raise ValueError("'train_type' argument is required, choices available are ['local', 'ai_platform']")

    config = YamlConfig.load(filepath=args.config)

    if args.train:
        print('Initialising training')
        trainer = Trainer(config=config)
        trainer.train()

    if args.predict:
        print('Initialising predicting')
        predicter = Predictor(config=config)
        predicter.run()


if __name__ == '__main__':
    main()

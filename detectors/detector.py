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
        self.data_path = self.config.get('data_path', '')
        self.img_shape = eval(config.get('train_params').get('image_shape'))
        self.model_params = Namespace(**config.get('model_params'))
        self.model_path = self.config.get('model_path')
        self.model = self.load_model()
        self.img_width = self.img_shape[0]
        self.img_height = self.img_shape[1]
        self.img_channels = self.img_shape[2]

    def load_model(self):
        if self.model_params.model == 'CNN':
            model = CNNModel(img_shape=(None,) + self.img_shape).build(self.model_params)
        elif self.model_params.model == 'VGG19':
            model = VGG19Model(self.img_shape).build(self.model_params)
        else:
            raise NotImplementedError(f"{self.model_params.model} model is currently not supported. "
                                      f"Please choose between CNN and VGG19")

        if self.model_path.startswith('gs://'):
            SystemOps.run_command(f"gsutil -m cp -r {self.model_path} ./")
            self.model_path = os.path.basename(self.model_path)

        model.load_weights(self.model_path)
        return model

    def predict(self, img_path: str):
        image = imread(img_path)
        image = resize(image, (self.img_width, self.img_height)) / 255
        image = image[None, :, :, :]
        result = self.model.predict(image)
        result = result[0][0]
        if result > 0.5:
            print(f'Image: {os.path.relpath(img_path, self.data_path)}, Prediction: Cancerous, '
                  f'Confidence: {round((result* 100), 2)}%')
        else:
            print(f'Image: {os.path.relpath(img_path, self.data_path)}, Prediction: Benign, '
                  f'Confidence: {round(((1 - result)* 100), 2)}%')

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

    def clean_up(self):
        SystemOps.check_and_delete(self.data_path)
        SystemOps.check_and_delete(self.model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true', required=False,
                        help='A boolean switch to tell the script to run the predictions')
    parser.add_argument('--train', action='store_true', required=False,
                        help='A boolean switch to tell the script to run training')
    parser.add_argument('--config', type=str, required=True,
                        help='Yaml configuration file path')

    args = parser.parse_args()

    if not args.train and not args.predict:
        raise ValueError('Please specify either --train or --predict command line argument while running')

    config = YamlConfig.load(filepath=args.config)

    if args.train:
        print('Initialising training')
        trainer = Trainer(config=config)
        trainer.train()
        trainer.clean_up()

    if args.predict:
        print('Initialising testing')
        predictor = Predictor(config=config)
        predictor.run()
        predictor.clean_up()


if __name__ == '__main__':
    main()

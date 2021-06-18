import argparse
import os

from cv2 import imread, resize

from detectors.common import YamlConfig
from detectors.tf_gcp.trainer.models.models import CNNModel
from detectors.tf_gcp.trainer.task import Trainer


class Predictor(object):

    def __init__(self, config: dict):
        self.config = config.get('predict_params', {})
        self.data_path = self.config.get('data_path', None)
        self.img_shape = config.get('model_params').get('image_shape')
        self.model = self.load_model(self.config.get('model_path', None))

    def load_model(self, model_path: str):
        model = CNNModel(img_shape=(None,) + self.img_shape).build()
        model.load_weights(model_path)
        return model

    def predict(self, img_path: str):
        image = imread(img_path)
        image = resize(image, self.img_shape) / 255
        image = image[None, :, :, :]
        result = self.model.predict(image)
        if result > 0.5:
            print(f'Image: {os.path.basename(img_path)}, Prediction: Cancerous, Confidence: {result*100}%')
        else:
            print(f'Image: {os.path.basename(img_path)}, Prediction: Cancerous, Benign: {(result-1)*100}%')

    def run(self):
        if os.path.isfile(self.data_path):
            self.predict(self.data_path)
            return
        for file in os.listdir(self.data_path):
            print(f'Reading images from {self.data_path}')
            abs_path = os.path.join(self.data_path, file)
            self.predict(abs_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true', required=False,
                        help='A boolean switch to tell the script to run the predictions')
    parser.add_argument('--train', action='store_true', required=False,
                        help='A boolean switch to tell the script to run training')
    parser.add_argument('--train-config', type=str, required=False, default='../config/config.yaml',
                        help='Yaml configuration file path')
    
    args = parser.parse_args()

    if not args.train and not args.predict:
        raise ValueError('Please specify either --train or --predict option while running')

    config = YamlConfig.load(filepath=args.train_config)

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

import argparse
import json
import os

from cv2 import imread, resize

from datetime import datetime

import detectors
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


class TrainerRunner(object):

    def __init__(self, config: dict, train_type: str):
        self.config = config
        self.train_type = train_type

    def run(self):
        if self.train_type == 'local':
            trainer = Trainer(config=self.config)
            trainer.train()
        elif self.train_type == 'ai_platform':
            train_config = json.dumps(self.config)
            out_dir = self.config.get('train_params').get('output_dir')
            job_name = f"breastcancer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            os.system(f'gcloud ai-platform jobs submit training {job_name} '
                      f'--package-path={os.getcwd()}'
                      f'--job-dir={out_dir}'
                      f'--module-name={detectors.tf_gcp.trainer.task}'
                      f'--train-config={train_config}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true', required=False,
                        help='A boolean switch to tell the script to run the predictions')
    parser.add_argument('--train', action='store_true', required=False,
                        help='A boolean switch to tell the script to run training')
    parser.add_argument('--train_type', type=str, required=False, choices=['local', 'ai_platform'],
                        help='to specify whether to train locally or to submit train job to AI platform')
    parser.add_argument('--train-config', type=str, required=True,
                        help='Yaml configuration file path')

    args = parser.parse_args()

    if not args.train and not args.predict:
        raise ValueError('Please specify either --train or --predict command line argument while running')

    if args.train and args.train_type is None:
        raise ValueError("'train_type' argument is required, choices available are ['local', 'ai_platform']")

    config = YamlConfig.load(filepath=args.train_config)

    if args.train:
        print('Initialising training')
        runner = TrainerRunner(config=config, train_type=args.train_type)
        runner.run()

    if args.predict:
        print('Initialising predicting')
        predicter = Predictor(config=config)
        predicter.run()


if __name__ == '__main__':
    main()

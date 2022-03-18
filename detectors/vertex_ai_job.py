import argparse
import os

from detectors.tf_gcp.common import YamlConfig
from detectors.tf_gcp.data_ops.io_ops import CloudIO
from detectors.tf_gcp.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument('--package-path', help='GCS or local path to training data',
                        required=False)
    parser.add_argument('--job-dir', type=str, help='GCS location to write checkpoints and export models',
                        required=False)
    parser.add_argument('--train-config', type=str, help='config file containing train configurations',
                        required=False)
    args = parser.parse_args()

    # Copy config file containing train configurations
    CloudIO.copy_from_gcs(args.train_config, 'tf_gcp/trainer/')
    config = YamlConfig.load(filepath=os.path.abspath('config.yaml'))

    # Create and run trainer
    trainer = Trainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()

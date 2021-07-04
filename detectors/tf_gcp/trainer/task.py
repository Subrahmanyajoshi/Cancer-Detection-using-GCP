import argparse
import json

from detectors.tf_gcp.trainer.trainer import Trainer


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
    config = json.loads(args.train_config)
    trainer = Trainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()

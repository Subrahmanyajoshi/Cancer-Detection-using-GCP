import argparse
import os

import split_folders


class DataSetSplitter(object):
    """Splits a large dataset of images into train, test and validation datasets"""

    SPLIT_RATIO = (0.8, 0.1, 0.1)  # Train, Test and Validation
    SEED = 1337

    def __init__(self, inp_dir: str, out_dir: str):
        self.__inp_dir = inp_dir
        self.__out_dir = out_dir

    def split(self):

        if not os.path.isdir(self.__inp_dir):
            raise ValueError("{} doesn't seem to be a existing directory".format(self.__inp_dir))

        if not os.path.isdir(self.__out_dir):
            raise ValueError("{} doesn't seem to be a existing directory".format(self.__out_dir))

        split_folders.ratio(self.__inp_dir, output=self.__out_dir,
                            seed=DataSetSplitter.SEED, ratio=DataSetSplitter.SPLIT_RATIO)


def main():
    parser = argparse.ArgumentParser(description='Dataset Splitter')
    parser.add_argument('--input-dir', type=str, required=True, help='path to input directory')
    parser.add_argument('--output-dir', type=str, required=True, help='path to output directory')

    args = parser.parse_args()
    splitter = DataSetSplitter(args.input_dir, args.output_dir)
    splitter.split()


if __name__ == '__main__':
    main()

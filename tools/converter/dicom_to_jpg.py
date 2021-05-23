import argparse
import os
from distutils.dir_util import copy_tree

import cv2
import pydicom as dicom


class DicomToJpgConverter(object):
    """Converts images in dicom format to jpg format in segmented form
        details about dicom image format-
            https://en.wikipedia.org/wiki/DICOM
    """

    def __init__(self, src: str, dest: str):
        self.__src = src
        self.__dest = dest

    def copy_directory(self):
        """Copies source directory contents to destination directory"""
        if not os.path.exists(self.__src):
            raise ValueError("Jpg folder path doesn't exist")

        if not os.path.exists(self.__dest):
            raise ValueError("dicom folder path doesn't exist")

        copy_tree(self.__src, self.__dest)

    def convert(self):
        """Converts images in dicom format to jpg format"""

        self.copy_directory()
        for root, dirs, files in os.walk(self.__dest):
            for file in files:
                if file.endswith('.dcm'):
                    full_path = os.path.join(root, file)
                    ds = dicom.dcmread(full_path)
                    pixel_array_numpy = ds.pixel_array
                    os.remove(full_path)
                    full_path = full_path.replace('.dcm', '.jpg')
                    cv2.imwrite(full_path, pixel_array_numpy)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dicom-path', type=str, required=True,
                        help="path to directory where dicom images are stored")
    parser.add_argument('--jpg-path', type=str, required=True,
                        help="Directory where converted images are supposed to be stored")
    args = parser.parse_args()
    converter = DicomToJpgConverter(args.dicom_path, args.jpg_path)
    converter.convert()


if __name__ == '__main__':
    main()

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'opencv-python-headless==4.5.3.56',
    'protobuf==3.17.3'
]

setup(
    name='Breast Cancer Detection',
    version='0.1',
    author='Subrahmanya Joshi',
    author_email='subrahmanyajoshi123@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='breast cancer detection using google cloud ml engine',
    requires=[]
)

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'scikit-learn==0.20.1',
    'tensorflow==2.1.0',
    'opencv-python 4.5.1.48'
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

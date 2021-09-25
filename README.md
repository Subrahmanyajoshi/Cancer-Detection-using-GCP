
## Overview
- The goal of this project is to provide a platform to build and train machine learning models for medical image 
  classification using Google Cloud's AI platform.
- In this project I have used mammography(breast cancer) images, but I am pretty sure any type of
  medical image datasets can be used by tweaking the model parameters [here](detectors/tf_gcp/trainer/models/models.py).

## Dataset
- The dataset was obtained from mendeley website [here](https://data.mendeley.com/datasets/ywsbh3ndr8/5).
- The dataset in the link above contains DDSM, INBREAST and a directory containing DDSM+INBREAST+MIAS
   combined datasets. Only DDSM dataset was used in this project.
   

## Preparing Train dataset
- No augmentation was done since the dataset was already augmented. If it's required, augmentation procedure is available 
   [here](tools/preprocessers/image_augmenter.ipynb)
- Open notebook train_data_creator.ipynb located at tools/preprocessors and run it.
- Input data folder should contain one folder per class containing images.
- Give an empty directory as the destination path and run the notebook.
- Once notebook is run completely, following folders/files will be created in destination directory.
    - all_images.zip: contains all images which are supposed to be used for training, in zipped format.
    - train: contains 2 npy files. one containing train image names and one containing labels.
    - val: contains 2 npy files. one containing validation image names and one containing labels.
- The entire destination folder needs to be uploaded to Google Storage before training is started.

## Training models locally

- Install packages from requirements.txt
```shell
pip install -r requirements.txt
```
- Open config file at config/config.yaml and update it accordingly.
- Go to project root and run following. It sets environment variable 
   PYTHONPATH to project root so that modules can be imported easily.
   
```shell
export PYTHONPATH=$(pwd):${PYTHONPATH}
```
- Run trainer
```shell
python3 -m detectors.detector --train --config='./config/config.yaml'
```


## Submitting Training job to AI Platform

- Go to google cloud console and create an instance of AI Notebooks. 
   If not known how to do that, follow the procedure given [here](https://cloud.google.com/notebooks/docs/create-new).
   (Create the notebook with low specifications, as we will not be running actual training here. 
   This just acts as a base machine to submit the job to AI platform. 
   The best choice is n1-standard-2 machines which have 7.5 gb memory and 2 vCpus).
- Launch Jupyter lab and open a terminal.
- Clone this repository.
```shell
git clone https://github.com/Subrahmanyajoshi/Breast-Cancer-Detection.git
```
- Create a google storage bucket. If not known how to do that, 
   follow the procedure given [here](https://cloud.google.com/storage/docs/creating-buckets)
- Upload the training dataset folder which contains all images zip file along with 'train' and 'val' 
   folders containing npy files into newly created bucket.
- open config file at config/config.yaml and update it accordingly. Make sure to mention full paths
   starting from 'gs://' while specifying paths inside the bucket.
- Open the notebook detectors/tf_gcp/ai_platform_trainer.ipynb and run the notebook 
   following the steps given there.
- The notebook will submit the training job to AI Platform. 

### Activating and using tensorboard to monitor training

- Tensorboard will be running on port 6006 by default.
- A firewall rule must be set up to open this port, follow the procedure given
   [here](https://docs.bitnami.com/google/faq/administration/use-firewall/).
- Once done, open a terminal and run following. Provide the path to tensorboard directory, 
   specified in config file.
```shell
tensorboard --logdir <path/to/log/directory> --bind_all
```
- Get the external Ip of the VM on which notebook is running, from 'VM Instances' page on google cloud console.
- Open a browser and open following link .
```text
http://<external_ip_address>:6006/
```

## Predicting
- Open config file at config/config.yaml and update model path, and data path at the very bottom.
- Install opencv-python-headless package
```shell
pip3 install opencv-python-headless==4.5.3.56
```
- Go to project root and run following. It sets environment variable PYTHONPATH to project root so that 
   modules can be imported easily.
```shell
export PYTHONPATH=$(pwd):${PYTHONPATH}
```
- Run predictor
```shell
python3 -m detectors.detector --predict --config='./config/config.yaml'
```
- Classification results will be printed on the screen. If data path is a directory, 
   all images inside the directories, sub-directories will be read and will be run against model.
   
## Results

- DDSM dataset images were visually not distinguishable.
- I was able to get the training accuracy of 99% within 3 epochs for both CNN and VGG19 models 
  owing to large number of images in the dataset.
- Testing accuracy was 100% for both CNN and VGG19. They classified all test images accurately as 
  benign and cancerous.

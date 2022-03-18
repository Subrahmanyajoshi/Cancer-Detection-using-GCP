# Training tensorflow model using Google Cloud Vertex AI

# Environment Variables
export PROJECT='cancer-detector-323506'
export REGION='us-central1'
export BUCKET='cancer-detector-323506'
export MACHINE_TYPE='n1-highmem-4'
export ACCELERATOR_TYPE=NVIDIA_TESLA_K80
export ACCELERATOR_COUNT=1
export REPLICA_COUNT=1
export EXECUTE_IMAGE_URI='us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-8:latest'
export LOCAL_PACKAGE_PATH='/home/jupyter/Cancer-Detection-using-GCP'
export PYTHON_MODULE='detectors.vertex_ai_job'
export JOBNAME=cancer_detector_$(date -u +%y%m%d_%H%M%S)

# gcloud configurations
gcloud config set project $PROJECT
gcloud config set compute/region $REGION                                                                

# Move latest config file to google storage bucket
gsutil -m cp -r ./config gs://$BUCKET

# Submit training job  
gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name=$JOBNAME \
    --worker-pool-spec=machine-type=$MACHINE_TYPE,replica-count=$REPLICA_COUNT,executor-image-uri=$EXECUTE_IMAGE_URI,local-package-path=$LOCAL_PACKAGE_PATH,python-module=$PYTHON_MODULE \
    --args=--train-config=gs://$BUCKET/config/config.yaml


# Submit training job with accelerators
#gcloud ai custom-jobs create \
#    --region=$REGION \
#    --display-name=$JOBNAME \
#    --worker-pool-spec=machine-type=$MACHINE_TYPE,replica-count=$REPLICA_COUNT,accelerator-type=$ACCELERATOR_TYPE,accelerator-count=$ACCELERATOR_COUNT,executor-image-uri=$EXECUTE_IMAGE_URI,local-package-path=$LOCAL_PACKAGE_PATH,python-module=$PYTHON_MODULE \
#    --args=--data-dir=gs://text-analysis-323506/data,--save-dir=gs://text-analysis-323506/saved_model,--batch-size=1024
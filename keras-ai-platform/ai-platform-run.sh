REGION="us-east1"
BUCKET_NAME="qognition-ai-ic"

now=$(date +"%Y%m%d_%H%M%S")
MODEL_DIR="gs://$BUCKET_NAME/Premade/logs_$now/"

MAIN_TRAINER_MODULE="trainer.task"
TRAINER_PACKAGE_PATH="$(pwd)/trainer/"
PACKAGE_STAGING_PATH="gs://$BUCKET_NAME/"
CONFIG="config.yaml"
STRATEGY="mirror"

JOB_NAME="Keras_model_$STRATEGY_$now"

gcloud ai-platform jobs submit training $JOB_NAME \
    --staging-bucket $PACKAGE_STAGING_PATH \
    --runtime-version 2.1 \
    --python-version 3.7 \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region $REGION \
    --config $CONFIG \
    -- \
    --strategy $STRATEGY


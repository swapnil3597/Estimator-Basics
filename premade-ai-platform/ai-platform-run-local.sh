BUCKET_NAME="qognition-ai-ic"

now=$(date +"%Y%m%d_%H%M%S")
TRAINER_PACKAGE_PATH="$(pwd)/trainer/"
MODEL_DIR="gs://$BUCKET_NAME/Premade/logs_$now/"

# python3 -m trainer.task \
#     --model-dir $MODEL_DIR

gcloud ai-platform local train --module-name trainer.task \
  --package-path $TRAINER_PACKAGE_PATH


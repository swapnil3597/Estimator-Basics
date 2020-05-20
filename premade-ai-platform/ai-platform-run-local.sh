BUCKET_NAME="qognition-ai-ic"

now=$(date +"%Y%m%d_%H%M%S")
MODEL_DIR="gs://$BUCKET_NAME/Premade/logs_$now/"

python3 -m trainer.task \
    --model-dir $MODEL_DIR


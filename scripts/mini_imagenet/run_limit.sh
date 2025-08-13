LOG_DIR="logs/limit"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs_base in 600
do
    LOG_FILE="$LOG_DIR/MINI-limit-epochs_${epochs_base}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs_base=$epochs_base"

    PRETRAINED_MODEL="./params/pretrain_MINI_${epochs_base}.pth"

    python train.py limit \
        -project limit \
        -dataset mini_imagenet \
        -dataroot data \
        -epochs_base 20 \
        -lr_base 0.0002 \
        -lrg 0.0002 \
        -gamma 0.3 \
        -model_dir $PRETRAINED_MODEL \
        -num_tasks 32 \
        -temperature 0.5 \
        -schedule Cosine \
        -model_dir $PRETRAINED_MODEL \
        -gpu $1 >> "$LOG_FILE"
done
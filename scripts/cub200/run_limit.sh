LOG_DIR="logs/limit"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs_base in 600
do
    LOG_FILE="$LOG_DIR/CUB-limit-epochs_${epochs_base}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs_base=$epochs_base"

    PRETRAINED_MODEL="./params/pretrain_CUB_${epochs_base}.pth"

    python train.py limit \
        -project limit \
        -dataset cub200 \
        -dataroot data \
        -epochs_base 40 \
        -lr_base 0.0002 \
        -lrg 0.0002 \
        -step 20 \
        -gamma 0.5 \
        -model_dir $PRETRAINED_MODEL \
        -num_tasks 32 \
        -gpu $1 >> "$LOG_FILE"
done
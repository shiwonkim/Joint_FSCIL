LOG_DIR="logs/s3c"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

epochs_new=100
for epochs_base in 100 200 400 600 800 1000
do
    LOG_FILE="$LOG_DIR/MINI-s3c-epochs_${epochs_base}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs_base=$epochs_base"

    python train.py s3c \
        -project s3c \
        -dataset mini_imagenet \
        -dataroot data/mini_imagenet \
        -base_mode 'ft_cos' \
        -new_mode 'avg_cos' \
        -gamma 0.1 \
        -lr_base 0.1 \
        -lr_new 0.01 \
        -decay 0.0005 \
        -epochs_base $epochs_base \
        -epochs_new $epochs_new \
        -schedule Cosine \
        -lamda_proto 5 \
        -temperature 16 \
        -gpu $1 >> "$LOG_FILE"
done
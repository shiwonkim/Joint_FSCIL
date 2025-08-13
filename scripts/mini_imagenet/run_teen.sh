LOG_DIR="logs/teen"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs_base in 1000
do
    LOG_FILE="$LOG_DIR/MINI-TEEN-epochs_${epochs_base}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs_base=$epochs_base"

    python train.py teen \
        -project teen \
        -dataset mini_imagenet \
        -dataroot data \
        -base_mode 'ft_cos' \
        -new_mode 'avg_cos' \
        -gamma 0.1 \
        -lr_base 0.1 \
        -decay 0.0005 \
        -epochs_base $epochs_base \
        -schedule Cosine \
        -gpu $1 \
        -temperature 32 \
        -batch_size_base 128 "${@:2}" >> "$LOG_FILE"
done
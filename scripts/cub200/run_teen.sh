LOG_DIR="logs/teen"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs_base in 400
do
    LOG_FILE="$LOG_DIR/CUB-TEEN-epochs_${epochs_base}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs_base=$epochs_base"

    python train.py teen \
        -project teen \
        -dataset cub200 \
        -dataroot data \
        -base_mode 'ft_cos' \
        -new_mode 'avg_cos' \
        -gamma 0.25 \
        -lr_base 0.004 \
        -lr_new 0.1 \
        -decay 0.0005 \
        -epochs_base $epochs_base \
        -schedule Cosine \
        -gpu $1 \
        -temperature 32 \
        -batch_size_base 128 \
        -softmax_t 16 \
        -shift_weight 0.5 >> "$LOG_FILE"
done
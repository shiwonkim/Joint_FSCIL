LOG_DIR="logs/fact"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs_base in 400
do
    LOG_FILE="$LOG_DIR/CUB-FACT-epochs_${epochs_base}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs_base=$epochs_base"

    python train.py fact \
        -project fact \
        -dataset cub200 \
        -dataroot data \
        -base_mode 'ft_cos' \
        -new_mode 'avg_cos' \
        -gamma 0.25 \
        -lr_base 0.005 \
        -lr_new 0.1 \
        -decay 0.0005 \
        -epochs_base $epochs_base \
        -schedule Cosine \
        -gpu $1 \
        -temperature 16 \
        -batch_size_base 256 \
        -balance 0.01 \
        -loss_iter 0 >> "$LOG_FILE"
done
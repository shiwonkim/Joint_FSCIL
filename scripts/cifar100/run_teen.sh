LOG_DIR="logs/teen"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs_base in 1000
do
    LOG_FILE="$LOG_DIR/CIFAR-TEEN-epochs_${epochs_base}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs_base=$epochs_base"

    python train.py teen \
        -project teen \
        -dataset cifar100 \
        -dataroot data/cifar100 \
        -base_mode 'ft_cos' \
        -new_mode 'avg_cos' \
        -lr_base 0.1 \
        -decay 0.0005 \
        -epochs_base $epochs_base \
        -schedule Cosine \
        -gpu $1 \
        -temperature 16 \
        -softmax_t 16 \
        -shift_weight 0.1 >> "$LOG_FILE"
done
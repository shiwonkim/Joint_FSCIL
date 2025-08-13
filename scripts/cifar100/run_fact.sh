LOG_DIR="logs/fact"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs_base in 800 1000
do
    LOG_FILE="$LOG_DIR/CIFAR-FACT-epochs_${epochs_base}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs_base=$epochs_base"

    python train.py fact \
        -project fact \
        -dataset cifar100 \
        -dataroot data/cifar100 \
        -base_mode 'ft_cos' \
        -new_mode 'avg_cos' \
        -gamma 0.1 \
        -lr_base 0.1 \
        -lr_new 0.1 \
        -decay 0.0005 \
        -epochs_base $epochs_base \
        -schedule Cosine \
        -gpu $1 \
        -temperature 16 \
        -balance 0.001 \
        -loss_iter 0 \
        -alpha 0.5 >> "$LOG_FILE"
done
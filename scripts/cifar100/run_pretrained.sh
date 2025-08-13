LOG_DIR="logs/pretrained"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs in 600 800 1000
do
    LOG_FILE="$LOG_DIR/CIFAR-pretrain-epochs_${epochs}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs=$epochs"

    python train.py joint \
        -project joint \
        -dataset cifar100 \
        -dataroot data/cifar100 \
        -base_mode 'ft_cos' \
        -new_mode 'avg_cos' \
        -gamma 0.1 \
        -lr_base 0.1 \
        -lr_new 0.1 \
        -decay 0.0005 \
        -epochs_base $epochs \
        -epochs_new $epochs \
        -gpu $1 \
        -log_dir train_log \
        -temperature 16 "${@:2}" >> "$LOG_FILE"
done
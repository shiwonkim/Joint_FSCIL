LOG_DIR="logs/pretrained"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs in 600
do
    LOG_FILE="$LOG_DIR/MINI-pretrain-epochs_${epochs}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs=$epochs"

    python train.py joint \
        -project joint \
        -dataset mini_imagenet \
        -dataroot data \
        -base_mode 'ft_cos' \
        -new_mode 'avg_cos' \
        -gamma 0.1 \
        -lr_base 0.1 \
        -lr_new 0.1 \
        -decay 0.0005 \
        -end_session 0 \
        -epochs_base $epochs \
        -epochs_new $epochs \
        -gpu $1 \
        -log_dir train_log \
        -temperature 16 "${@:2}" >> "$LOG_FILE"
done
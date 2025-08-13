LOG_DIR="logs/cec"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs_base in 100
do
    LOG_FILE="$LOG_DIR/CUB-CEC-epochs_${epochs_base}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs_base=$epochs_base"

    python train.py cec \
        -project cec \
        -dataset cub200 \
        -dataroot data \
        -base_mode 'ft_cos' \
        -new_mode 'avg_cos' \
        -episode_way 15 \
        -episode_shot 1 \
        -low_way 15 \
        -low_shot 1 \
        -gamma 0.1 \
        -gamma_meta 0.5 \
        -lr_base 0.1 \
        -lr_base_meta 0.002 \
        -lr_new 0.1 \
        -lrg 0.0002 \
        -decay 0.0005 \
        -epochs_base $epochs_base \
        -schedule Cosine \
        -gpu $1 \
        -temperature 16 >> "$LOG_FILE"
done
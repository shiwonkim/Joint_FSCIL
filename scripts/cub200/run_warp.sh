LOG_DIR="logs/warp"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

epochs_new=30
for epochs_base in 100 200 400 600 800 1000
do
    LOG_FILE="$LOG_DIR/CUB-warp-epochs_${epochs_base}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs_base=$epochs_base"

    python train.py warp \
        -project warp \
        -dataset cub200 \
        -dataroot data/cub200 \
        -base_mode 'ft_dot' \
        -new_mode 'ft_cos' \
        -gamma 0.1 \
        -lr_base 0.1 \
        -lr_new 0.01 \
        -decay 0.0005 \
        -epochs_base $epochs_base \
        -epochs_new $epochs_new \
        -schedule Cosine \
        -fraction_to_keep 0.1 \
        -temperature 16 \
        -gpu $1 >> "$LOG_FILE"
done
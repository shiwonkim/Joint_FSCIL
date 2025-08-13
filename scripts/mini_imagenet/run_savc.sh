LOG_DIR="logs/savc"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs_base in 120
do
    LOG_FILE="$LOG_DIR/MINI-SAVC-epochs_${epochs_base}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs_base=$epochs_base"

    python train.py savc \
        -project savc \
        -dataset mini_imagenet \
        -dataroot data \
        -base_mode 'ft_cos' \
        -new_mode 'avg_cos' \
        -gamma 0.1 \
        -lr_base 0.1 \
        -lr_new 0.1 \
        -decay 0.0005 \
        -epochs_base 120 \
        -schedule Cosine \
        -temperature 16 \
        -moco_dim 128 \
        -moco_k 8192 \
        -mlp \
        -moco_t 0.07 \
        -moco_m 0.999 \
        -size_crops 84 50 \
        -min_scale_crops 0.2 0.05 \
        -max_scale_crops 1.0 0.14 \
        -num_crops 2 4 \
        -constrained_cropping \
        -alpha 0.2 \
        -beta 0.8 \
        -epochs_base $epochs_base \
        -gpu $1 "${@:2}" >> "$LOG_FILE"
done
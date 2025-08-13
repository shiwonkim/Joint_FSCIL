LOG_DIR="logs/savc"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs_base in 50
do
    LOG_FILE="$LOG_DIR/CIFAR-SAVC-epochs_${epochs_base}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs_base=$epochs_base"

    python train.py savc \
        -project savc \
        -dataset cifar100 \
        -dataroot data/cifar100 \
        -base_mode 'ft_cos' \
        -new_mode 'avg_cos' \
        -lr_base 0.1 \
        -lr_new 0.001 \
        -decay 0.0005 \
        -schedule Cosine \
        -temperature 16 \
        -moco_dim 32 \
        -moco_k 8192 \
        -mlp \
        -moco_t 0.07 \
        -moco_m 0.995 \
        -size_crops 32 18 \
        -min_scale_crops 0.9 0.2 \
        -max_scale_crops 1.0 0.7 \
        -num_crops 2 4 \
        -alpha 0.2 \
        -beta 0.8 \
        -constrained_cropping \
        -epochs_base $epochs_base \
        -gpu $1 "${@:2}" >> "$LOG_FILE"
done
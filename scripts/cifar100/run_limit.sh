LOG_DIR="logs/limit"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs_base in 100 200 400 600 800 1000
do
    LOG_FILE="$LOG_DIR/CIFAR-limit-epochs_${epochs_base}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs_base=$epochs_base"

    PRETRAINED_MODEL="./params/pretrain_CIFAR_${epochs_base}.pth"

    python train.py limit \
        -project limit \
        -dataset cifar100 \
        -dataroot data/cifar100 \
        -epochs_base 20 \
        -lr_base 0.0002 \
        -lrg 0.0002 \
        -gamma 0.3 \
        -model_dir $PRETRAINED_MODEL \
        -temperature 16 \
        -schedule Cosine \
        -num_tasks 32 \
        -pretrained_epochs $epochs_base \
        -gpu $1 >> "$LOG_FILE"
done
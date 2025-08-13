LOG_DIR="logs/oracle"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs in 100 200 400 500
do
    LOG_FILE="$LOG_DIR/CIFAR-oracle-epochs_${epochs}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs=$epochs"

    python train.py joint \
        -project joint \
        -dataroot data/cifar100 \
        -dataset cifar100 \
        -epochs_base $epochs \
        -epochs_new $epochs \
        -step 30 \
        -lr_base 0.1 \
        -lr_new 0.1 \
        -schedule Cosine \
        -batch_size_base 128 \
        -cmo \
        -cmo_mixup_prob 0.2 \
        -cmo_end_data_aug 0 \
        -balanced_softmax \
        -imbsam \
        -gpu $1 \
        -log_dir train_log "${@:2}" >> "$LOG_FILE"
done
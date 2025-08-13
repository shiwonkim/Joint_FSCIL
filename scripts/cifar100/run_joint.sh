LOG_DIR="logs/joint"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs_base in 100 200 400 600 800 1000
do
    LOG_SUFFIX="CIFAR-JOINT"
    if [[ "${@:2}" == *"-data_aug"* ]]; then
        LOG_SUFFIX="${LOG_SUFFIX}-data_aug"
    fi
    if [[ "${@:2}" == *"-balanced_loss"* ]]; then
        LOG_SUFFIX="${LOG_SUFFIX}-balanced_loss"
    fi
    if [[ "${@:2}" == *"-batch_prop"* ]]; then
        LOG_SUFFIX="${LOG_SUFFIX}-batch_prop"
    fi
    if [[ "${@:2}" == *"-deepsmote"* ]]; then
        LOG_SUFFIX="${LOG_SUFFIX}-deepsmote"
    fi
    if [[ "${@:2}" == *"-imbsam"* ]]; then
        LOG_SUFFIX="${LOG_SUFFIX}-imbsam"
    fi
    if [[ "${@:2}" == *"-cmo"* ]]; then
        LOG_SUFFIX="${LOG_SUFFIX}-cmo"
    fi
    if [[ "${@:2}" == *"-cmo_mixup"* ]]; then
        LOG_SUFFIX="${LOG_SUFFIX}-mixup"
    fi
    if [[ "${@:2}" == *"-balanced_softmax"* ]]; then
        LOG_SUFFIX="${LOG_SUFFIX}-balanced_softmax"
    fi
    if [[ "${@:2}" == *"-ldam"* ]]; then
        LOG_SUFFIX="${LOG_SUFFIX}-ldam"
    fi
    if [[ "${@:2}" == *"-etf"* ]]; then
        LOG_SUFFIX="${LOG_SUFFIX}-etf"
    fi

    LOG_FILE="$LOG_DIR/${LOG_SUFFIX}-epochs_${epochs}-$(date +%Y%m%d-%H%M%S).txt"
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
        -schedule Cosine \
        -gpu $1 \
        -temperature 16 "${@:2}" >> "$LOG_FILE"
done
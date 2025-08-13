LOG_DIR="logs/oracle"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

EPOCHS=700
LOG_FILE="$LOG_DIR/CUB-oracle-epochs_${EPOCHS}-$(date +%Y%m%d-%H%M%S).txt"
echo "Running with epochs=$EPOCHS"

python train.py joint \
-gpu $1 \
-project joint \
-dataroot data \
-dataset cub200 \
-epochs_base $EPOCHS \
-epochs_new $EPOCHS \
-step 110 \
-lr_base 0.001 \
-lr_new 0.001 \
-schedule Cosine \
-batch_size_base 256 \
-cmo \
-cmo_mixup_prob 0.1 \
-cmo_end_data_aug 0 \
-balanced_softmax \
-imbsam "${@:2}" >> "$LOG_FILE"
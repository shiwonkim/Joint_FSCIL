LOG_DIR="logs/oracle"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

EPOCHS=500
LOG_FILE="$LOG_DIR/MINI-oracle-epochs_${EPOCHS}-$(date +%Y%m%d-%H%M%S).txt"
echo "Running with epochs=$EPOCHS"

python train.py joint \
-gpu $1 \
-project joint \
-dataroot data \
-dataset mini_imagenet \
-epochs_base $EPOCHS \
-epochs_new $EPOCHS \
-step 30 \
-lr_base 0.1 \
-lr_new 0.1 \
-schedule Cosine \
-batch_size_base 128 \
-cmo \
-cmo_mixup_prob 0.2 \
-cmo_end_data_aug 0 \
-balanced_softmax \
-imbsam "${@:2}" >> "$LOG_FILE"
LOG_DIR="logs/joint"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

EPOCHS=300
LOG_FILE="$LOG_DIR/MINI-joint-epochs_${EPOCHS}-$(date +%Y%m%d-%H%M%S).txt"
echo "Running with epochs=$EPOCHS"

python train.py joint \
-gpu $1 \
-project joint \
-dataroot data \
-dataset mini_imagenet \
-epochs_base $EPOCHS \
-epochs_new $EPOCHS \
-step 40 \
-lr_base 0.1 \
-lr_new 0.1 \
-schedule Cosine \
-batch_size_base 128 "${@:2}" >> "$LOG_FILE"
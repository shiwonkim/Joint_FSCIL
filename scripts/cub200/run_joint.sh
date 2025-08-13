LOG_DIR="logs/joint"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

EPOCHS=400
LOG_FILE="$LOG_DIR/CUB-joint-epochs_${EPOCHS}-$(date +%Y%m%d-%H%M%S).txt"
echo "Running with epochs=$EPOCHS"

python train.py joint \
-gpu $1 \
-project joint \
-dataroot data \
-dataset cub200 \
-epochs_base $EPOCHS \
-epochs_new $EPOCHS \
-step 100 \
-lr_base 0.001 \
-lr_new 0.001 \
-schedule Cosine \
-batch_size_base 64 "${@:2}" >> "$LOG_FILE"
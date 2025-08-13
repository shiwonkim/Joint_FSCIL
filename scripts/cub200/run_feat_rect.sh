LOG_DIR="logs/feat_rect"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs_base in 100
do
    LOG_FILE="$LOG_DIR/CIFAR-feat_rect-epochs_${epochs_base}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs_base=$epochs_base"

    PRETRAINED_MODEL="./params/feat_rect/cub_net_0_task_0.pth"

    python train.py feat_rect \
        -image_size 224 \
        -project feat_rect \
        -dataset cub200 \
        -epochs_base $epochs_base \
        -lr_base 0.0002 \
        -lr_new 0.01 \
        -step 20 \
        -gamma 0.5 \
        -gpu $1 \
        -start_session 1 \
        -num_workers 4 \
        -new_mode \
        avg_cos \
        -stage0_chkpt $PRETRAINED_MODEL \
        -epochs_fr 100 \
        -feature_rectification_rkd distance \
        -fr_cos 0.1 \
        -fr_rkd 0 \
        -fr_ce_current 0 \
        -fr_ce_novel 0.5 \
        -batch_size_fr 128 \
        -fr_ce_global 0 \
        -fr_kl 1 -output_blocks 8 9 10 \
        -rkd_intra 1 \
        -rkd_inter 1 \
        -p 32 \
        -k 32 \
        -batch_size_base 128 \
        -rkd_split 'intraI_interI'\
        -eps 1e-5 >> "$LOG_FILE"
  done
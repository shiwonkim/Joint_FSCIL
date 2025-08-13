LOG_DIR="logs/feat_rect"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

for epochs_base in 400 600 800 1000
do
    LOG_FILE="$LOG_DIR/CIFAR-feat_rect-epochs_${epochs_base}-$(date +%Y%m%d-%H%M%S).txt"
    echo "Running with epochs_base=$epochs_base"

    PRETRAINED_MODEL="./params/feat_rect/cifar100/net_0_task_0.pth"

    python train.py feat_rect \
    -project feat_rect \
    -dataroot data/cifar100 \
    -image_size 32 \
    -dataset cifar100 \
    -epochs_base $epochs_base \
    -lr_base 0.0002 \
    -lr_new 0.01 \
    -step 20 \
    -gamma 0.5 \
    -gpu 0 \
    -start_session 1 \
    -num_workers 4 \
    -new_mode \
    avg_cos \
    -stage0_chkpt $PRETRAINED_MODEL \
    -epochs_fr $epochs_base \
    -feature_rectification_rkd distance \
    -fr_cos 0.1 \
    -fr_rkd 0 \
    -fr_ce_current 0 \
    -fr_ce_novel 2 \
    -batch_size_fr 128 \
    -fr_ce_global 0 \
    -fr_kl 0.5 -output_blocks 7 8 9 \
    -rkd_intra 1 \
    -rkd_inter 1 \
    -p 8 \
    -k 16 \
    -batch_size_base 128 \
    -rkd_split 'intraI_interI' >> "$LOG_FILE"
  done
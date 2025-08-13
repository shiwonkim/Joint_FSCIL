epochs_gen=200
epochs_base=100

base_dir="results/cifar100/deepsmote"
data_path="${base_dir}/epochs_${epochs_gen}/upsampled_data"

# Train Generator
python train.py joint \
    -project deepsmote \
    -dataroot data/cifar100 \
    -dataset cifar100 \
    -deepsmote \
    -epochs_gen $epochs_gen \
    -gpu 0

# Generate Images
python train.py joint \
    -project deepsmote \
    -dataroot data/cifar100 \
    -dataset cifar100 \
    -deepsmote_generate \
    -epochs_gen $epochs_gen \
    -gpu 0

# Train Classifier
python train.py joint \
    -project joint \
    -dataroot data/cifar100 \
    -dataset cifar100 \
    -deepsmote \
    -deepsmote_path $data_path \
    -start_session 8 \
    -epochs_base $epochs_base \
    -epochs_new $epochs_base
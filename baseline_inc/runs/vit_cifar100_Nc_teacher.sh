PYTHONPATH=$PYTHONPATH:./baseline_inc

teacher_sizes=(100 200 400 600 800 1000)

base_yaml="./baseline_inc/options/continual_vit_fewshot_base/cifar100_m0_F_baseline_vit_mysmall_patch4_seq_cos_300_purevit_cifar32.yaml"

yaml_dir="./baseline_inc/options/continual_vit_fewshot_base/"

for size in "${teacher_sizes[@]}"; do
    new_yaml="${yaml_dir}cifar100_m0_F_baseline_vit_patch4_${size}.yaml"

    sed "s|teacher_path: .*|teacher_path: ./params/pretrain_CIFAR_${size}.pth|" "$base_yaml" > "$new_yaml"

    echo "Running experiment with teacher_path: ./params/pretrain_CIFAR_${size}.pth"
    PYTHONPATH=$PYTHONPATH:./baseline_inc CUDA_VISIBLE_DEVICES=0 python -m inclearn \
      --options "$new_yaml" \
      ./baseline_inc/options/data/cifar100_fewshot_1orders.yaml \
      --initial-increment 60 \
      --increment 5 \
      --label fewshot_CIFAR100_m0_F_baseline_vit_patch4_${size} \
      --save task \
      --workers $1
done
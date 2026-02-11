# Does Prior Data Matter? Exploring Joint Training in the Context of Few-Shot Class-Incremental Learning

<a href="https://shiwonkim.github.io/">Shiwon Kim</a>\*, <a href="https://dongjunhwang.github.io/">Dongjun Hwang</a>\*, <a href="https://sungwon-woo.github.io/">Sungwon Woo</a>\*, <a href="https://www.cs.cmu.edu/~rsingh/">Rita Singh</a> (\* equal contribution)

This is the official PyTorch implementation of our ICCVW 2025 (<a href="https://sites.google.com/view/clvision2025/overview?authuser=0">CLVision</a>) publication:<br>
"**Does Prior Data Matter? Exploring Joint Training in the Context of Few-Shot Class-Incremental Learning.**" [[paper](https://arxiv.org/abs/2503.10003)]

[Datasets](#datasets) • [Methods Reproduced](#methods-reproduced) • [Experiments](#experiments) • [Citation](#citation)
<br>

<img width="739" height="271" alt="Image" src="https://github.com/user-attachments/assets/a15bfe84-b520-4f6f-af1d-06b013db5cf7" />
<br><br>

> **Abstract**: Class-incremental learning (CIL) aims to adapt to continuously emerging new classes while preserving knowledge of previously learned ones. Few-shot class-incremental learning (FSCIL) presents a greater challenge that requires the model to learn new classes from only a limited number of samples per class. While incremental learning typically assumes restricted access to past data, it often remains available in many real-world scenarios. This raises a practical question: should one retrain the model on the full dataset (i.e., joint training), or continue updating it solely with new data? In CIL, joint training is considered an ideal benchmark that provides a reference for evaluating the trade-offs between performance and computational cost. However, in FSCIL, joint training becomes less reliable due to severe imbalance between base and incremental classes. This results in the absence of a practical baseline, making it unclear which strategy is preferable for practitioners. To this end, we revisit joint training in the context of FSCIL by incorporating imbalance mitigation techniques, and suggest a new imbalance-aware joint training benchmark for FSCIL. We then conduct extensive comparisons between this benchmark and FSCIL methods to analyze which approach is most suitable when prior data is accessible. Our analysis offers realistic insights and guidance for selecting training strategies in real-world FSCIL scenarios.

## Datasets
We use the same data `index_list` suggested in the previous FSCIL work.<br>Please refer to [CEC](https://github.com/icoz69/CEC-CVPR2021?tab=readme-ov-file) for detailed instructions on data preparation.

### Dataset Structure

The codebase expects the datasets to be organized in the following structure:
```
data/
├── cifar100/          # CIFAR-100 dataset files
├── cub200/            # CUB-200 dataset files
├── mini_imagenet/     # miniImageNet dataset files
└── index_list/        # session splits
    ├── cifar100/session_*.txt
    ├── cub200/session_*.txt
    └── mini_imagenet/session_*.txt
```

## Methods Reproduced
The repository features implementations of joint training and eight FSCIL methods in `models/`.

#### Joint Training
- `joint.py`: Joint training baseline with plug-and-play imbalanced learning techniques
- `deepsmote/`: DeepSMOTE - Synthetic minority oversampling for deep neural networks [[paper](https://arxiv.org/abs/2105.02340)]

#### FSCIL
- `cec.py`: CEC - Continually Evolved Classifiers (CVPR 2021) [[paper](https://arxiv.org/abs/2104.03047)]
- `fact.py`: FACT - ForwArd Compatible Training (CVPR 2022) [[paper](https://arxiv.org/abs/2203.06953)]
- `s3c.py`: S3C - Self-Supervised Stochastic Classifier (ECCV 2022) [[paper](https://arxiv.org/abs/2307.02246)]
- `teen.py`: TEEN - Training-frEE calibratioN (NeurIPS 2023) [[paper](https://arxiv.org/abs/2312.05229)]
- `savc.py`: SAVC - Semantic-Aware Virtual Contrastive model (CVPR 2023) [[paper](https://arxiv.org/abs/2304.00426)]
- `limit.py`: LIMIT - LearnIng Multi-phase Incremental Tasks (TPAMI 2023) [[paper](https://arxiv.org/abs/2203.17030)]
- `warp.py`: WaRP - Weight spAce Rotation Process (ICLR 2023) [[paper](https://openreview.net/forum?id=kPLzOfPfA2l)]
- `feat_rect.py`: YourSelf - Feature rectification for ViTs (ECCV 2024) [[paper](https://arxiv.org/abs/2407.07468)]

## Experiments
Examples of model training using `scripts/`.

```bash
# Train CEC on CIFAR-100
bash ./scripts/cifar100/run_cec.sh 0

# Joint training with data augmentation
bash ./scripts/cifar100/run_joint.sh 0 -data_aug

# Train FACT on CUB-200
bash ./scripts/cub200/run_fact.sh 0
```

You can also directly execute `train.py` for individual experiments.
```bash
# CEC
python train.py cec -project cec -dataset cifar100 -dataroot data/cifar100 -base_mode ft_cos -new_mode avg_cos -gpu 0

# Joint training with balanced loss
python train.py joint -project joint -dataset cifar100 -balanced_loss -epochs_base 400 -gpu 0
```

### Key Training Arguments
- `-project`: Specifies the method to be reproduced
- `-dataset`: cifar100, cub200, or mini_imagenet (add `_joint` suffix for joint training)
- `-dataroot`: Path to dataset files
- `-epochs_base`: Number of training epochs for base session (typically 100-1000)
- `-epochs_new`: Number of training epochs for incremental sessions
- `-gpu`: GPU device ID
- `-shot_num`: Number of samples per class in incremental sessions (default: 5)

## Citation
If you find this code useful, please consider citing our paper:
```
@inproceedings{kim2025does,
  title={Does prior data matter? Exploring joint training in the context of few-shot class-incremental learning},
  author={Kim, Shiwon and Hwang, Dongjun and Woo, Sungwon and Singh, Rita},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5185--5194},
  year={2025}
}
```

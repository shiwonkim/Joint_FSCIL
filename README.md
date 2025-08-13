<div align="center">

<h2> Exploring Joint Training in the Context of Few-Shot Class-Incremental Learning</h2>

<div>
 ICCV Workshop on <a href="https://sites.google.com/view/clvision2025/overview?authuser=0"> Continual Learning in Computer Vision </a> (2025)
</div>
</div>

<br>

> **Abstract**: Class-incremental learning (CIL) aims to adapt to continuously emerging new classes while preserving knowledge of previously learned ones. Few-shot class-incremental learning (FSCIL) presents a greater challenge that requires the model to learn new classes from only a limited number of samples per class. While incremental learning typically assumes restricted access to past data, it often remains available in many real-world scenarios. This raises a practical question: should one retrain the model on the full dataset (i.e., joint training), or continue updating it solely with new data? In CIL, joint training serves as an ideal benchmark that offers a reference for evaluating the trade-offs between performance and computational cost. However, in FSCIL, joint training becomes less reliable due to severe class imbalance. This results in the absence of a practical baseline, making it unclear which strategy is preferable for practitioners. To this end, we revisit joint training in the context of FSCIL by integrating imbalance mitigation techniques. Through extensive comparisons with existing FSCIL methods, we analyze which training strategies are suitable when access to prior data is permitted. Our work provides realistic insights and practical guidance for method selection in real-world FSCIL applications.

Official implementation of "Does Prior Data Matters? Exploring Joint Training in the Context of Few-Shot Class-Incremental Learning (ICCVW 2025)".

## Datasets and Pre-trained Model
We follow FSCIL setting to use the same data index_list for training.

Please refer [CEC](https://github.com/icoz69/CEC-CVPR2021?tab=readme-ov-file) for the detailed preparation.

### Dataset Structure

The codebase expects datasets in the following structure:
```
data/
├── cifar100/           # CIFAR-100 dataset files
├── cub200/            # CUB-200 dataset files
├── mini_imagenet/     # Mini-ImageNet dataset files
└── index_list/        # Pre-defined class splits
    ├── cifar100/session_*.txt
    ├── cub200/session_*.txt
    └── mini_imagenet/session_*.txt
```

## Core Architecture

### Main Entry Point
- `train.py`: Unified training script that supports multiple FSCIL methods via project selection
- Uses dynamic module importing to load different method implementations from `models/`

### Model Implementations (`models/`)
The codebase implements various FSCIL methods:
- `cec.py`: CEC - Constrained few-shot class-incremental learning
- `fact.py`: FACT - Few-shot class-incremental learning with transformers
- `teen.py`: TEEN - Transformer-based episodic learning
- `joint.py`: Joint training baseline with imbalance-aware techniques
- `savc.py`: SAVC - Self-supervised augmented view consistency
- `limit.py`: LIMIT - Learning to incrementally learn with minimal trials
- `s3c.py`: S3C - Semantic-aware few-shot continual learning
- `warp.py`: WaRP - Warped feature representation learning
- `feat_rect.py`: YourSelf - Feature rectification approach
- `deepsmote/`: DeepSMOTE data augmentation

### Data Architecture
- `dataloader/`: Dataset-specific loaders for CIFAR-100, CUB-200, Mini-ImageNet
- `data/index_list/`: Pre-defined class splits for few-shot incremental sessions
- Each dataset has session files defining which classes appear in each incremental session

### Base Framework
- `models/base.py`: Abstract `Trainer` class that all methods inherit from
- Provides common functionality: logging, model saving, optimizer setup, session management
- `utils/`: Shared utilities for metrics, losses, feature tools, and helper functions

## Commands

### Training FSCIL Methods
```bash
# Train CEC on CIFAR-100
bash ./scripts/cifar100/run_cec.sh 0

# Train joint baseline with data augmentation
bash ./scripts/cifar100/run_joint.sh 0 -data_aug

# Train FACT method on CUB-200
bash ./scripts/cub200/run_fact.sh 0
```
- You can execute every FSCIL methods and joint training approaches on three benchmark datasets. Please refer `./scripts/` in detail.

### Direct Python Training
```bash
# CEC training
python train.py cec -project cec -dataset cifar100 -dataroot data/cifar100 -base_mode ft_cos -new_mode avg_cos -gpu 0

# Joint training with balanced loss
python train.py joint -project joint -dataset cifar100 -balanced_loss -epochs_base 400 -gpu 0
```

### Key Training Parameters
- `-project`: Selects the FSCIL method (cec, fact, teen, joint, etc.)
- `-dataset`: cifar100, cub200, mini_imagenet (add `_joint` suffix for joint training)
- `-dataroot`: Path to dataset directory
- `-epochs_base`: Training epochs for base session (typically 100-1000)
- `-epochs_new`: Training epochs for incremental sessions
- `-gpu`: GPU device number
- `-shot_num`: Number of shots per class in few-shot sessions (default: 5)

## Citation

TBD

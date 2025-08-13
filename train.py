import argparse
import importlib
import sys

from utils.tools import *

MODEL_DIR=None
DATA_DIR='data/'
PROJECT=None

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # Dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['cifar100', 'cub200', 'mini_imagenet',
                                 'cifar100_joint', 'cub200_joint', 'mini_imagenet_joint'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)

    # Pre-training
    parser.add_argument('-val_ratio', type=float, default=0.1)
    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-optim', type=str, default='SGD', choices=['SGD', 'Adam'])
    parser.add_argument('-schedule', type=str, default='Cosine',
                        choices=['Step', 'Milestone', 'Cosine'])
    
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=20)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='whether to use average data embedding to init or not')

    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set to 0 to use all the availiable training images for new sessions')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos', choices=['ft_dot', 'ft_cos']) # ft_dot for linear classifier, ft_cos for cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos', choices=['ft_dot', 'ft_cos', 'avg_cos']) # avg_cos for average data embedding + cosine classifier

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-end_session', type=int, default=None)
    parser.add_argument('-shot_num', type=int, default=5)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='load model parameters from a specific directory')

    # Training
    parser.add_argument('-gpu', default='0,1')
    parser.add_argument('-num_workers', type=int, default=4)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-no_pin_memory', action='store_true')
    parser.add_argument('-image_size', default=32, type=int)

    # Saving the results
    parser.add_argument('-log_dir', type=str, default='results')

    # Only Inference
    parser.add_argument('-no_training', action='store_true')
    
    return parser


def add_command_line_parser(params):
    project = params[1]
    parser = get_command_line_parser() # base parser
    print(project)
    if project == 'cec':
        parser.add_argument('-lr_base_meta', type=float, default=0.002)
        parser.add_argument('-gamma_meta', type=float, default=0.5)
        
        parser.add_argument('-lrg', type=float, default=0.1, help='learning rate for graph attention network')
        parser.add_argument('-low_shot', type=int, default=1)
        parser.add_argument('-low_way', type=int, default=15)

        # Episode learning
        parser.add_argument('-train_episode', type=int, default=50)
        parser.add_argument('-episode_shot', type=int, default=1)
        parser.add_argument('-episode_way', type=int, default=15)
        parser.add_argument('-episode_query', type=int, default=15)
    
    elif project == 'fact':
        parser.add_argument('-balance', type=float, default=1.0)
        parser.add_argument('-loss_iter', type=int, default=200)
        parser.add_argument('-alpha', type=float, default=2.0)
        parser.add_argument('-eta', type=float, default=0.1)
    
    elif project == 'teen':
        parser.add_argument('-only_do_incre', action='store_true', help='load pre-trained model and skip base training')
        parser.add_argument('-softmax_t', type=float, default=16)
        parser.add_argument('-shift_weight', type=float, default=0.5, help='weights of delta prototypes')
        parser.add_argument('-soft_mode', type=str, default='soft_proto', choices=['soft_proto', 'soft_embed', 'hard_proto'])
        
        # Dataset and network
        parser.add_argument('-feat_norm', action='store_true', help='Set to True to normalize the features')

    elif project == 'savc':
        parser.add_argument('-moco_dim', default=128, type=int,
                            help='feature dimension (default: 128)')
        parser.add_argument('-moco_k', default=65536, type=int,
                            help='queue size; number of negative keys (default: 65536)')
        parser.add_argument('-moco_m', default=0.999, type=float,
                            help='moco momentum of updating key encoder (default: 0.999)')
        parser.add_argument('-moco_t', default=0.07, type=float,
                            help='softmax temperature (default: 0.07)')
        parser.add_argument('-mlp', action='store_true',
                            help='use mlp head')
        parser.add_argument("-num_crops", type=int, default=[2, 4], nargs="+",
                            help="amount of crops")
        parser.add_argument("-size_crops", type=int, default=[224, 96], nargs="+",
                            help="resolution of inputs")
        parser.add_argument("-min_scale_crops", type=float, default=[0.14, 0.05], nargs="+",
                            help="min area of crops")
        parser.add_argument("-max_scale_crops", type=float, default=[1, 0.14], nargs="+",
                            help="max area of crops")
        parser.add_argument('-constrained_cropping', action='store_true',
                            help='condition small crops on key crop')
        parser.add_argument('-auto_augment', type=int, default=[], nargs='+',
                            help='Apply auto-augment 50 % of times to the selected crops')
        parser.add_argument('-alpha', type=float, default=0.5, help='coefficient of the global contrastive loss')
        parser.add_argument('-beta', type=float, default=0.5, help='coefficient of the local contrastive loss')
        parser.add_argument('-incft', action='store_true', help='incrmental finetuning')

    elif project == 'limit':
        parser.add_argument('-meta_class_way', type=int, default=60,
                            help='total classes(including know and unknown) to sample in training process')
        parser.add_argument('-meta_new_class', type=int, default=5)
        parser.add_argument('-num_tasks', type=int, default=256)
        parser.add_argument('-sample_class', type=int, default=16)
        parser.add_argument('-sample_shot', type=int, default=1)

        # SAME with CEC
        parser.add_argument('-lrg', type=float, default=0.1, help='learning rate for graph attention network')
        parser.add_argument('-low_shot', type=int, default=1)
        parser.add_argument('-low_way', type=int, default=15)

        # Episode learning
        parser.add_argument('-train_episode', type=int, default=50)
        parser.add_argument('-episode_shot', type=int, default=1)
        parser.add_argument('-episode_way', type=int, default=15)
        parser.add_argument('-episode_query', type=int, default=15)

        parser.add_argument('-pretrained_epochs', type=int, default=100)

    elif project == 'feat_rect':
        # for param aug
        parser.add_argument('-K', type=int, default=2, help='free lunch k')
        parser.add_argument('-stage0_chkpt', type=str, default='')  # stage0 model is trained in another repo.

        # for feature mapping
        parser.add_argument('-base_fr', type=int, default=1)
        parser.add_argument('-dataset_seed', type=int, default=None) # data_utils.py 
        parser.add_argument('-feature_rectification_rkd', type=str, default='distance')
        parser.add_argument('-lr_fr', type=float, default=0.1)
        parser.add_argument('-milestones_fr', nargs='+', type=int, default=[60, 70, 80, 90])
        parser.add_argument('-gamma_fr', type=float, default=0.1)
        parser.add_argument('-batch_size_fr', type=int, default=256)
        parser.add_argument('-epochs_fr', type=int, default=100)
        parser.add_argument('-iter_fr', type=int, default=50)
        parser.add_argument('-resume_fr', type=str, default=None)

        # feature rectify
        parser.add_argument('-fr_cos', type=float, default=0.1)
        parser.add_argument('-fr_kl', type=float, default=0)
        parser.add_argument('-fr_rkd', type=float, default=1)
        parser.add_argument('-fr_ce_current', type=float, default=1)
        parser.add_argument('-fr_ce_novel', type=float, default=1)
        parser.add_argument('-fr_ce_global', type=float, default=1)
        parser.add_argument('-rkd_inter', type=float, default=0.)
        parser.add_argument('-rkd_intra', type=float, default=0.)
        parser.add_argument('-rkd_intra_extra', type=float, default=0.)
        parser.add_argument('-rkd_inter_extra', type=float, default=0.)

        parser.add_argument('-p', type=int, default=64)
        parser.add_argument('-k', type=int, default=8)

        parser.add_argument('-output_blocks', nargs='+', required=True, default=8, type=int)
        parser.add_argument('-re_cal_rectified_center', default=False, type=bool)
        parser.add_argument('-rkd_split', type=str, default='intraI_interI')
        parser.add_argument('-extra_rkd_split', type=str, default='')
        parser.add_argument('-re_extract_avg', type=bool, default=False)
        parser.add_argument('-eps', type=float, default=1e-6)

    elif project == 's3c':
        parser.add_argument('-not_data_init_new', action='store_true', help='whether to use average data embedding to init or not')

        parser.add_argument('-Glove_vec_dim', type=int, default=100)
        parser.add_argument('-Glove_vec_file', type=str, default='random')
        parser.add_argument('-lamda_semantics', type=float, default=1.0)
        parser.add_argument('-lr_semantic', type=float, default=0.1, help='learning rate for semantic network')
        parser.add_argument('-weight_rho', type=float, default=0.01)
        parser.add_argument('-lamda_dist', type=float, default=1.0)
        parser.add_argument('-lamda_proto', type=float, default=1.0)

    elif project == 'warp':
        parser.add_argument('-rotation', action='store_true')
        parser.add_argument('-fraction_to_keep', type=float, default=0.1)
    
    elif project == 'joint':
        parser.add_argument('-data_aug', action='store_true')
        parser.add_argument('-balanced_loss', action='store_true')
        parser.add_argument('-batch_prop', action='store_true')
        parser.add_argument('-batch_prop_advanced', action='store_true')
        
        # DeepSMOTE
        parser.add_argument('-deepsmote', action='store_true')
        parser.add_argument('-deepsmote_path', type=str, default='./results/deepsmote/upsampled_data')
        parser.add_argument('-deepsmote_generate', action='store_true')
        parser.add_argument('-epochs_gen', type=int, default=200)
        
        # ImbSAM
        parser.add_argument('-imbsam', action='store_true')
        
        # CMO (Cutmix/Mixup)
        parser.add_argument('-cmo', action='store_true')
        parser.add_argument('-cmo_mixup', action='store_true')
        parser.add_argument('-cmo_weighted_alpha', default=1, type=float, help='weighted alpha for sampling probability (q(1,k))')
        parser.add_argument('-cmo_beta', default=1, type=float, help='beta distribution')
        parser.add_argument('-cmo_mixup_prob', default=0.5, type=float, help='mixup probability')
        parser.add_argument('-cmo_start_data_aug', default=3, type=int, help='start epoch for aug')
        parser.add_argument('-cmo_end_data_aug', default=1, type=int, help='when to turn off aug')
        
        # Balanced Softmax
        parser.add_argument('-balanced_softmax', action='store_true')
        
        # LDAM
        parser.add_argument('-ldam', action='store_true')
        parser.add_argument('-ldam_drw', action='store_true')

        # ETF classifier (DR loss)
        parser.add_argument('-etf', action='store_true')
        parser.add_argument('-reg_Ew', type=bool, default=True, help='l2 norm constraints for classifier vector w')
        parser.add_argument('-reg_lam', type=float, default=0.0)

    else:
        raise NotImplementedError
    
    args = parser.parse_args(params[2:])
    
    return args


if __name__ == '__main__':
    args = add_command_line_parser(sys.argv)
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)

    if args.project == 'cec':
        trainer = importlib.import_module('models.cec').CEC(args)
    elif args.project == 'fact':
        trainer = importlib.import_module('models.fact').FACT(args)
    elif args.project == 'teen':
        trainer = importlib.import_module('models.teen').TEEN(args)
    elif args.project == 'warp':
        trainer = importlib.import_module('models.warp').WaRP(args)
    elif args.project == 's3c':
        trainer = importlib.import_module('models.s3c').S3C(args)
    elif args.project == 'savc':
        trainer = importlib.import_module('models.savc').SAVC(args)
    elif args.project == 'limit':
        trainer = importlib.import_module('models.limit').LIMIT(args)
    elif args.project == 'joint':
        trainer = importlib.import_module('models.joint').JointTrainer(args)
    elif args.project == 'deepsmote':
        trainer = importlib.import_module('models.deepsmote.trainer').DeepSMOTE(args)
    elif args.project == 'feat_rect':
        trainer = importlib.import_module('models.feat_rect').FEAT_RECT(args)
    
    trainer.train()

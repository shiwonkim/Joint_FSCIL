import os
import time

import torch
import random
import numpy as np
import pprint as pprint
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict
matplotlib.use('Agg')

from sklearn.metrics import confusion_matrix
from ptflops import get_model_complexity_info


_utils_pp = pprint.PrettyPrinter()

def pprint(x):
    _utils_pp.pprint(x)


class TimeCalculator(object):
    # ms = 1/1000 s
    def __init__(self):
        self.current_iter = 0
        self.mil_sec = 0.
        self.start_event = None
        self.end_event = None

    def time_start(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def time_end(self):
        self.end_event.record()
        self.end_event.synchronize()
        elapsed_time = self.start_event.elapsed_time(self.end_event)
        self.save(elapsed_time)
        return elapsed_time

    def save(self, cur_mil_sec):
        # if self.current_iter == 0:
        #     self.current_iter += 1
        # else:
        self.mil_sec += cur_mil_sec
        # self.current_iter += 1

    def return_total_sec(self):
        return self.mil_sec


def calc_gflops(result_list, model, input_size):
    with torch.no_grad():
        macs, param_size = get_model_complexity_info(model, input_size, as_strings=False,
                                                     print_per_layer_stat=True, verbose=True)
        
        print(f"Computational complexity (GFlops): {macs * 2 / 1000000000} G")
        print(f"Parameter Size (M): {param_size / 1000000}")
        print(f"-------------")

        result_list.append(f"Computational complexity (GFlops): {macs * 2 / 1000000000} G")
        result_list.append(f"Parameter Size (M): {param_size / 1000000}")
    
    return result_list


def set_seed(seed):
    if seed == 0:
        print('random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def count_acc_topk(x, y, k=5):
    _, maxk = torch.topk(x, k, dim=-1)
    total = y.size(0)
    test_labels = y.view(-1,1)
    topk = (test_labels == maxk).sum().item()
    
    return float(topk / total)


def accuracy(output, targets, topk=1):
    """Computes the precision@k for the specified values of k"""
    output, targets = torch.tensor(output), torch.tensor(targets)

    batch_size = targets.shape[0]
    if batch_size == 0:
        return 0.
    nb_classes = len(np.unique(targets))
    topk = min(topk, nb_classes)

    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    correct_k = correct[:topk].reshape(-1).float().sum(0).item()
    
    return round(correct_k / batch_size, 3)


def count_acc_per_task(ypreds, ytrue, init_task_size=0, task_size=10, topk=1):
    """Computes accuracy for the whole test & per task.

    :param ypred: The predictions array.
    :param ytrue: The ground-truth array.
    :param task_size: The size of the task.
    :return: A dictionnary.
    """
    all_acc = {}
    all_acc_list = []

    all_acc["total"] = accuracy(ypreds, ytrue, topk=topk)

    if task_size is not None:

        if init_task_size > 0:
            idxes = np.where(np.logical_and(ytrue >= 0, ytrue < init_task_size))[0]

            label = "{}-{}".format(
                str(0).rjust(2, "0"),
                str(init_task_size - 1).rjust(2, "0")
            )
            acc = accuracy(ypreds[idxes], ytrue[idxes], topk=topk)
            all_acc[label] = acc
            all_acc_list.append(acc * 100)

        for class_id in range(init_task_size, int(np.max(ytrue)) + task_size, task_size):
            if class_id > int(np.max(ytrue)):
                break

            idxes = np.where(np.logical_and(ytrue >= class_id, ytrue < class_id + task_size))[0]

            label = "{}-{}".format(
                str(class_id).rjust(2, "0"),
                str(class_id + task_size - 1).rjust(2, "0")
            )
            acc = accuracy(ypreds[idxes], ytrue[idxes], topk=topk)
            all_acc[label] = acc
            all_acc_list.append(acc * 100)

    return all_acc, all_acc_list


def save_conf_matrix(logits, label, filename):
    pred = torch.argmax(logits, dim=1)
    cm = confusion_matrix(label, pred, normalize='true')
    num_cls = len(cm)

    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    cax = ax.imshow(cm, cmap=plt.cm.jet) 
    
    # Color bar
    cbar = plt.colorbar(cax)
    cbar.ax.tick_params(labelsize=10)
    
    if num_cls <= 100:
        plt.yticks([0,19,39,59,79,99],[0,20,40,60,80,100], fontsize=10)
        plt.xticks([0,19,39,59,79,99],[0,20,40,60,80,100], fontsize=10)
    elif num_cls <= 200:
        plt.yticks([0,39,79,119,159,199],[0,40,80,120,160,200], fontsize=10)
        plt.xticks([0,39,79,119,159,199],[0,40,80,120,160,200], fontsize=10)
    else:
        plt.yticks([0,199,399,599,799,999],[0,200,400,600,800,1000], fontsize=10)
        plt.xticks([0,199,399,599,799,999],[0,200,400,600,800,1000], fontsize=10)
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(filename+'.pdf', bbox_inches='tight')
    plt.savefig(filename+'.png', bbox_inches='tight')
    plt.close()


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()


def get_logger_str(losses: Dict, layer_idx: list, epoch: int):
    str = f'\nEpoch [{epoch}]: '
    for i in layer_idx:
        str += f'\n\tlayer{i}\t'
        cur_loss = losses[f'layer{i}']
        for k, v in cur_loss.items():
            str += f'{k}={v.item():.4f}\t'
    return str
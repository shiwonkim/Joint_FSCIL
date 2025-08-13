import os
import abc
from dataloader.data_utils import *
from utils.tools import (
    Averager, Timer, ensure_path
)

class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.args = set_up_datasets(self.args)
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()
        self.init_log()
        self.result_list = [self.args]

    @abc.abstractmethod
    def train(self):
        pass

    def get_base_optimizer(self, args, model):
        # Set optimizer
        if args.optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), args.lr_base,
                                        momentum=args.momentum, nesterov=True, weight_decay=args.decay)
        elif args.optim == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_base, weight_decay=args.decay)
        
        # Set scheduler
        if args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
        elif args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        elif args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_base)

        return optimizer, scheduler


    def init_log(self):
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] * self.args.sessions
        self.trlog['top5_acc'] = [0.0] * self.args.sessions
        self.trlog['seen_acc'] = [0.0] * self.args.sessions
        self.trlog['unseen_acc'] = [0.0] * self.args.sessions
        self.trlog['g_acc'] = {}
        self.trlog['g_auc'] = [0.0] * self.args.sessions

    
    def log_train_info(self, tl, ta, epoch, lrc, vl=None, va=None):
        self.trlog['train_loss'].append(tl)
        self.trlog['train_acc'].append(ta)
        
        if vl is not None:
            self.trlog['val_loss'].append(vl)
            self.trlog['val_acc'].append(va)
            self.result_list.append(
                'Epoch %d, lr %.4f, train_loss %.5f, train_acc %.5f, val_loss %.5f, val_acc %.5f' % (
                    epoch+1, lrc, tl, ta, vl, va))
        else:
            self.result_list.append(
                'Epoch %d, lr %.4f, train_loss %.5f, train_acc %.5f' % (
                    epoch+1, lrc, tl, ta))

    
    def log_session_info(self, session, acc, g_acc, g_auc):
        if session == 0:
            self.result_list.append('\nBase Session, best val Acc {:.3f} (epoch {})\n'.format(
                self.trlog['max_acc'][session], self.trlog['max_acc_epoch']+1))

        self.trlog['max_acc'][session] = float('%f' % (acc['total'] * 100))
        self.trlog['top5_acc'][session] = float('%f' % (acc['top5'] * 100))
        self.trlog['seen_acc'][session] = float('%f' % (acc['seen'] * 100))
        self.trlog['unseen_acc'][session] = float('%f' % (acc['unseen'] * 100))
        self.trlog['g_acc'][session] = g_acc
        self.trlog['g_auc'][session] = float('%f' % (g_auc))

        self.result_list.append('Session {}, test Acc {:.3f}, Acc@5 {:.3f}'.format(session, self.trlog['max_acc'][session],
                                                                                self.trlog['top5_acc'][session]))
        self.result_list.append('  seen Acc {:.3f}, unseen Acc {:.3f}'.format(self.trlog['seen_acc'][session],
                                                                            self.trlog['unseen_acc'][session]))
        self.result_list.append('  generalized Acc {}'.format(self.trlog['g_acc'][session]))
        self.result_list.append('  generalized AUC {}\n'.format(self.trlog['g_auc'][session]))

    
    def log_final_info(self, total_time, t_end_time_gpu):
        self.result_list.append('Test acc: {}'.format(self.trlog['max_acc']))
        self.result_list.append('Test top-5 acc: {}'.format(self.trlog['top5_acc']))
        self.result_list.append('Seen acc: {}'.format(self.trlog['seen_acc']))
        self.result_list.append('Unseen acc: {}'.format(self.trlog['unseen_acc']))
        self.result_list.append('Generalized AUC: {}'.format(self.trlog['g_auc']))
        
        time_msg = '\nTotal time used %.2f secs (%.2f mins)' % (total_time, total_time / 60)
        gpu_time_msg = 'Total time used %.2f (ms) with TimeCalculator' % t_end_time_gpu

        self.result_list.append(time_msg)
        self.result_list.append(gpu_time_msg)
    
    
    def save_model(self, model, session, save_path, mode=None, optimizer=None):
        if mode == 'joint' and session > 0:
            filename = '_last_acc.pth'
        else:
            filename = '_best_acc.pth'
        
        save_model_dir = os.path.join(save_path, 'session' + str(session) + filename)
        torch.save(dict(params=model.state_dict()), save_model_dir)
        
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(save_path, 'optim_best.pth'))
    
    
    def set_save_path(self, epochs=None):
        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        if self.args.project == 'limit':
            self.args.save_path = self.args.save_path + '-' + str(epochs) + '/'
        
        if self.args.project == 'joint':
            ablations = []
            if self.args.data_aug:
                ablations.append('data_aug')
            if self.args.balanced_loss:
                ablations.append('balanced_loss')
            if self.args.batch_prop:
                ablations.append('batch_prop')
            if self.args.batch_prop_advanced:
                ablations.append('batch_prop_advanced')
            if self.args.deepsmote:
                ablations.append('deepsmote')
            if self.args.imbsam:
                ablations.append('imbsam')
            if self.args.cmo:
                ablations.append('cmo')
            if self.args.balanced_softmax:
                ablations.append('balanced_softmax')
            if self.args.ldam:
                ablations.append('ldam')
            if self.args.cmo_mixup:
                ablations.append('mixup')
            if self.args.ldam_drw:
                ablations.append('drw')
            if self.args.etf:
                ablations.append('etf')
                if self.args.reg_Ew:
                    ablations.append('reg_Ew')

            if epochs is not None:
                ablations.append(str(epochs))
            
            if len(ablations) == 0:
                self.args.save_path = self.args.save_path + 'standard/'
            else:
                self.args.save_path = self.args.save_path + '-'.join(ablations) + '/'

        if self.args.project == 'deepsmote':
            self.args.save_path = self.args.save_path + 'epochs_' + str(self.args.epochs_gen)
        else:
            self.args.save_path = self.args.save_path + 'epochs_' + str(self.args.epochs_base)

        self.args.save_path = os.path.join(self.args.log_dir, self.args.save_path)
        ensure_path(self.args.save_path)
        
        return None

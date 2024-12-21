import torch
import argparse
from pathlib import Path
import numpy as np
import torch.backends.cudnn as cudnn
import yaml
from torch.utils.data import DataLoader
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import pickle
import os
import math
import sys


def import_class(name: str):
    """import class"""
    if name is None:
        raise Exception("Input is None")
    else:
        components = name.split('.')
        mod = __import__(components[0])  # import return model
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)   # 从dim=1维度上选取maxk个值的索引(按值大小排序)  (b, maxk)
    pred = pred.t()   # (maxk, b)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_args_parser():
    parser = argparse.ArgumentParser("Skeleton MAE Fine", add_help=False)

    parser.add_argument("--result_dir", default="./result/fine", 
                        type=str, help="path where to save, empty for no saving")
    parser.add_argument("--config", default="./config/fine.yaml", 
                        type=str, help="Path to the configuration file")

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')


    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model_args', default=dict(), help='the arguments of model')
    parser.add_argument("--pretrain_weights", default=None, type=str, help=" the weights of pretrain")
    parser.add_argument("--frozen", default=False, type=bool, help="If False, the moduel train by end to end")

    # training
    parser.add_argument("--accumulated_size", default=128, type=int, help="The accumulated size for train")
    parser.add_argument("--train_size", default=16, type=int,
                        help="Batch size GPU or CPU")
    parser.add_argument("--val_size", default=16, type=int,
                        help="the batch size of validation on GPU or CPU")
    
    # set check point
    parser.add_argument("--start_epoch", type=int, default=0, help="start training from which epoch")
    parser.add_argument("--epochs", default=100, type=int, help="stop training in which epochs")
    parser.add_argument("--check_weights", default=None, help="the weights for newwork initialization")

    parser.add_argument("--seed", default=1, type=int)

    # optimizer
    parser.add_argument("--optimizer", default="Adam", type=str, help="type of optimizer")
    parser.add_argument('--base_lr', type=float, default=0.005, help='initial learning rate')
    parser.add_argument('--nesterov', type=bool, default=False, help='use nesterov or not')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
    parser.add_argument('--step', type=int, default=[60, 90, 110], help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=5)

    parser.add_argument("--device", default=0, help="device to use for training or testing")
    parser.add_argument("--num_workers", default=2, help="the number of worker for data loader")

    parser.add_argument("--print_log", default=True, type=bool, help="print log or not")

    return parser


class AverageMeter(object):
    """Compute and stores the average and current value"""
    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class Processor(object):
    def __init__(self, args):
        self.args = args

        self.data_loader = None
        self.val_data_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss = None

        self.view_index = None

        # fix seed
        self.init_seed()
        self.gpu()
        self.load_model()
        self.load_data()
        self.load_optimizer()

        # # predict
        self.best_acc = 0
        self.best_acc_epoch = 0

    def init_seed(self):
        torch.cuda.manual_seed_all(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def gpu(self):
        if type(self.args.device) != str:
            self.device = torch.device(type='cuda', index=self.args.device)
        else:
            self.device = torch.device("cpu")
        self.print_log("Using device {}".format(self.device))

    def load_data(self):
        feeder = import_class(self.args.feeder)

        self.data_loader = DataLoader(
            dataset=feeder(**self.args.train_feeder_args),
            shuffle=True,
            batch_size=self.args.train_size,
            drop_last=True,
            pin_memory=True,
            num_workers=self.args.num_workers
        )
        self.val_data_loader = DataLoader(
            dataset=feeder(**self.args.test_feeder_args),
            shuffle=False,
            batch_size=self.args.val_size,
            drop_last=False,
            pin_memory=True,
            num_workers=self.args.num_workers
        )

        self.print_log("Data load finished")

    def load_model(self):
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='sum')

        model = import_class(self.args.model)
        self.model = model(**self.args.model_args)
                
        if self.args.check_weights is not None:
            self.print_log("Load weights from {}".format(self.args.check_weights))
            # load check point weight
            if '.pkl' in self.args.check_weights:
                with open(self.args.check_weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.args.check_weights)
            self.model.load_state_dict(weights)
        elif self.args.pretrain_weights is not None:
            # load encoder parameters
            try:
                self.model.load_state_dict(torch.load(self.args.pretrain_weights)['net'], strict=False)
                self.print_log("Load pretrain weights from {}".format(self.args.pretrain_weights))
            except ValueError:
                print("The weights of pretrain is None, must be give a string value")
        else:
            self.print_log("No pre-training weights for fine-tuned models")
        
        if self.args.frozen:
            for name, param in self.model.named_parameters():
                if name.split('.')[0] not in ["fc"]:
                    param.requires_grad = False

        self.model.to(self.device)
        self.print_log("Model load finished " + self.args.model)

    def load_optimizer(self):
        if self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr,
                betas=(0.9, 0.95),
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.base_lr,
                weight_decay=self.args.weight_decay)
        else:
            raise ValueError()
        # 动态学习率
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.step, gamma=0.1)
        self.print_log('Optimizer {} load finished'.format(self.args.optimizer))

    def adjust_learning_rate(self, epoch):
        if self.args.optimizer == 'SGD' or self.args.optimizer == 'Adam': 
            if epoch < self.args.warm_up_epoch:
                lr = self.args.base_lr * (epoch + 1) / self.args.warm_up_epoch # linear scaling rule
            else:
                lr = self.args.base_lr * (self.args.lr_decay_rate ** np.sum(epoch >= np.array(self.args.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr  # step rule for SGD and Adam
            return lr
        elif self.args.optimizer == 'AdamW':
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.99 
        else:
            raise ValueError()    # not implemented for other optimizers

    def print_log(self, s: str, print_time=True):
        if print_time:
            localtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
            s = "[ " + localtime + ' ] ' + s
        print(s)
        if self.args.print_log:
            with open(f'{self.args.result_dir}/log.txt', 'a') as file:
                print(s, file=file)

    def train_one_epoch(self, epoch: int, writer):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.train()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        accumulated_steps = self.args.accumulated_size // self.args.train_size if self.args.accumulated_size > self.args.train_size else 1

        process = enumerate(tqdm(self.data_loader, desc='Train', colour='green'))
        for idx, (data, label, _) in process:
            with torch.no_grad():
                data = torch.as_tensor(data).float().to(self.device)
                label = torch.as_tensor(label).long().to(self.device)
            
            pre = self.model(data)
            loss = self.loss(pre, label)

            loss_value = loss.data.item()
            losses.update(loss_value)
            if not math.isfinite(loss_value):
                print("Loss is {}, stoping training".format(loss_value))
                sys.exit(2)
            prec1, prec5 = accuracy(pre.detach(), label, topk=(1, 5))          

            top1.update(prec1.data.item(), data.size(0))
            top5.update(prec5.data.item(), data.size(0))

            loss = loss / accumulated_steps
            loss.backward()
            if ((idx+1) % accumulated_steps == 0) or ((idx+1) == len(self.data_loader)):
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.adjust_learning_rate(epoch)
        
        if writer is not None:
            writer.add_scalar('acc1', top1.avg, epoch)
            writer.add_scalar('acc5', top5.avg, epoch)
        
        self.print_log("Epoch: {}, LR: {}".format(epoch+1, lr))
        self.print_log("Top1: {:.2f}%, Top5: {:.2f}%".format(top1.avg, top5.avg))
        self.print_log('Avg Loss: {:.4f}'.format(losses.avg))

        return top1.avg

    @torch.no_grad()
    def val_one_epoch(self, epoch: int, writer):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        self.model.eval()
        process = tqdm(self.val_data_loader, desc='Valid', colour='blue')
        for data, label, _ in process:
            data = torch.as_tensor(data).float().to(self.device)
            label = torch.as_tensor(label).long().to(self.device)
            pre = self.model(data)
            loss = self.loss(pre, label)
            losses.update(loss.data.item())

            prec1, prec5 = accuracy(pre.detach(), label, topk=(1, 5))
            top1.update(prec1.data.item(), data.size(0))
            top5.update(prec5.data.item(), data.size(0))
    
        # update best accuracy
        if top1.avg > self.best_acc:
            self.best_acc = top1.avg
            self.best_acc_epoch = epoch+1
        
        if writer is not None:
            writer.add_scalar('acc1', top1.avg, epoch)
            writer.add_scalar('acc5', top5.avg, epoch)

        self.print_log('Top1: {:.2f}%, Top5: {:.2f}%'.format(top1.avg, top5.avg))
        self.print_log('Evaluating: loss: {:.4f}'.format(losses.avg))
        self.print_log("Best acc: {:.2f}%, at in {} epochs".format(self.best_acc, self.best_acc_epoch))

        return top1.avg

    def start(self):
        self.print_log("------------------- Config Parameters --------------------------")
        for argument, value in sorted(vars(self.args).items()):
            self.print_log(f"{argument}: {value}")
        self.print_log("------------------- Model Parameters --------------------------")

        save_path = self.args.result_dir
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        train_writer = SummaryWriter(os.path.join(self.args.result_dir, 'train'), 'train')
        val_writer = SummaryWriter(os.path.join(self.args.result_dir, 'val'), 'val')

        def count_parameters(model):
            # numel():获取总共有多少个元素
            return sum(pa.numel() for pa in model.parameters() if pa.requires_grad)
        self.print_log("Model Total Parameters: {:.3f} M".format(count_parameters(self.model) / 1e6))

        self.print_log("--------------------- Start Training ----------------------")
        val_acc = 0
        for epoch in range(self.args.start_epoch, self.args.epochs):
            _ = self.train_one_epoch(epoch=epoch, writer=train_writer)
            val_accuary = self.val_one_epoch(epoch=epoch, writer=val_writer)
            if epoch == self.args.start_epoch:
                val_acc = val_accuary
            else:
                if val_accuary > val_acc:
                    val_acc = val_accuary
                    state = {'net': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
                    torch.save(state, save_path + "/val_weight_" + str(epoch + 1) + ".pth")
            # last epoch to save model
            if (epoch+1) % 10 == 0:
                state = {'net': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, save_path+"/weight_"+str(epoch+1)+".pth")
        
        self.print_log("-----------------------------------------------------")
        self.print_log("Fally, the test acc: {}".format(self.best_acc))


if __name__ == '__main__':
    p = get_args_parser()
    arg = p.parse_args()

    # load arg from config file
    if arg.config is not None:
        with open(arg.config, 'r', encoding="utf-8") as f:
            default_arg = yaml.load(f, yaml.FullLoader)
        key = vars(arg).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        p.set_defaults(**default_arg)

    arg = p.parse_args()

    # # Crate result dir
    if arg.result_dir:
        Path(arg.result_dir).mkdir(parents=True, exist_ok=True)

    processor = Processor(arg)
    processor.start()

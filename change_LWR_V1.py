import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torch import Tensor
from typing import Tuple

import models
import utils
from dataset.dataloader_LWR import dataloader
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from tensorboardX import SummaryWriter


class LWR(torch.nn.Module):
    def __init__(self, k: int, num_batches_per_epoch: int, dataset_length: int, output_shape: Tuple[int],
                 max_epochs: int, tau=3., update_rate=0.9, softmax_dim=1):
        super().__init__()
        self.k = k
        self.update_rate = update_rate
        self.max_epochs = max_epochs

        self.num_batches_per_epoch = num_batches_per_epoch
        self.tau = tau  # 温度系数
        self.alpha = 1.

        self.softmax_dim = softmax_dim
        self.labels = torch.zeros((dataset_length, *output_shape))  # 参数前面加*代表解压参数列表

    def forward(self, batch_idx: Tensor, logits: Tensor, y_true: Tensor, cur_epoch: int):
        self.alpha = 1 - self.update_rate * (cur_epoch - cur_epoch % self.k) / self.max_epochs  # 交叉熵loss前面的系数
        if cur_epoch <= self.k:
            return F.cross_entropy(logits, y_true), torch.tensor(0)
        else:
            return self.loss_fn_with_kl(logits, y_true, batch_idx)

    def loss_fn_with_kl(self, logits: Tensor, y_true: Tensor, batch_idx: Tensor):
        loss1 = self.alpha * F.cross_entropy(logits, y_true)
        loss2 = (1 - self.alpha) * self.tau ** 2 * F.kl_div(F.log_softmax(logits / self.tau, dim=self.softmax_dim),
                                                            self.labels[batch_idx, ...].to(logits.get_device()),
                                                            reduction='batchmean')
        return loss1, loss2


parser = argparse.ArgumentParser(description='PyTorch LWR Example')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--dataset", type=str, default="CIFAR100")
parser.add_argument("--outdir", type=str, default="save_LWR")
parser.add_argument("--model", type=str, default="vgg19")
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--dropout', default=0., type=float, help='Input the dropout rate: default(0.0)')
parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                    help='Learning rate step gamma (default: 0.1)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--temp', default=3.0, type=float, help='temperature scaling')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True  # 对于固定不变的网络可以起到优化作用
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"


def train(model, train_loader, optimizer, lwr, cur_epoch):
    model.train()
    loss_avg = utils.AverageMeter()
    accTop1_avg = utils.AverageMeter()
    accTop5_avg = utils.AverageMeter()
    loss_kd = utils.AverageMeter()
    loss_label = utils.AverageMeter()
    end = time.time()

    with tqdm(total=len(train_loader)) as t:
        for i, (batch_idx, data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            if cur_epoch % args.k == 0:
                lwr.labels[batch_idx, ...] = F.softmax(output / args.temp, dim=1).detach().clone().cpu()
            loss1, loss2 = lwr(batch_idx, output, target, cur_epoch)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = utils.accuracy(output, target, topk=(1, 5))
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
            loss_avg.update(loss.item())
            loss_kd.update(loss2.item())
            loss_label.update(loss1.item())

            t.update()

    train_metrics = {'train_loss': loss_avg.value(),
                     'train_accTop1': accTop1_avg.value(),
                     'train_accTop5': accTop5_avg.value(),
                     'kd_loss': loss_kd.value(),
                     'label_loss': loss_label.value(),
                     'time': time.time() - end}

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
    logging.info("- Train metrics: " + metrics_string)
    return train_metrics


def evaluate(model, test_loader):
    model.eval()
    loss_avg = utils.AverageMeter()
    accTop1_avg = utils.AverageMeter()
    accTop5_avg = utils.AverageMeter()
    end = time.time()

    with torch.no_grad():
        for _, data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='mean')
            loss_avg.update(loss.item())
            metrics = utils.accuracy(output, target, topk=(1, 5))
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())

    test_metrics = {'test_loss': loss_avg.value(),
                    'test_accTop1': accTop1_avg.value(),
                    'test_accTop5': accTop5_avg.value(),
                    'time': time.time() - end}

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in test_metrics.items())
    logging.info("- Test  metrics: " + metrics_string)
    return test_metrics


if __name__ == '__main__':
    begin_time = time.time()

    utils.solve_dir(os.path.join(args.outdir, args.model, 'save_model'))
    utils.solve_dir(os.path.join(args.outdir, args.model, 'log'))

    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    utils.set_logger(os.path.join(args.outdir, args.model, 'log', now_time + 'train.log'))

    w = vars(args)
    metrics_string = " ;\n".join("{}: {}".format(k, v) for k, v in w.items())
    logging.info("- All args are followed:\n" + metrics_string)

    logging.info("Loading the datasets...")

    if args.dataset == 'CIFAR10':
        num_classes = 10
        model_folder = "model_cifar"
        root = './Data'
    elif args.dataset == 'CIFAR100':
        num_classes = 100
        model_folder = "model_cifar"
        root = './Data'
    elif args.dataset == 'imagenet':
        num_classes = 1000
        model_folder = "model_imagenet"
        root = './Data'

    # Load data
    train_loader, test_loader, dataset_len = dataloader(data_name=args.dataset, batch_size=args.batch_size, root=root)
    logging.info("- Done.")

    model_fd = getattr(models, model_folder)
    if "resnet" in args.model:
        model_cfg = getattr(model_fd, 'resnet')
        model = getattr(model_cfg, args.model)(num_classes=num_classes)
    elif "vgg" in args.model:
        model_cfg = getattr(model_fd, 'vgg')
        model = getattr(model_cfg, args.model)(num_classes=num_classes, dropout=args.dropout)
    elif "densenet" in args.model:
        model_cfg = getattr(model_fd, 'densenet')
        model = getattr(model_cfg, args.model)(num_classes=num_classes)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    num_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
    logging.info('Total params: %.2fM' % num_params)

    lwr = LWR(k=args.k, update_rate=0.9, num_batches_per_epoch=dataset_len // args.batch_size,
              dataset_length=dataset_len,
              output_shape=(num_classes,), tau=args.temp, max_epochs=args.num_epochs, softmax_dim=1)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.wd)
    scheduler = MultiStepLR(optimizer, milestones=[150, 225], gamma=args.gamma, verbose=True)
    best_acc = 0

    writer = SummaryWriter(log_dir=os.path.join(args.outdir, args.model))

    for i in range(args.num_epochs):
        logging.info("Epoch {}/{}".format(i + 1, args.num_epochs))
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], i + 1)

        train_metrics = train(model, train_loader, optimizer, lwr, i + 1)
        writer.add_scalar('Train/Loss', train_metrics['train_loss'], i + 1)
        writer.add_scalar('Train/AccTop1', train_metrics['train_accTop1'], i + 1)
        writer.add_scalar('Train/label_loss', train_metrics['label_loss'], i + 1)
        writer.add_scalar('Train/kd_loss', train_metrics['kd_loss'], i + 1)

        test_metrics = evaluate(model, test_loader)
        writer.add_scalar('Test/Loss', test_metrics['test_loss'], i + 1)
        writer.add_scalar('Test/AccTop1', test_metrics['test_accTop1'], i + 1)
        writer.add_scalar('Test/AccTop5', test_metrics['test_accTop5'], i + 1)

        test_acc = test_metrics['test_accTop1']
        save_dic = {'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'epoch': i + 1,
                    'test_accTop1': test_metrics['test_accTop1'],
                    'test_accTop5': test_metrics['test_accTop5']}
        last_path = os.path.join(args.outdir, args.model, 'save_model', 'last_model.pth')
        # torch.save(save_dic, last_path)
        if test_acc >= best_acc:
            logging.info("- Found better accuracy")
            best_acc = test_acc
            best_path = os.path.join(args.outdir, args.model, 'save_model', 'best_model.pth')
            # torch.save(save_dic, last_path)
        scheduler.step()

    writer.close()
    logging.info("best_acc is {}".format(best_acc))
    logging.info('Total time: {:.2f} minutes'.format((time.time() - begin_time) / 60.0))
    logging.info('All tasks have been done!')

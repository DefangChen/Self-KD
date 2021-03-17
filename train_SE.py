import os
import argparse
import shutil
from math import pi
from math import cos

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import models
from dataset import data_loader
import utils
import time
from tqdm import tqdm
import logging
import glob

parser = argparse.ArgumentParser(description='PyTorch Snapshot Ensemble')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--outdir', default='save_SE', type=str)
parser.add_argument('--arch', type=str, default='densenetd40k12',
                    help='models architecture')
parser.add_argument('--dataset', '-d', type=str, default='CIFAR100',
                    help='dataset choice')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4 )')
parser.add_argument('--num_epochs', default=300, type=int,
                    help='number of total iterations')
parser.add_argument('--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--init_lr', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--cycles', default=6, type=int)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=5e-4, type=float,
                    help='weight decay (default: 1e-4)')
# dest表示参数的别名
parser.add_argument('--resume', default='', type=str,
                    help='path to  latest checkpoint (default: None)')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"


def proposed_lr(initial_lr, iteration, epoch_per_cycle):
    return initial_lr * (cos(pi * iteration / epoch_per_cycle) + 1) / 2


def train(model, cur_epoch, optimizer, criterion, train_loader, epoch_per_cycle):
    model.train()
    loss_avg = utils.AverageMeter()
    accTop1_avg = utils.AverageMeter()
    accTop5_avg = utils.AverageMeter()
    end = time.time()

    lr = proposed_lr(args.init_lr, cur_epoch, epoch_per_cycle)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    with tqdm(total=len(train_loader)) as t:
        for _, (train_batch, labels_batch) in enumerate(train_loader):
            train_batch = train_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)

            output_batch = model(train_batch)
            loss = criterion(output_batch, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = utils.accuracy(output_batch, labels_batch, topk=(1, 5))  # metircs代表指标
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
            loss_avg.update(loss.item())

            t.update()

    train_metrics = {'train_loss': loss_avg.value(),
                     'train_accTop1': accTop1_avg.value(),
                     'train_accTop5': accTop5_avg.value(),
                     'time': time.time() - end}

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
    logging.info("- Train metrics: " + metrics_string)
    return train_metrics


def evaluate(test_loader, model, criterion):
    model.eval()
    loss_avg = utils.AverageMeter()
    accTop1_avg = utils.AverageMeter()
    accTop5_avg = utils.AverageMeter()
    end = time.time()

    with torch.no_grad():
        for test_batch, labels_batch in test_loader:
            test_batch = test_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)

            output_batch = model(test_batch)
            loss = criterion(output_batch, labels_batch)

            metrics = utils.accuracy(output_batch, labels_batch, topk=(1, 5))
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
            loss_avg.update(loss.item(), test_batch.size(0))

        # compute mean of all metrics in summary
    test_metrics = {'test_loss': loss_avg.value(),
                    'test_accTop1': accTop1_avg.value(),
                    'test_accTop5': accTop5_avg.value(),
                    'time': time.time() - end}

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in test_metrics.items())
    logging.info("- Test  metrics: " + metrics_string)
    return test_metrics


# Model需要把创建模型的class名称传入
def test_se(Model, weights, use_model_num, test_loader, criterion):
    index = len(weights) - use_model_num  # 取最后use_model_num个模型
    weights = weights[index:]
    model_list = [Model() for _ in weights]  # 先初始化新的模型

    loss_avg = utils.AverageMeter()
    accTop1_avg = utils.AverageMeter()
    accTop5_avg = utils.AverageMeter()
    end = time.time()

    for model, weight in zip(model_list, weights):
        model.load_state_dict(weight)
        model.eval()
        model = nn.DataParallel(model).to(device)

    for data, target in test_loader:
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        output_list = [model(data).unsqueeze(0) for model in model_list]
        # 按照第0维度进行拼接；求取平均值；消除第0维度
        output = torch.mean(torch.cat(output_list), 0).squeeze()
        loss = criterion(output, target)
        loss_avg.update(loss.item(), data.size(0))
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
    # save_path = args.save_path = os.path.join(args.save_folder, args.arch)
    if not os.path.exists(args.outdir):
        print("Directory does not exist! Making directory {}".format(args.outdir))
        os.makedirs(args.outdir)
        os.makedirs(args.outdir + "/save_snapshot")
        os.makedirs(args.outdir + "/log")

    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    utils.set_logger(os.path.join(args.outdir, 'log', now_time + 'train.log'))

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
    train_loader, test_loader = data_loader.dataloader(data_name=args.dataset, batch_size=args.batch_size, root=root)
    logging.info("- Done.")

    temp_dic = {'num_classes': num_classes}
    if args.arch == "densenetd40k12":
        model = models.densenetd40k12(**temp_dic)
        Model = models.densenetd40k12
    elif args.arch == "densenetd100k12":
        model = models.densenetd100k12(**temp_dic)
        Model = models.densenetd100k12
    elif args.arch == "densenetd190k12":
        model = models.densenetd190k12(**temp_dic)
        Model = models.densenetd190k12
    elif args.arch == "densenetd100k40":
        model = models.densenetd100k40(**temp_dic)
        Model = models.densenetd100k40
    else:
        raise Exception("Invalid network!")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    num_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
    logging.info('Total params: %.2fM' % num_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum, nesterov=True, weight_decay=args.wd)
    snapshots = []
    epochs_per_cycle = args.num_epochs // args.cycles  # 一个学习率调整周期 多少个epochs

    for i in range(args.cycles):
        logging.info("The {} cycle starts".format(i + 1))
        for j in range(epochs_per_cycle):
            train(model, j, optimizer, criterion, train_loader, epochs_per_cycle)
            evaluate(test_loader, model, criterion)
        snapshots.append(model.state_dict())
        last_path = os.path.join(args.outdir, "save_snapshot", 'cycle'+str(i)+' model.pth')
        torch.save({'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'epoch': epochs_per_cycle*i + j,
                    }, last_path)

    test_se(Model, snapshots, use_model_num=5, test_loader=test_loader)
    logging.info('Total time: {:.2f} minutes'.format((time.time() - begin_time) / 60.0))
    logging.info('All tasks have been done!')

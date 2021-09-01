"""
nohup python train_CSKD.py --gpu 0 --model resnet32 > CSKD_resnet32.out 2>&1 &
nohup python train_CSKD.py --gpu 0 --model wide_resnet20_8 > CSKD_resnet32.out 2>&1 &
"""

import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import models
import utils
from tensorboardX import SummaryWriter
from dataset.dataloader_CSKD import load_dataset

parser = argparse.ArgumentParser(description='Pytorch CSKD')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--dataset", type=str, default="CIFAR100")
parser.add_argument("--outdir", type=str, default="save_CSKD_model")
parser.add_argument("--model", type=str, default="resnet32")
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--dropout', default=0., type=float, help='Input the dropout rate: default(0.0)')
parser.add_argument('--cls', '-cls', action='store_true', default=True, help='adding cls loss')
parser.add_argument('--temp', default=3.0, type=float, help='temperature scaling')
parser.add_argument('--lamda', default=1.0, type=float, help='cls loss weight ratio')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True  # 对于固定不变的网络可以起到优化作用
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"


# 与model的用法相同，重载了forward函数
class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super().__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input / self.temp_factor, dim=1)  # 学生
        q = torch.softmax(target / self.temp_factor, dim=1)  # 教师
        loss = self.kl_div(log_p, q) * (self.temp_factor ** 2) / input.size(0)
        return loss


def train(train_loader, model, optimizer, criterion1, criterion_kd):
    model.train()
    loss_avg = utils.AverageMeter()
    accTop1_avg = utils.AverageMeter()
    accTop5_avg = utils.AverageMeter()
    loss_kd = utils.AverageMeter()
    label_loss = utils.AverageMeter()
    end = time.time()

    with tqdm(total=len(train_loader)) as t:
        for i, (train_batch, labels_batch) in enumerate(train_loader):
            train_batch = train_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)
            batch_size = train_batch.size(0)
            # 取minibatch当中的前一半sample作为与label求loss的样本
            targets_ = labels_batch[:batch_size // 2]
            outputs = model(train_batch[:batch_size // 2])

            loss = criterion1(outputs, targets_)
            label_loss.update(loss.item())

            with torch.no_grad():
                outputs_cls = model(train_batch[batch_size // 2:])
            cls_loss = criterion_kd(outputs, outputs_cls.detach())
            loss += args.lamda * cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_kd.update(args.lamda * cls_loss.item())
            loss_avg.update(loss.item())
            metrics = utils.accuracy(outputs, targets_, topk=(1, 5))  # metircs代表指标
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())

            t.update()

    train_metrics = {'train_loss': loss_avg.value(),
                     'loss_kd': loss_kd.value(),
                     'label_loss': label_loss.value(),
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
            # only one element tensors can be converted to Python scalars
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
            loss_avg.update(loss.item())

    # compute mean of all metrics in summary
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
    logging.info("- All args are followed: \n" + metrics_string)

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

    if not args.cls:
        train_loader, test_loader = load_dataset(args.dataset, root, batch_size=args.batch_size)
    else:
        train_loader, test_loader = load_dataset(args.dataset, root, 'pair', batch_size=args.batch_size)
    logging.info("- Done.")

    model_fd = getattr(models, model_folder)
    if "resnet" in args.model or "ResNet" in args.model:
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

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = KDLoss(args.temp)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    best_acc = 0
    writer = SummaryWriter(log_dir=os.path.join(args.outdir, args.model))
    for i in range(args.num_epochs):
        logging.info("Epoch {}/{}".format(i + 1, args.num_epochs))
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], i + 1)

        train_metrics = train(train_loader, model, optimizer, criterion1, criterion2)
        writer.add_scalar('Train/Loss', train_metrics['train_loss'], i + 1)
        writer.add_scalar('Train/AccTop1', train_metrics['train_accTop1'], i + 1)
        writer.add_scalar('Train/AccTop5', train_metrics['train_accTop5'], i + 1)
        writer.add_scalar('Train/kd_loss', train_metrics['loss_kd'], i + 1)
        writer.add_scalar('Train/label_loss', train_metrics['label_loss'], i + 1)

        test_metrics = evaluate(test_loader, model, criterion1)
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
            # torch.save(save_dic, best_path)
        scheduler.step()

    writer.close()
    logging.info("best_acc is {}".format(best_acc))
    logging.info('Total time: {:.2f} minutes'.format((time.time() - begin_time) / 60.0))
    logging.info('All tasks have been done!')

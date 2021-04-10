import argparse
import copy
import logging
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import models
import utils
from dataset import data_loader
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Snapshot Ensemble')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--outdir', default='save_SD', type=str)
parser.add_argument('--arch', type=str, default='resnet32',
                    help='models architecture')
parser.add_argument('--dataset', '-d', type=str, default='CIFAR100')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4 )')
parser.add_argument('--num_epochs', default=300, type=int,
                    help='number of total iterations')
parser.add_argument('--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--init_lr', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=0.0001, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--T', type=float, default=3,
                    help='distillation temperature (default: 1)')
parser.add_argument('--lambda_s', type=float, default=1)
parser.add_argument('--lambda_t', type=float, default=1)
parser.add_argument('--cycle', type=float, default=4)
parser.add_argument('--dropout', default=0., type=float, help='Input the dropout rate: default(0.0)')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"


# iteration代表第几次迭代；iteration_per_cycle代表每个mini_gen有多少个迭代
def lr_snapshot(iteration, iteration_per_cycle, initial_lr=args.init_lr, lr_method='cosine'):
    is_snapshot = False
    if lr_method == 'cosine':
        x = math.pi * ((iteration - 1) % iteration_per_cycle / iteration_per_cycle)
        lr = initial_lr / 2 * (math.cos(x) + 1)
    elif lr_method == 'linear':
        t = ((iteration - 1) % iteration_per_cycle + 1) / iteration_per_cycle
        lr = (1 - t) * initial_lr + t * 0.001
    else:
        pass
    if (iteration - 1) % iteration_per_cycle == iteration_per_cycle - 1:
        is_snapshot = True
    return lr, is_snapshot


def train(train_loader, model, optimizer, teacher, cur_epoch, T, iteration_per_epoch, iteration_per_cycle):
    model.train()

    loss_avg = utils.AverageMeter()
    accTop1_avg = utils.AverageMeter()
    accTop5_avg = utils.AverageMeter()
    loss_avgkd = utils.AverageMeter()
    end = time.time()

    cur_iter = cur_epoch * iteration_per_epoch
    with tqdm(total=len(train_loader)) as t:
        for i, (train_batch, labels_batch) in enumerate(train_loader):
            cur_iter += 1
            lr, is_snapshot = lr_snapshot(cur_iter, iteration_per_cycle)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            train_batch = train_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)

            output_batch = model(train_batch)
            loss1 = criterion(output_batch, labels_batch)

            if teacher is not None:
                teacher.eval()
                with torch.no_grad():
                    output_teacher = teacher(train_batch).detach()
                # 除去了学生的温度系数
                loss2 = (- F.log_softmax(output_batch, 1) * F.softmax(output_teacher / T, dim=1)).sum(dim=1).mean()
                # loss_kd = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output_batch/T, dim=1),
                #                                               F.softmax(output_teacher / T, dim=1))*T**2
                loss = args.lambda_s * loss1 + args.lambda_t * loss2
                loss_avgkd.update(loss2.item())
            else:
                loss = loss1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if is_snapshot:
                logging.info("Snapshot!Generate a new teacher!")
                teacher = copy.deepcopy(model)

            metrics = utils.accuracy(output_batch, labels_batch, topk=(1, 5))  # metircs代表指标
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
            loss_avg.update(loss.item())

            t.update()

    train_metrics = {'train_loss': loss_avg.value(),
                     'train_accTop1': accTop1_avg.value(),
                     'train_accTop5': accTop5_avg.value(),
                     'loss_kd': loss_avgkd.value(),
                     'time': time.time() - end}

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
    logging.info("- Train metrics: " + metrics_string)
    return train_metrics, teacher


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
            loss_avg.update(loss.item())

    test_metrics = {'test_loss': loss_avg.value(),
                    'test_accTop1': accTop1_avg.value(),
                    'test_accTop5': accTop5_avg.value(),
                    'time': time.time() - end}

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in test_metrics.items())
    logging.info("- Test  metrics: " + metrics_string)
    return test_metrics


if __name__ == '__main__':
    begin_time = time.time()
    args.lambda_s = 1 + 1 / args.T
    args.lambda_t = 1

    utils.solve_dir(args.outdir)
    utils.solve_dir(os.path.join(args.outdir, args.arch))
    utils.solve_dir(os.path.join(args.outdir, args.arch, 'save_snapshot'))
    utils.solve_dir(os.path.join(args.outdir, args.arch, 'log'))

    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    utils.set_logger(os.path.join(args.outdir, args.arch, 'log', now_time + 'train.log'))

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
    train_loader, test_loader = data_loader.dataloader(data_name=args.dataset, batch_size=args.batch_size, root=root)
    logging.info("- Done.")

    iteration_total = args.num_epochs * len(train_loader)  # 迭代总次数
    iteration_per_cycle = iteration_total // args.cycle  # 每个cycle的迭代次数
    iteration_per_epoch = len(train_loader)

    model_fd = getattr(models, model_folder)
    if "resnet" in args.arch:
        model_cfg = getattr(model_fd, 'resnet')
        model = getattr(model_cfg, args.arch)(num_classes=num_classes)
    elif "vgg" in args.arch:
        model_cfg = getattr(model_fd, 'vgg')
        model = getattr(model_cfg, args.arch)(num_classes=num_classes, dropout=args.dropout)
    elif "densenet" in args.arch:
        model_cfg = getattr(model_fd, 'densenet')
        model = getattr(model_cfg, args.arch)(num_classes=num_classes)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    num_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
    logging.info('Total params: %.2fM' % num_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum, nesterov=True,
                          weight_decay=args.wd)
    teacher = None
    best_acc = 0

    writer = SummaryWriter(log_dir=os.path.join(args.outdir, args.arch))

    for i in range(args.num_epochs):
        logging.info("Epoch {}/{}".format(i + 1, args.num_epochs))
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], i + 1)

        train_metrics, teacher = train(train_loader, model, optimizer, teacher, i, args.T, iteration_per_epoch,
                                       iteration_per_cycle)

        writer.add_scalar('Train/Loss', train_metrics['train_loss'], i + 1)
        writer.add_scalar('Train/AccTop1', train_metrics['train_accTop1'], i + 1)
        writer.add_scalar('Train/KD_Loss', train_metrics['loss_kd'], i + 1)

        test_metrics = evaluate(test_loader, model, criterion)

        writer.add_scalar('Test/Loss', test_metrics['test_loss'], i + 1)
        writer.add_scalar('Test/AccTop1', test_metrics['test_accTop1'], i + 1)
        writer.add_scalar('Test/AccTop5', test_metrics['test_accTop5'], i + 1)

        test_acc = test_metrics['test_accTop1']

        save_dic = {'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'epoch': i + 1,
                    'test_accTop1': test_metrics['test_accTop1'],
                    'test_accTop5': test_metrics['test_accTop5']}
        if teacher is not None:
            save_dic['teacher_dict'] = teacher.state_dict()
        last_path = os.path.join(args.outdir, args.arch, 'save_snapshot', 'last.pth')
        torch.save(save_dic, last_path)

        if test_acc >= best_acc:
            logging.info("- Found better accuracy")
            best_acc = test_acc
            best_path = os.path.join(args.outdir, args.arch, 'save_snapshot', 'best.pth')
            torch.save(save_dic, best_path)

    writer.close()
    print("best_acc is ", best_acc)
    logging.info('Total time: {:.2f} minutes'.format((time.time() - begin_time) / 60.0))
    logging.info('All tasks have been done!')

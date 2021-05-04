import argparse
import copy
import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import models
import utils
from dataset import dataloader_New

from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='A New Method')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--atten', default=3, type=int)  # attention的数量
parser.add_argument('--outdir', default='save_New', type=str)
parser.add_argument('--arch', type=str, default='resnet32', help='models architecture')
parser.add_argument('--dataset', '-d', type=str, default='CIFAR100')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4 )')
parser.add_argument('--num_epochs', default=300, type=int,
                    help='number of total iterations')
parser.add_argument('--batch_size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--init_lr', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--wd', default=0.0001, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--T', type=float, default=3,
                    help='distillation temperature (default: 3)')
parser.add_argument('--k', type=float, default=5)
parser.add_argument('--dropout', default=0., type=float, help='Input the dropout rate: default(0.0)')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"


class DataAug(nn.Module):
    def __init__(self, data_name):
        super().__init__()
        if data_name == "CIFAR10":
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        elif data_name == "CIFAR100":
            normalize = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        elif data_name == "imagenet":
            normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.layer = nn.Sequential(transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   normalize)

    def forward(self, x):
        x = self.layer(x)
        return x


def train(train_loader, model, optimizer, criterion, teachers, T, aug):
    model.train()

    loss_total = utils.AverageMeter()
    loss_kd = utils.AverageMeter()
    loss_label = utils.AverageMeter()
    accTop1_avg = utils.AverageMeter()
    accTop5_avg = utils.AverageMeter()
    end = time.time()

    with tqdm(total=len(train_loader)) as t:
        for _, (train_batch, labels_batch) in enumerate(train_loader):
            train_batch = train_batch.to(device)
            labels_batch = labels_batch.to(device)

            train_batch_stu = aug(train_batch)
            student_query, output = model(train_batch_stu)
            student_query = student_query[:, None, :]  # Bx1x64
            loss1 = criterion(output, labels_batch)

            train_batch_tea = []
            for i in range(len(teachers)):
                train_batch_tea.append(aug(train_batch))

            if len(teachers) != 0:
                for teacher in teachers:
                    teacher.eval()
                teacher_keys, teacher_outputs = teachers[0](train_batch_tea[0])
                teacher_keys = teacher_keys[:, :, None]
                teacher_outputs = teacher_outputs[:, None, :]
                for i in range(1, len(teachers)):
                    teacher = teachers[i]
                    train_batch = train_batch_tea[i]
                    temp1, temp2 = teacher(train_batch)
                    temp1 = temp1[:, :, None]
                    temp2 = temp2[:, None, :]
                    teacher_keys = torch.cat((teacher_keys, temp1), 2)  # B x 64 x atten
                    teacher_outputs = torch.cat((teacher_outputs, temp2), 1)  # B x atten x 100
                teacher_outputs = F.softmax(teacher_outputs / T, dim=2)
                energy = torch.bmm(student_query, teacher_keys)  # bmm是批处理当中的矩阵乘法
                attention = F.softmax(energy, dim=-1)  # B x 1 x atten 权重归一化
                final_teacher = torch.bmm(attention, teacher_outputs)  # Bx1x100
                final_teacher = final_teacher.squeeze(1)  # Bx100
                final_teacher = final_teacher.detach()

                loss2 = F.kl_div(F.log_softmax(output / T, dim=1), final_teacher, reduction='batchmean') * T ** 2
                total_loss = loss1 + loss2
            else:
                loss2 = torch.tensor(0)
                total_loss = loss1

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            metrics = utils.accuracy(output, labels_batch, topk=(1, 5))  # metircs代表指标
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
            loss_total.update(total_loss.item())
            loss_kd.update(loss2.item())
            loss_label.update(loss1.item())

            t.update()

    train_metrics = {'train_loss': loss_total.value(),
                     'train_accTop1': accTop1_avg.value(),
                     'train_accTop5': accTop5_avg.value(),
                     'label_loss': loss_label.value(),
                     'loss_kd': loss_kd.value(),
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
            test_batch = test_batch.to(device)
            labels_batch = labels_batch.to(device)

            _, output_batch = model(test_batch)
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

    utils.solve_dir(args.outdir)
    utils.solve_dir(os.path.join(args.outdir, args.arch))
    utils.solve_dir(os.path.join(args.outdir, args.arch, 'atten' + str(args.atten), 'save_snapshot'))
    utils.solve_dir(os.path.join(args.outdir, args.arch, 'atten' + str(args.atten), 'log'))

    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    utils.set_logger(os.path.join(args.outdir, args.arch, 'atten' + str(args.atten), 'log', now_time + 'train.log'))

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
    train_loader, test_loader = dataloader_New.dataloader(data_name=args.dataset, batch_size=args.batch_size, root=root)
    logging.info("- Done.")

    model_fd = getattr(models, model_folder)
    if "resnet" in args.arch:
        model_cfg = getattr(model_fd, 'resnet')
        model = getattr(model_cfg, args.arch)(num_classes=num_classes, KD=True)
    elif "vgg" in args.arch:
        model_cfg = getattr(model_fd, 'vgg')
        model = getattr(model_cfg, args.arch)(num_classes=num_classes, KD=True, dropout=args.dropout)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    num_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
    logging.info('Total params: %.2fM' % num_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum, nesterov=True,
                          weight_decay=args.wd)

    teachers = []
    best_acc = 0
    writer = SummaryWriter(log_dir=os.path.join(args.outdir, args.arch, 'atten' + str(args.atten)))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)
    aug = DataAug(data_name=args.dataset)

    for i in range(args.num_epochs):
        logging.info("Epoch {}/{}".format(i + 1, args.num_epochs))
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], i + 1)

        train_metrics = train(train_loader, model, optimizer, criterion, teachers, args.T, aug)
        writer.add_scalar('Train/Loss', train_metrics['train_loss'], i + 1)
        writer.add_scalar('Train/AccTop1', train_metrics['train_accTop1'], i + 1)
        writer.add_scalar('Train/kd_loss', train_metrics['loss_kd'], i + 1)
        writer.add_scalar('Train/label_loss', train_metrics['label_loss'], i + 1)

        test_metrics = evaluate(test_loader, model, criterion)
        writer.add_scalar('Test/Loss', test_metrics['test_loss'], i + 1)
        writer.add_scalar('Test/AccTop1', test_metrics['test_accTop1'], i + 1)
        writer.add_scalar('Test/AccTop5', test_metrics['test_accTop5'], i + 1)

        # 每k个epoch更新一次teachers
        if (i + 1) % args.k == 0:
            teacher_new = copy.deepcopy(model)
            teachers.append(teacher_new)
            if len(teachers) > args.atten:
                teachers = teachers[-args.atten:]

        test_acc = test_metrics['test_accTop1']
        save_dic = {'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'epoch': i + 1,
                    'test_accTop1': test_metrics['test_accTop1'],
                    'test_accTop5': test_metrics['test_accTop5']}
        last_path = os.path.join(args.outdir, args.arch, 'atten' + str(args.atten), 'save_snapshot', 'last.pth')
        # torch.save(save_dic, last_path)
        if test_acc >= best_acc:
            logging.info("- Found better accuracy")
            best_acc = test_acc
            best_path = os.path.join(args.outdir, args.arch, 'atten' + str(args.atten), 'save_snapshot', 'best.pth')
            # torch.save(save_dic, best_path)
        scheduler.step()

    writer.close()
    logging.info("best_acc is {}".format(best_acc))
    logging.info('Total time: {:.2f} minutes'.format((time.time() - begin_time) / 60.0))
    logging.info('All tasks have been done!')

"""
nohup python train_New.py --gpu 2 --arch resnet32 --outdir save_New --factor 8 --atten 3 > New_resnet32_atten3.out 2>&1 &
"""

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

parser = argparse.ArgumentParser(description='A New Method')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--atten', default=3, type=int)  # attention的数量
parser.add_argument('--outdir', default='save_New_V1', type=str)
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
parser.add_argument('--factor', type=int, default=8)
parser.add_argument('--tea_avg', action='store_true')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"


class Linear_Model(nn.Module):
    def __init__(self, dimension_num):
        super(Linear_Model, self).__init__()
        self.fc = nn.Linear(dimension_num, dimension_num // args.factor)

    def forward(self, x):
        x = self.fc(x)
        return x


def train(train_loader, model, optimizer, criterion, teachers, T, query_weight, key_weight, cur_epoch):
    model.train()

    loss_total = utils.AverageMeter()
    loss_kd = utils.AverageMeter()
    loss_label = utils.AverageMeter()
    accTop1_avg = utils.AverageMeter()
    accTop5_avg = utils.AverageMeter()
    end = time.time()

    alpha = 1 - 0.9 * (cur_epoch - cur_epoch % args.k) / args.num_epochs  # 交叉熵loss前面的系数

    with tqdm(total=len(train_loader)) as t:
        for _, (img, labels_batch) in enumerate(train_loader):
            img=img.to(device)
            labels_batch = labels_batch.to(device)

            student_query, output = model(img)
            student_query = student_query.detach()
            student_query = query_weight(student_query)  # Bx8
            student_query = student_query[:, None, :]  # Bx1x8

            loss1 = criterion(output, labels_batch)

            if len(teachers) != 0:
                for teacher in teachers:
                    teacher.eval()
                teacher_keys, teacher_outputs = teachers[0](train_batch_tea[0])
                teacher_keys = teacher_keys.detach()
                teacher_outputs = teacher_outputs.detach()
                teacher_keys = key_weight(teacher_keys)  # Bx8
                teacher_keys = teacher_keys[:, :, None]  # Bx8x1
                teacher_outputs = teacher_outputs[:, None, :]  # Bx1x100
                for i in range(1, len(teachers)):
                    teacher = teachers[i]
                    train_batch = train_batch_tea[i]
                    temp1, temp2 = teacher(train_batch)
                    temp1 = temp1.detach()
                    temp2 = temp2.detach()
                    temp1 = key_weight(temp1)
                    temp1 = temp1[:, :, None]
                    temp2 = temp2[:, None, :]
                    teacher_keys = torch.cat((teacher_keys, temp1), 2)  # B x 8 x atten
                    teacher_outputs = torch.cat((teacher_outputs, temp2), 1)  # B x atten x 100
                # teacher_outputs = F.softmax(teacher_outputs / T, dim=2)
                if args.tea_avg:
                    final_teacher = teacher_outputs.mean(dim=1)
                else:
                    energy = torch.bmm(student_query, teacher_keys)  # / math.sqrt(student_query.size(2))
                    attention = F.softmax(energy, dim=-1)  # B x 1 x atten 权重归一化
                    final_teacher = torch.bmm(attention, teacher_outputs)  # Bx1x100
                    final_teacher = final_teacher.squeeze(1)  # Bx100
                final_teacher = F.softmax(final_teacher / T, dim=1)

                loss2 = F.kl_div(F.log_softmax(output / T, dim=1), final_teacher, reduction='batchmean') * T ** 2
                total_loss = alpha * loss1 + (1 - alpha) * loss2
                loss_kd.update((1-alpha) * loss2.item())
                loss_label.update(alpha * loss1.item())
            else:
                loss2 = torch.tensor(0)
                total_loss = loss1
                loss_kd.update(loss2.item())
                loss_label.update(loss1.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            metrics = utils.accuracy(output, labels_batch, topk=(1, 5))
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
            loss_total.update(total_loss.item())

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

    if args.tea_avg:
        ss = "avg"
    else:
        ss = "atten"
    utils.solve_dir(os.path.join(args.outdir, args.arch, ss + str(args.atten), 'save_snapshot'))
    utils.solve_dir(os.path.join(args.outdir, args.arch, ss + str(args.atten), 'log'))

    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    utils.set_logger(os.path.join(args.outdir, args.arch, ss + str(args.atten), 'log', now_time + 'train.log'))

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

    model_fd = getattr(models, model_folder)
    if "resnet" in args.arch:
        model_cfg = getattr(model_fd, 'resnet')
        model = getattr(model_cfg, args.arch)(num_classes=num_classes, KD=True)
    elif "vgg" in args.arch:
        model_cfg = getattr(model_fd, 'vgg')
        model = getattr(model_cfg, args.arch)(num_classes=num_classes, KD=True, dropout=args.dropout)
    elif "densenet" in args.arch:
        model_cfg = getattr(model_fd, 'densenet')
        model = getattr(model_cfg, args.arch)(num_classes=num_classes, KD=True)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    dataiter = iter(train_loader)
    img_temp = next(dataiter)[0][0]
    img_temp = img_temp.to(device)
    # print(img_temp.shape)  # torch.Size([128, 3, 32, 32])
    xf, _ = model(img_temp)
    dimension_num = xf.size(1)
    query_weight = Linear_Model(dimension_num)
    key_weight = Linear_Model(dimension_num)
    if torch.cuda.device_count() > 1:
        query_weight = nn.DataParallel(query_weight).to(device)
        key_weight = nn.DataParallel(key_weight).to(device)
    else:
        query_weight = query_weight.to(device)
        key_weight = key_weight.to(device)

    num_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
    logging.info('Total params: %.2fM' % num_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params': model.parameters()},
                           {'params': query_weight.parameters()},
                           {'params': key_weight.parameters()}],
                          lr=args.init_lr, momentum=args.momentum, nesterov=True, weight_decay=args.wd)

    teachers = []
    best_acc = 0
    writer = SummaryWriter(log_dir=os.path.join(args.outdir, args.arch, ss + str(args.atten)))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    for i in range(args.num_epochs):
        logging.info("Epoch {}/{}".format(i + 1, args.num_epochs))
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], i + 1)

        train_metrics = train(train_loader, model, optimizer, criterion, teachers, args.T, query_weight, key_weight,
                              i + 1)
        writer.add_scalar('Train/Loss', train_metrics['train_loss'], i + 1)
        writer.add_scalar('Train/AccTop1', train_metrics['train_accTop1'], i + 1)
        writer.add_scalar('Train/kd_loss', train_metrics['loss_kd'], i + 1)
        writer.add_scalar('Train/label_loss', train_metrics['label_loss'], i + 1)

        test_metrics = evaluate(test_loader, model, criterion)
        writer.add_scalar('Test/Loss', test_metrics['test_loss'], i + 1)
        writer.add_scalar('Test/AccTop1', test_metrics['test_accTop1'], i + 1)
        writer.add_scalar('Test/AccTop5', test_metrics['test_accTop5'], i + 1)

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
        last_path = os.path.join(args.outdir, args.arch, ss + str(args.atten), 'save_snapshot', 'last.pth')
        # torch.save(save_dic, last_path)
        if test_acc >= best_acc:
            logging.info("- Found better accuracy")
            best_acc = test_acc
            best_path = os.path.join(args.outdir, args.arch, ss + str(args.atten), 'save_snapshot', 'best.pth')
            # torch.save(save_dic, best_path)
        scheduler.step()

    writer.close()
    logging.info("best_acc is {}".format(best_acc))
    logging.info('Total time: {:.2f} minutes'.format((time.time() - begin_time) / 60.0))
    logging.info('All tasks have been done!')

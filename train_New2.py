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
from torch.optim.lr_scheduler import MultiStepLR

import models
import utils
from dataset import data_loader

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='A New Method2')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--atten', default=3, type=int)  # attention的数量
parser.add_argument('--outdir', default='save_New2', type=str)
parser.add_argument('--arch', type=str, default='resnet32', help='models architecture')
parser.add_argument('--dataset', '-d', type=str, default='CIFAR100')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4 )')
parser.add_argument('--num_epochs', default=300, type=int,
                    help='number of total iterations')
parser.add_argument('--batch_size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--init_lr', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                    help='Learning rate step gamma (default: 0.1)')
parser.add_argument('--warm_up', default=0, type=int)  # 热身阶段的epoch数量
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=0.0001, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--T', type=float, default=3,
                    help='distillation temperature (default: 3)')
parser.add_argument('--lambda_1', type=float, default=1)
parser.add_argument('--lambda_2', type=float, default=1)
parser.add_argument('--lambda_3', type=float, default=0.001)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--dropout', default=0., type=float, help='Input the dropout rate: default(0.0)')
parser.add_argument('--sd_KD', action='store_true', help='KD mode in snapshot distillation with model')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"


def train_normal(train_loader, model, optimizer, criterion):
    model.train()
    loss_avg = utils.AverageMeter()
    accTop1_avg = utils.AverageMeter()
    accTop5_avg = utils.AverageMeter()
    end = time.time()

    with tqdm(total=len(train_loader)) as t:
        for _, (train_batch, labels_batch) in enumerate(train_loader):
            train_batch = train_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)
            output_batch = model(train_batch)[0]
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


def train(train_loader, model, optimizer, criterion, teachers, T, cur_epoch):
    model.train()

    loss_total = utils.AverageMeter()
    loss_label = utils.AverageMeter()
    loss_fea = utils.AverageMeter()
    loss_soft = utils.AverageMeter()
    accTop1_avg = utils.AverageMeter()
    accTop5_avg = utils.AverageMeter()
    end = time.time()

    with tqdm(total=len(train_loader)) as t:
        for _, (train_batch, labels_batch) in enumerate(train_loader):
            train_batch = train_batch.to(device)
            labels_batch = labels_batch.to(device)

            output, student_query, s_feas = model(train_batch)
            student_query = student_query[:, None, :]  # Bx1x64
            loss1 = criterion(output, labels_batch)

            if len(teachers) != 0:
                teacher_feas = []
                for teacher in teachers:
                    teacher.eval()
                with torch.no_grad():
                    teacher_outputs, teacher_keys, t_feas = teachers[0](train_batch)
                    teacher_keys = teacher_keys[:, :, None]
                    teacher_outputs = teacher_outputs[:, None, :]
                    for i in range(len(t_feas)):
                        teacher_feas.append([t_feas[i]])

                    for teacher in teachers[1:]:
                        temp2, temp1, tfs = teacher(train_batch)
                        temp1 = temp1[:, :, None]
                        temp2 = temp2[:, None, :]
                        teacher_keys = torch.cat([teacher_keys, temp1], -1)  # B x 64 x atten
                        teacher_outputs = torch.cat([teacher_outputs, temp2], 1)  # B x atten x 100
                        for i in range(len(tfs)):
                            teacher_feas[i].append(tfs[i])

                    teacher_outputs = F.softmax(teacher_outputs / T, dim=2)
                    energy = torch.bmm(student_query, teacher_keys)  # bmm是批处理当中的矩阵乘法
                    attention = F.softmax(energy, dim=-1)  # B x 1 x atten
                    final_teacher = torch.bmm(attention, teacher_outputs)  # Bx1x100
                    final_teacher = final_teacher.squeeze(1)  # Bx100
                    # final_teacher = final_teacher.detach()

                    final_feas = []
                    for feas in teacher_feas:
                        temp = feas[0]
                        temp = temp[:, None, :, :, :]
                        for j in feas[1:]:
                            j = j[:, None, :, :, :]
                            temp = torch.cat([temp, j], dim=1)
                        final_feas.append(temp)
                    for i in range(len(final_feas)):
                        fea = final_feas[i]
                        fea = torch.einsum('afb, abcde -> afcde', attention, fea)
                        fea = torch.squeeze(fea)
                        final_feas[i] = fea
                    # TODO:此处有bug

                if args.sd_KD == True:
                    loss2 = (- F.log_softmax(output / T, 1) * final_teacher).sum(dim=1).mean() * T ** 2
                else:
                    loss2 = (- F.log_softmax(output, 1) * final_teacher).sum(dim=1).mean()
                loss3 = torch.tensor(0.0).to(device)
                for i in range(len(s_feas)):
                    loss3 += torch.pow((s_feas[i] - final_feas[i]), 2).sum()
                loss3 /= output.size(0)
                # print("loss1:", loss1* args.lambda_1)
                # print("loss2:", loss2* args.lambda_2)
                # print("loss3:", loss3* args.lambda_3)

                total_loss = loss1 * args.lambda_1 + loss2 * args.lambda_2 + loss3 * args.lambda_3
            else:
                loss2 = torch.tensor(0)
                loss3 = torch.tensor(0)
                total_loss = loss1

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            metrics = utils.accuracy(output, labels_batch, topk=(1, 5))  # metircs代表指标
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
            loss_total.update(total_loss.item())
            loss_label.update(loss1.item())
            loss_soft.update(loss2.item())
            loss_fea.update(loss3.item())

            t.update()

    if (cur_epoch + 1-args.warm_up) % args.k == 0:
        new_teacher = copy.deepcopy(model)
        teachers.append(new_teacher)
        if len(teachers) > args.atten:
            teachers = teachers[-args.atten:]

    train_metrics = {'train_loss': loss_total.value(),
                     'train_accTop1': accTop1_avg.value(),
                     'train_accTop5': accTop5_avg.value(),
                     'loss_label': loss_label.value(),
                     'loss_soft': loss_soft.value(),
                     'loss_fea': loss_fea.value(),
                     'time': time.time() - end}

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
    logging.info("- Train metrics: " + metrics_string)
    return train_metrics, teachers


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

            output_batch = model(test_batch)[0]
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
    utils.solve_dir(os.path.join(args.outdir, args.arch,
                                 'atten' + str(args.atten) + '_warmup' + str(args.warm_up),
                                 'save_snapshot'))
    utils.solve_dir(os.path.join(args.outdir, args.arch,
                                 'atten' + str(args.atten) + '_warmup' + str(args.warm_up),
                                 'log'))

    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    utils.set_logger(os.path.join(args.outdir, args.arch,
                                  'atten' + str(args.atten) + '_warmup' + str(args.warm_up),
                                  'log', now_time + 'train.log'))

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
        model_cfg = getattr(model_fd, 'resnet_New2')
        model = getattr(model_cfg, args.arch)(num_classes=num_classes, KD=True)
    elif "vgg" in args.arch:
        model_cfg = getattr(model_fd, 'vgg_New2')
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
    scheduler = MultiStepLR(optimizer, milestones=[150, 225], gamma=args.gamma, verbose=True)
    writer = SummaryWriter(
        log_dir=os.path.join(args.outdir, args.arch, 'atten' + str(args.atten) + '_warmup' + str(args.warm_up)))

    for i in range(args.warm_up):
        logging.info("Epoch {}/{}".format(i + 1, args.num_epochs))
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], i + 1)

        train_metrics = train_normal(train_loader, model, optimizer, criterion)
        writer.add_scalar('Train/Loss', train_metrics['train_loss'], i + 1)
        writer.add_scalar('Train/AccTop1', train_metrics['train_accTop1'], i + 1)
        writer.add_scalar('Train/Label_Loss', train_metrics['train_loss'], i + 1)
        writer.add_scalar('Train/Soft_Loss', 0, i + 1)
        writer.add_scalar('Train/Feature_Loss', 0, i + 1)

        test_metrics = evaluate(test_loader, model, criterion)
        writer.add_scalar('Test/Loss', test_metrics['test_loss'], i + 1)
        writer.add_scalar('Test/AccTop1', test_metrics['test_accTop1'], i + 1)
        writer.add_scalar('Test/AccTop5', test_metrics['test_accTop5'], i + 1)

        test_acc = test_metrics['test_accTop1']
        if test_acc >= best_acc:
            logging.info("- Found better accuracy")
            best_acc = test_acc
            teacher_new = copy.deepcopy(model)
            teachers.append(teacher_new)
            if len(teachers) > args.atten:
                teachers = teachers[-args.atten:]

        scheduler.step()

    for i in range(args.warm_up, args.num_epochs):
        logging.info("Epoch {}/{}".format(i + 1, args.num_epochs))
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], i + 1)

        train_metrics, teachers = train(train_loader, model, optimizer, criterion, teachers, args.T, i)
        writer.add_scalar('Train/Loss', train_metrics['train_loss'], i + 1)
        writer.add_scalar('Train/AccTop1', train_metrics['train_accTop1'], i + 1)
        writer.add_scalar('Train/Label_Loss', train_metrics['loss_label'], i + 1)
        writer.add_scalar('Train/Soft_Loss', train_metrics['loss_soft'], i + 1)
        writer.add_scalar('Train/Feature_Loss', train_metrics['loss_fea'], i + 1)

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
        last_path = os.path.join(args.outdir, args.arch,
                                 'atten' + str(args.atten) + '_warmup' + str(args.warm_up),
                                 'save_snapshot', 'last.pth')
        torch.save(save_dic, last_path)
        if test_acc >= best_acc:
            logging.info("- Found better accuracy")
            best_acc = test_acc
            best_path = os.path.join(args.outdir, args.arch,
                                     'atten' + str(args.atten) + '_warmup' + str(args.warm_up),
                                     'save_snapshot', 'best.pth')
            torch.save(save_dic, best_path)
        scheduler.step()

    writer.close()
    logging.info("best_acc is {}".format(best_acc))
    logging.info('Total time: {:.2f} minutes'.format((time.time() - begin_time) / 60.0))
    logging.info('All tasks have been done!')

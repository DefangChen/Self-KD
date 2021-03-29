import argparse
import logging
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

import utils
from dataset import data_loader
from models.model_cifar import resnet_own

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 training')
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument('--gpu', default='4,5', type=str)
parser.add_argument('--outdir', default='save_own_teacher', type=str)
parser.add_argument('--arch', type=str, default='multi_resnet50_kd',
                    help='models architecture')
parser.add_argument('--dataset', '-d', type=str, default='CIFAR100',
                    help='dataset choice')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4 )')
parser.add_argument('--num_epochs', default=300, type=int,
                    help='number of total iterations (default: 64,000)')
parser.add_argument('--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--dropout', default=0., type=float)
parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                    help='Learning rate step gamma (default: 0.1)')
# dest表示参数的别名
parser.add_argument('--resume', default='', type=str,
                    help='path to  latest checkpoint (default: None)')
parser.add_argument('--warm-up', action='store_true',
                    help='for n = 18, the models needs to warm up for 400 '
                         'iterations')

# kd parameter
parser.add_argument('--temperature', default=3, type=int,
                    help='temperature to smooth the logits')
parser.add_argument('--alpha', default=0.1, type=float,
                    help='weight of kd loss')
parser.add_argument('--beta', default=1e-6, type=float,
                    help='weight of feature loss')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"


def train(train_loader, model, optimizer, criterion, current_epoch):
    model.train()
    end = time.time()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    middle1_losses = utils.AverageMeter()
    middle2_losses = utils.AverageMeter()
    middle3_losses = utils.AverageMeter()
    losses1_kd = utils.AverageMeter()
    losses2_kd = utils.AverageMeter()
    losses3_kd = utils.AverageMeter()
    feature_losses_1 = utils.AverageMeter()
    feature_losses_2 = utils.AverageMeter()
    feature_losses_3 = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    middle1_top1 = utils.AverageMeter()
    middle2_top1 = utils.AverageMeter()
    middle3_top1 = utils.AverageMeter()
    accTop1_avg = utils.AverageMeter()
    accTop5_avg = utils.AverageMeter()

    # utils.adjust_learning_rate(args, optimizer, current_epoch)
    with tqdm(total=len(train_loader)) as t:
        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output, middle_output1, middle_output2, middle_output3, final_fea, middle1_fea, middle2_fea, middle3_fea = model(
                input)

            # label loss
            loss = criterion(output, target)
            losses.update(loss.item())
            middle1_loss = criterion(middle_output1, target)
            middle1_losses.update(middle1_loss.item())
            middle2_loss = criterion(middle_output2, target)
            middle2_losses.update(middle2_loss.item())
            middle3_loss = criterion(middle_output3, target)
            middle3_losses.update(middle3_loss.item())

            temp4 = output / args.temperature
            temp4 = torch.softmax(temp4, dim=1)  # 做KD用的softmax后的输出

            # KD loss
            loss1by4 = utils.kd_loss_function(middle_output1, temp4.detach(), args) * (args.temperature ** 2)
            losses1_kd.update(loss1by4.item())
            loss2by4 = utils.kd_loss_function(middle_output2, temp4.detach(), args) * (args.temperature ** 2)
            losses2_kd.update(loss2by4.item())
            loss3by4 = utils.kd_loss_function(middle_output3, temp4.detach(), args) * (args.temperature ** 2)
            losses3_kd.update(loss3by4.item())

            # L2 loss from features TODO:feature_map_loss可能存在bug 没有取平均
            feature_loss_1 = utils.feature_loss_function(middle1_fea, final_fea.detach())
            feature_losses_1.update(feature_loss_1.item())
            feature_loss_2 = utils.feature_loss_function(middle2_fea, final_fea.detach())
            feature_losses_2.update(feature_loss_2.item())
            feature_loss_3 = utils.feature_loss_function(middle3_fea, final_fea.detach())
            feature_losses_3.update(feature_loss_3.item())

            total_loss = (1 - args.alpha) * (loss + middle1_loss + middle2_loss + middle3_loss) + \
                         args.alpha * (loss1by4 + loss2by4 + loss3by4) + \
                         args.beta * (feature_loss_1 + feature_loss_2 + feature_loss_3)
            total_losses.update(total_loss.item())

            metrics = utils.accuracy(output, target, topk=(1, 5))
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
            middle1_prec1 = utils.accuracy(middle_output1, target, topk=(1,))
            middle1_top1.update(middle1_prec1[0].item())
            middle2_prec1 = utils.accuracy(middle_output2, target, topk=(1,))
            middle2_top1.update(middle2_prec1[0].item())
            middle3_prec1 = utils.accuracy(middle_output3, target, topk=(1,))
            middle3_top1.update(middle3_prec1[0].item())

            optimizer.zero_grad()
            total_loss.backward()  # 反向传播
            optimizer.step()

            t.update()

        train_metrics = {'train_loss': total_losses.value(),
                         'train_accTop1': accTop1_avg.value(),
                         'train_accTop5': accTop5_avg.value(),
                         'middle_losses': middle1_losses.value() + middle2_losses.value() + middle3_losses.value(),
                         'losses_kd': losses1_kd.value() + losses2_kd.value() + losses3_kd.value(),
                         'feature_losses': feature_losses_1.value() + feature_losses_2.value() + feature_losses_3.value(),
                         'time': time.time() - end, }

        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
        logging.info("- Train metrics: " + metrics_string)
        return train_metrics


def evaluate(test_loader, model, criterion):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    middle1_losses = utils.AverageMeter()
    middle2_losses = utils.AverageMeter()
    middle3_losses = utils.AverageMeter()
    middle1_top1 = utils.AverageMeter()
    middle2_top1 = utils.AverageMeter()
    middle3_top1 = utils.AverageMeter()
    accTop1_avg = utils.AverageMeter()
    accTop5_avg = utils.AverageMeter()

    model.eval()
    end = time.time()

    for i, (input, target) in enumerate(test_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        output, middle_output1, middle_output2, middle_output3, \
        final_fea, middle1_fea, middle2_fea, middle3_fea = model(input)

        loss = criterion(output, target)
        losses.update(loss.item())
        middle1_loss = criterion(middle_output1, target)
        middle1_losses.update(middle1_loss.item())
        middle2_loss = criterion(middle_output2, target)
        middle2_losses.update(middle2_loss.item())
        middle3_loss = criterion(middle_output3, target)
        middle3_losses.update(middle3_loss.item())

        prec = utils.accuracy(output.data, target, topk=(1, 5))
        accTop1_avg.update(prec[0].item())
        accTop5_avg.update(prec[1].item())

        middle1_prec1 = utils.accuracy(middle_output1, target, topk=(1,))
        middle1_top1.update(middle1_prec1[0].item())
        middle2_prec1 = utils.accuracy(middle_output2, target, topk=(1,))
        middle2_top1.update(middle2_prec1[0].item())
        middle3_prec1 = utils.accuracy(middle_output3, target, topk=(1,))
        middle3_top1.update(middle3_prec1[0].item())
        batch_time.update(time.time() - end)
        end = time.time()

    test_metrics = {'test_loss': losses.value(),
                    'test_accTop1': accTop1_avg.value(),
                    'test_accTop5': accTop5_avg.value(),
                    'middle1_top1': middle1_top1.value(),
                    'middle2_top1': middle2_top1.value(),
                    'middle3_top1': middle2_top1.value(),
                    'time': time.time() - end}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in test_metrics.items())
    logging.info("- Test  metrics: " + metrics_string)
    return test_metrics


def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, scheduler):
    start_epoch = 0
    best_acc = 0.0

    for epoch in range(start_epoch, args.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))
        train_metrics = train(train_loader, model, optimizer, criterion, epoch)
        test_metrics = evaluate(test_loader, model, criterion)

        last_path = os.path.join(args.outdir, args.arch, "save_model", 'last.pth')
        torch.save({'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'test_accTop1': test_metrics['test_accTop1'],
                    'test_accTop5': test_metrics['test_accTop5']}, last_path)

        test_acc = test_metrics['test_accTop1']
        is_best = (test_acc >= best_acc)
        if is_best:
            logging.info("- Found better accuracy")
            best_acc = test_acc
            test_metrics['epoch'] = epoch + 1
            utils.save_dict_to_json(test_metrics,
                                    os.path.join(args.outdir, args.arch, "save_model", "test_best_metrics.json"))
            shutil.copyfile(last_path, os.path.join(args.outdir, args.arch, "save_model", 'best.pth'))
        scheduler.step()


if __name__ == '__main__':
    begin_time = time.time()
    utils.solve_dir(args.outdir)
    utils.solve_dir(os.path.join(args.outdir, args.arch))
    utils.solve_dir(os.path.join(args.outdir, args.arch, 'save_model'))
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

    if args.arch == "multi_resnet50_kd":
        model = resnet_own.multi_resnet50_kd(num_classes)
    elif args.arch == "multi_resnet18_kd":
        model = resnet_own.multi_resnet18_kd(num_classes)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    # 从args.resume这个路径下的文件 载入已有的模型参数
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint `{}`".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info('=> loaded checkpoint `{}` (epoch: {})'.format(args.resume, checkpoint['epoch']))
        else:
            logging.info('=> no checkpoint found at `{}`'.format(args.resume))

    num_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
    logging.info('Total params: %.2fM' % num_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.wd)
    scheduler = MultiStepLR(optimizer, milestones=[150, 225], gamma=args.gamma, verbose=True)

    logging.info("Starting training for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, scheduler)
    logging.info('Total time: {:.2f} minutes'.format((time.time() - begin_time) / 60.0))

import argparse
import logging
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

import models
import utils
from dataset.data_loader import dataloader

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser.add_argument('--model', metavar='ARCH', default='vgg16', type=str,
                    choices=model_names,
                    help='models architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('--dataset', default='CIFAR100', type=str, help='Input the name of dataset: default(CIFAR100)')
parser.add_argument('--num_epochs', default=300, type=int, help='Input the number of epoches: default(300)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--batch_size', default=128, type=int, help='Input the batch size: default(128)')
parser.add_argument("--outdir", type=str, default="save_baseline")
parser.add_argument('--lr', default=0.1, type=float, help='Input the learning rate: default(0.1)')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--efficient', action='store_true',
                    help='Decide whether or not to use efficient implementation(only of densenet): default(False)')
parser.add_argument('--wd', default=1e-4, type=float, help='Input the weight decay rate: default(5e-4)')
parser.add_argument('--dropout', default=0., type=float, help='Input the dropout rate: default(0.0)')
parser.add_argument('--resume', default='', type=str, help='Input the path of resume models: default('')')
parser.add_argument('--num_workers', default=8, type=int, help='Input the number of works: default(8)')
parser.add_argument('--gpu', default='3', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(train_loader, model, optimizer, criterion, accuracy):
    model.train()

    loss_avg = utils.AverageMeter()
    accTop1_avg = utils.AverageMeter()
    accTop5_avg = utils.AverageMeter()
    end = time.time()

    with tqdm(total=len(train_loader)) as t:
        for _, (train_batch, labels_batch) in enumerate(train_loader):
            train_batch = train_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)

            output_batch = model(train_batch)
            loss = criterion(output_batch, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = accuracy(output_batch, labels_batch, topk=(1, 5))  # metircs代表指标
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
            loss_avg.update(loss.item())

            t.update()

    train_metrics = {'train_loss': loss_avg.value(),
                     'train_accTop1': accTop1_avg.value(),
                     'train_accTop5': accTop5_avg.value(),
                     'time': time.time() - end}

    # join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
    logging.info("- Train metrics: " + metrics_string)
    return train_metrics


def evaluate(test_loader, model, criterion, accuracy):
    model.eval()
    loss_avg = utils.AverageMeter()
    accTop1_avg = utils.AverageMeter()
    accTop5_avg = utils.AverageMeter()
    end = time.time()

    with torch.no_grad():
        for test_batch, labels_batch in test_loader:
            test_batch = test_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)

            # compute models output
            output_batch = model(test_batch)
            loss = criterion(output_batch, labels_batch)

            # Update average loss and accuracy
            metrics = accuracy(output_batch, labels_batch, topk=(1, 5))  # topk指的是什么？
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


def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, accuracy, model_dir, args):
    start_epoch = 0
    best_acc = 0.0
    scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    writer = SummaryWriter(log_dir=model_dir)
    choose_accTop1 = True
    result_train_metrics = list(range(args.num_epochs))
    result_test_metrics = list(range(args.num_epochs))

    if args.resume:
        # Load checkpoint.
        logging.info('Resuming from checkpoint..')
        resumePath = os.path.join(args.resume, 'last.pth')
        assert os.path.isfile(resumePath), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(resumePath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim_dict'])
        start_epoch = checkpoint['epoch']
        scheduler.step(start_epoch - 1)
        if choose_accTop1:
            best_acc = checkpoint['test_accTop1']
        else:
            best_acc = checkpoint['test_accTop5']
        result_train_metrics = torch.load(os.path.join(args.resume, 'train_metrics'))
        result_test_metrics = torch.load(os.path.join(args.resume, 'test_metrics'))

    for epoch in range(start_epoch, args.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))

        train_metrics = train(train_loader, model, optimizer, criterion, accuracy)
        writer.add_scalar('Train/Loss', train_metrics['train_loss'], epoch + 1)
        writer.add_scalar('Train/AccTop1', train_metrics['train_accTop1'], epoch + 1)
        writer.add_scalar('Train/AccTop5', train_metrics['train_accTop5'], epoch + 1)

        test_metrics = evaluate(test_loader, model, criterion, accuracy)
        test_acc = test_metrics['test_accTop1']

        writer.add_scalar('Test/Loss', test_metrics['test_loss'], epoch + 1)
        writer.add_scalar('Test/AccTop1', test_metrics['test_accTop1'], epoch + 1)
        writer.add_scalar('Test/AccTop5', test_metrics['test_accTop5'], epoch + 1)

        result_train_metrics[epoch] = train_metrics
        result_test_metrics[epoch] = test_metrics

        # Save latest train/test metrics
        # torch.save(result_train_metrics, os.path.join(model_dir, 'train_metrics'))
        # torch.save(result_test_metrics, os.path.join(model_dir, 'test_metrics'))

        # last_path = os.path.join(model_dir, 'save_model', 'last.pth')
        # torch.save({'state_dict': model.state_dict(),
        #             'optim_dict': optimizer.state_dict(),
        #             'epoch': epoch + 1,
        #             'test_accTop1': test_metrics['test_accTop1'],
        #             'test_accTop5': test_metrics['test_accTop5']}, last_path)
        if test_acc >= best_acc:
            logging.info("- Found better accuracy")
            best_acc = test_acc
            test_metrics['epoch'] = epoch + 1
            # utils.save_dict_to_json(test_metrics, os.path.join(model_dir, "test_best_metrics.json"))
            # shutil.copyfile(last_path, os.path.join(model_dir, 'save_model', 'best.pth'))
        scheduler.step()
    writer.close()


if __name__ == '__main__':
    begin_time = time.time()
    model_dir = args.outdir
    utils.solve_dir(os.path.join(model_dir, args.model, 'save_model'))
    utils.solve_dir(os.path.join(model_dir, args.model, 'log'))

    model_dir = os.path.join(model_dir, args.model)

    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    utils.set_logger(os.path.join(model_dir, 'log', now_time + ' train.log'))

    w = vars(args)
    metrics_string = " ;\n".join("{}: {}".format(k, v) for k, v in w.items())
    logging.info("- All args are followed:\n" + metrics_string)

    logging.info("Loading the datasets...")
    # set number of classes
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
    train_loader, test_loader = dataloader(data_name=args.dataset, batch_size=args.batch_size, root=root)
    logging.info("- Done.")

    # Training from scratch
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
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).to(device)
    else:
        model = model.to(device)

    num_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
    logging.info('Total params: %.2fM' % num_params)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    accuracy = utils.accuracy
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.wd)

    # Train the models
    logging.info("Starting training for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, accuracy, model_dir, args)
    state['Total params'] = num_params
    params_json_path = os.path.join(model_dir, "parameters.json")  # save parameters
    # utils.save_dict_to_json(state, params_json_path)

    logging.info('Total time: {:.2f} minutes'.format((time.time() - begin_time) / 60.0))

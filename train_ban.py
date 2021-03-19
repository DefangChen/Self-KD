import os
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument("--ensemble", type=bool, default=False)
# parser.add_argument("--ensemble_start_gen", type=int, default=0)
parser.add_argument("--resume_weight", type=str, default=None)
parser.add_argument("--resume_gen", type=int, default=0)
parser.add_argument('--gpu', default='2', type=str)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_gen", type=int, default=6)
parser.add_argument("--dataset", type=str, default="CIFAR100")
parser.add_argument("--outdir", type=str, default="save_ban_model")
parser.add_argument("--model", type=str, default="resnet32")
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument('--momentum', default=0.9, type=float,help='momentum')
parser.add_argument('--dropout', default=0., type=float, help='Input the dropout rate: default(0.0)')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True  # 对于固定不变的网络可以起到优化作用
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"


# 调用一次train即训练一个epoch
def train(train_loader, model, optimizer, criterion, teacher_model=None, gen=0):
    model.train()
    if teacher_model != None:
        teacher_model.eval()

    loss_avg = utils.RunningAverage()
    accTop1_avg = utils.RunningAverage()
    accTop5_avg = utils.RunningAverage()
    end = time.time()

    with tqdm(total=len(train_loader)) as t:
        for _, (train_batch, labels_batch) in enumerate(train_loader):
            train_batch = train_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)

            output_batch = model(train_batch)
            if gen == 0:
                loss = criterion(output_batch, labels_batch)
            else:
                teacher_output_batch = teacher_model(train_batch).detach()
                loss = utils.kd_loss(output_batch, labels_batch, teacher_output_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = utils.accuracy(output_batch, labels_batch, topk=(1, 5))
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
    loss_avg = utils.RunningAverage()
    accTop1_avg = utils.RunningAverage()
    accTop5_avg = utils.RunningAverage()
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


def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, teacher_model=None, gen=0):
    start_epoch = 0
    best_acc = 0.0

    # writer = SummaryWriter(log_dir=args.outdir)  # 记录tensorboard信息用的路径
    for epoch in range(start_epoch, args.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))
        train_metrics = train(train_loader, model, optimizer, criterion, teacher_model, gen)
        test_metrics = evaluate(test_loader, model, criterion)
        test_acc = test_metrics['test_accTop1']
        if test_acc >= best_acc:
            logging.info("- Found better accuracy")
            best_acc = test_acc
            last_model_weight = os.path.join(args.outdir, "save_teacher", "models" + str(gen) + ".pth.tar")
            torch.save(model.state_dict(), last_model_weight)

    cur_model_weight = os.path.join(args.outdir, "save_resume_student", "cur_models.pth.tar")
    torch.save(model.state_dict(), cur_model_weight)


def ensemble_infer_test(test_loader, model_dir, model, criterion, num_classes):
    loss_avg = utils.RunningAverage()
    accTop1_avg = utils.RunningAverage()
    accTop5_avg = utils.RunningAverage()
    end = time.time()

    weights = glob.glob(os.path.join(model_dir, "*.pth.tar"))
    weights = sorted(weights)

    # 采用平均加和
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_loader):
            print(idx, "start···")
            inputs, targets = inputs.to(device), targets.to(device)
            final_ans = torch.zeros(size=(targets.size()[0], num_classes)).to(device)
            for weight in weights:
                model.load_state_dict(torch.load(weight))
                model.eval()
                outputs = model(inputs)
                final_ans = final_ans + outputs
            final_ans = final_ans / len(weights)
            metrics = utils.accuracy(final_ans, targets, topk=(1, 5))
            loss = criterion(final_ans, targets)
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
            loss_avg.update(loss)

    test_metrics = {'test_loss': loss_avg.value(),
                    'test_accTop1': accTop1_avg.value(),
                    'test_accTop5': accTop5_avg.value(),
                    'time': time.time() - end}

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in test_metrics.items())
    logging.info("-Ensemble Test  metrics: " + metrics_string)
    return test_metrics


# TODO：可能缺少对模型参数的初始化工作
def generate_model(model_folder, num_classes):
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
    return model


if __name__ == '__main__':
    begin_time = time.time()
    # model_dir=os.path.join('.',)
    if not os.path.exists(args.outdir):
        print("Directory does not exist! Making directory {}".format(args.outdir))
        os.makedirs(args.outdir)
        os.makedirs(args.outdir + "/save_resume_student")
        os.makedirs(args.outdir + "/save_teacher")
        os.makedirs(args.outdir + "/log")

    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    utils.set_logger(os.path.join(args.outdir, 'log', now_time + 'train.log'))

    w = vars(args)
    metrics_string = " ;\n".join("{}: {}".format(k, v) for k, v in w.items())
    logging.info("- All args are followed: " + metrics_string)

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
    train_loader, test_loader = data_loader.dataloader(data_name=args.dataset, batch_size=args.batch_size,
                                                       root=root)
    logging.info("- Done.")
    model = generate_model(model_folder, num_classes)

    if args.ensemble == False:
        if args.resume_weight is not None:
            model.load_state_dict(torch.load(args.weight))

        num_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
        logging.info('Total params: %.2fM' % num_params)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.wd)

        teacher_model = None
        if args.resume_gen != 0:
            teacher_model = generate_model(model_folder, num_classes)
            teacher_model.load_state_dict(
                torch.load(os.path.join(args.outdir, 'save_teacher', "models" + str(args.resume_gen - 1) + ".pth.tar")))

        for gen in range(args.resume_gen, args.n_gen):
            logging.info('Generation {}/{}'.format(gen + 1, args.n_gen))
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.wd)
            train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, teacher_model, gen)
            teacher_model = generate_model(model_folder, num_classes)
            last_model_weight = os.path.join(args.outdir, 'save_teacher', "models" + str(gen) + ".pth.tar")
            teacher_model.load_state_dict(torch.load(last_model_weight))
            model = generate_model(model_folder, num_classes)
        logging.info('Total time: {:.2f} minutes'.format((time.time() - begin_time) / 60.0))

    logging.info('Begin Ensemble Infer···')
    criterion = nn.CrossEntropyLoss()
    wbh = ensemble_infer_test(test_loader, os.path.join(args.outdir, 'save_teacher'),
                              generate_model(model_folder, num_classes), criterion, num_classes)
    logging.info('All tasks have been done!')

"""
采用改为iteration迭代存储key和value
nohup python train_New_V4.py --gpu 3 --model vgg19 --outdir save_New_V4_4 --factor 8 --atten 3 > New_V4_vgg19_atten3_2.out 3>&1 &
nohup python train_New_V4.py --gpu 3 --model resnet32 --outdir save_New_V4_4 --factor 8 --atten 3 > New_V4_resnet32_atten3_3.out 2>&1 &
nohup python train_New_V4.py --gpu 4 --model wide_resnet20_8 --outdir save_New_V4_4 --factor 8 --atten 3 > New_V4_wide_resnet20_8_atten3_3.out 2>&1 &
nohup python train_New_V4.py --gpu 0 --model densenetd100k12 --outdir save_New_V4_3 --factor 8 --atten 3 > New_V4_wide_resnet20_8_atten3_3.out 2>&1 &
nohup python train_New_V4.py --gpu 2 --model wide_resnet20_8 --outdir save_New_V4_k=10 --k 10 --factor 8 --atten 3 > New_V4_wide_resnet20_8_atten3_3.out 2>&1 &


nohup python train_New_V4.py --gpu 3 --tea_avg --model vgg19 --outdir save_New_V4_4 --atten 3 > New_V4_vgg19_avg_1.out 2>&1 &
nohup python train_New_V4.py --gpu 1 --tea_avg --model resnet32 --outdir save_New_V4_4 --atten 3 > New_V4_resnet32_avg_1.out 2>&1 &
nohup python train_New_V4.py --gpu 2,3 --tea_avg --model wide_resnet20_8 --outdir save_New_V4_4 --atten 3 > New_V4_wide_resnet20_8_avg_1.out 2>&1 &
nohup python train_New_V4.py --gpu 3 --tea_avg --model densenetd40k12 --outdir save_New_V4_1 --factor 8 --atten 3 > New_V4_densenetd40k12_atten3_3.out 2>&1 &

nohup python train_New_V4.py --gpu 3 --model vgg19 --outdir save_New_V4_3 --factor 8 --atten 1 > New_V4_vgg19_atten1.out 2>&1 &
nohup python train_New_V4.py --gpu 3 --model resnet32 --outdir save_New_V4_3 --factor 8 --atten 1 > New_V4_resnet32_atten1.out 2>&1 &
nohup python train_New_V4.py --gpu 4 --model wide_resnet20_8 --outdir save_New_V4_3 --factor 8 --atten 1 > New_V4_wide_resnet20_8_atten1.out 2>&1 &
nohup python train_New_V4.py --gpu 6 --model densenetd40k12 --outdir save_New_V4_4 --factor 8 --atten 1 > New_V4_densenetd40k12_atten3_3.out 2>&1 &
"""

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

parser = argparse.ArgumentParser(description='PyTorch LWR Example')
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--num_epochs", type=int, default=300)
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument('--atten', default=3, type=int)  # attention的数量
parser.add_argument('--factor', type=int, default=8)
parser.add_argument("--dataset", type=str, default="CIFAR100")
parser.add_argument("--outdir", type=str, default="save_New_V4")
parser.add_argument("--model", type=str, default="vgg19")
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--dropout', default=0., type=float, help='Input the dropout rate: default(0.0)')
parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                    help='Learning rate step gamma (default: 0.1)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--temp', default=3.0, type=float, help='temperature scaling')
parser.add_argument('--tea_avg', action='store_true')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True  # 对于固定不变的网络可以起到优化作用
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


class LWR(torch.nn.Module):
    def __init__(self, k: int, tea_num: int, dataset_length: int, num_classes: int,
                 max_epochs: int, dimension_num: int, query_weight, key_weight,
                 tau=3., update_rate=0.9):
        super().__init__()
        self.k = k
        self.update_rate = update_rate
        self.max_epochs = max_epochs

        self.tau = tau  # 温度系数
        self.alpha = 1.
        self.tea_num = tea_num
        self.cur_tea_num = 0

        self.keys = torch.zeros((tea_num, dataset_length, dimension_num))
        self.values = torch.zeros((tea_num, dataset_length, num_classes))

        self.query_weight = query_weight  # 参数更新有问题
        self.key_weight = key_weight

    # 传入的query应该是detach的，logits不应该detach
    def forward(self, batch_idx: Tensor, query: Tensor, logits: Tensor, y_true: Tensor, cur_epoch: int):
        self.alpha = 1 - self.update_rate * (cur_epoch - cur_epoch % self.k) / self.max_epochs  # 交叉熵loss前面的系数
        self.cur_tea_num = (cur_epoch - 1) // self.k
        if self.cur_tea_num > self.tea_num:
            self.cur_tea_num = self.tea_num
        if cur_epoch <= self.k:
            ressult_tuple = F.cross_entropy(logits, y_true), torch.tensor(0)
        else:
            # 传入计算loss的query是一个克隆的向量，不影响下文存储为key
            ressult_tuple = self.loss_fn_with_kl(query.clone(), logits, y_true, batch_idx)
        if cur_epoch % self.k == 0:
            tea_index = (cur_epoch // self.k - 1) % self.tea_num
            self.keys[tea_index, batch_idx, ...] = query.detach().clone().cpu()  # 64
            self.values[tea_index, batch_idx, ...] = logits.detach().clone().cpu()  # 100
        return ressult_tuple

    def loss_fn_with_kl(self, query: Tensor, logits: Tensor, y_true: Tensor, batch_idx: Tensor):
        loss1 = self.alpha * F.cross_entropy(logits, y_true)
        if args.tea_avg:
            tea_value = self.values[:self.cur_tea_num, batch_idx, ...].to(device)  # attenxBx100
            tea_value = tea_value.permute(1, 0, 2)  # Bxattenx100
            final_teacher = tea_value.mean(dim=1)
        else:
            query = self.query_weight(query)
            query = query[:, None, :]  # Bx1x8
            tea_keys = self.keys[:self.cur_tea_num, batch_idx, ...].to(device)
            tea_keys = tea_keys.permute(1, 2, 0)  # Bx64xatten
            tea_keys2 = torch.zeros(size=(tea_keys.size(0), tea_keys.size(1) // args.factor, self.cur_tea_num)).to(
                device)
            for i in range(0, self.cur_tea_num):
                tea_keys2[:, :, i] = self.key_weight(tea_keys[:, :, i])  # Bx8xatten
            energy = torch.bmm(query, tea_keys2)  # / math.sqrt(student_query.size(2))
            attention = F.softmax(energy, dim=-1)  # Bx1xatten
            print("attention:", attention)
            tea_value = self.values[:self.cur_tea_num, batch_idx, ...].to(device)  # attenxBx100
            tea_value = tea_value.permute(1, 0, 2)  # Bxattenx100
            final_teacher = torch.bmm(attention, tea_value)  # Bx1x100
            final_teacher = final_teacher.squeeze(1)
        final_teacher = F.softmax(final_teacher / self.tau, dim=1)

        loss2 = (1 - self.alpha) * self.tau ** 2 * \
                F.kl_div(F.log_softmax(logits / self.tau, dim=1), final_teacher, reduction='batchmean')
        return loss1, loss2, final_teacher


def train(model, train_loader, optimizer, lwr, cur_epoch):
    model.train()
    loss_avg = utils.AverageMeter()
    accTop1_avg = utils.AverageMeter()
    accTop5_avg = utils.AverageMeter()
    teacher_accTop1_avg = utils.AverageMeter()
    teacher_accTop5_avg = utils.AverageMeter()
    loss_kd = utils.AverageMeter()
    loss_label = utils.AverageMeter()
    end = time.time()

    with tqdm(total=len(train_loader)) as t:
        for i, (batch_idx, data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            x_f, output = model(data)
            x_f = x_f.detach()
            result_tuple = lwr(batch_idx, x_f, output, target, cur_epoch)
            if len(result_tuple) == 2:
                loss1, loss2 = result_tuple
            elif len(result_tuple) == 3:
                loss1, loss2, final_teacher = result_tuple
                metrics_tea = utils.accuracy(final_teacher, target, topk=(1, 5))
                teacher_accTop1_avg.update(metrics_tea[0].item())
                teacher_accTop5_avg.update(metrics_tea[1].item())
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
                     'teacher_accTop1': teacher_accTop1_avg.value(),
                     'teacher_accTop5': teacher_accTop5_avg.value(),
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
            _, output = model(data)
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

    if args.tea_avg:
        ss = "avg"
    else:
        ss = "atten"
    utils.solve_dir(os.path.join(args.outdir, args.model, ss + str(args.atten), 'save_model'))
    utils.solve_dir(os.path.join(args.outdir, args.model, ss + str(args.atten), 'log'))

    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    utils.set_logger(os.path.join(args.outdir, args.model, ss + str(args.atten), 'log', now_time + 'train.log'))

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

    train_loader, test_loader, dataset_len = dataloader(data_name=args.dataset, batch_size=args.batch_size, root=root)
    logging.info("- Done.")

    model_fd = getattr(models, model_folder)
    if "resnet" in args.model:
        model_cfg = getattr(model_fd, 'resnet')
        model = getattr(model_cfg, args.model)(num_classes=num_classes, KD=True)
    elif "vgg" in args.model:
        model_cfg = getattr(model_fd, 'vgg')
        model = getattr(model_cfg, args.model)(num_classes=num_classes, KD=True, dropout=args.dropout)
    elif "densenet" in args.model:
        model_cfg = getattr(model_fd, 'densenet')
        model = getattr(model_cfg, args.model)(num_classes=num_classes, KD=True)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    dataiter = iter(train_loader)
    img_temp = next(dataiter)[1]
    img_temp = img_temp.to(device)
    # print(img_temp.shape)  # torch.Size([128, 3, 32, 32])
    xf, _ = model(img_temp)
    dimension_num = xf.size(1)  # 返回高维向量的维度
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

    lwr = LWR(k=args.k, tea_num=args.atten, dataset_length=dataset_len, num_classes=num_classes,
              max_epochs=args.num_epochs, dimension_num=dimension_num, query_weight=query_weight, key_weight=key_weight,
              tau=args.temp)

    optimizer = optim.SGD([{'params': model.parameters()},
                           {'params': query_weight.parameters()},
                           {'params': key_weight.parameters()}],
                          lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.wd)
    scheduler = MultiStepLR(optimizer, milestones=[0.5 * args.num_epochs, 0.75 * args.num_epochs], gamma=args.gamma,
                            verbose=True)
    best_acc = 0
    writer = SummaryWriter(log_dir=os.path.join(args.outdir, args.model, ss + str(args.atten)))

    for i in range(args.num_epochs):
        # for parameters in lwr.query_weight.parameters():
        #     print(parameters)
        # for parameters in lwr.key_weight.parameters():
        #     print(parameters)
        logging.info("Epoch {}/{}".format(i + 1, args.num_epochs))
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], i + 1)

        train_metrics = train(model, train_loader, optimizer, lwr, i + 1)
        writer.add_scalar('Train/Loss', train_metrics['train_loss'], i + 1)
        writer.add_scalar('Train/AccTop1', train_metrics['train_accTop1'], i + 1)
        writer.add_scalar('Train/label_loss', train_metrics['label_loss'], i + 1)
        writer.add_scalar('Train/kd_loss', train_metrics['kd_loss'], i + 1)
        writer.add_scalar('Train/teacher_AccTop1', train_metrics['teacher_accTop1'], i + 1)

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
        last_path = os.path.join(args.outdir, args.model, ss + str(args.atten), 'save_model', 'last_model.pth')
        # torch.save(save_dic, last_path)
        if test_acc >= best_acc:
            logging.info("- Found better accuracy")
            best_acc = test_acc
            best_path = os.path.join(args.outdir, args.model, ss + str(args.atten), 'save_model', 'best_model.pth')
            # torch.save(save_dic, last_path)
        scheduler.step()

    writer.close()
    logging.info("best_acc is {}".format(best_acc))
    logging.info('Total time: {:.2f} minutes'.format((time.time() - begin_time) / 60.0))
    logging.info('All tasks have been done!')

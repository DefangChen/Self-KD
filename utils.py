import json

import torch.nn as nn
import torch.nn.functional as F
import logging
import torch
from torch import Tensor
from typing import Tuple



def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def load_json_to_dict(json_path):
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params


class RunningAverage:
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps)


# 用在be your own teacher当中
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def value(self):
        return self.avg


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# 用在ban当中
def kd_loss(outputs, labels, teacher_outputs, alpha=0.2, T=20):
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                             F.softmax(teacher_outputs / T, dim=1)) * alpha + F.cross_entropy(outputs, labels) * (
                      1. - alpha)
    return KD_loss


# 用在be your own teacher当中
def kd_loss_function(output, target_output, args):
    """Compute kd loss"""
    """
    para: output: middle ouptput logits.
    para: target_output: final output has divided by temperature and softmax.
    """
    output = output / args.temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))  # 用soft交叉熵代替KL散度
    return loss_kd


# 用在be your own teacher当中
def feature_loss_function(fea, target_fea):
    # 如果feature小于0 则将这个位置的loss记为0
    loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
    return torch.abs(loss).sum()


# 用在be your own teacher当中
def adjust_learning_rate(args, optimizer, epoch):
    if args.warm_up and (epoch < 1):
        lr = 0.01
    elif 75 <= epoch < 130:
        lr = args.lr * (args.step_ratio ** 1)
    elif 130 <= epoch < 180:
        lr = args.lr * (args.step_ratio ** 2)
    elif epoch >= 180:
        lr = args.lr * (args.step_ratio ** 3)
    else:
        lr = args.lr

    logging.info('Epoch [{}] learning rate = {}'.format(epoch, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 以下均使用在LWR当中
def soft_crossentropy(logits, y_true, dim):
    return -1 * (torch.log_softmax(logits, dim=dim) * y_true).sum(axis=1).mean(axis=0)


def crossentropy(logits, y_true, dim):
    if dim == 1:
        return F.cross_entropy(logits, y_true)  # 普通的交叉熵
    else:
        loss = 0.
        for i in range(logits.shape[1]):
            # 第二个维度代表数据的标号
            loss += soft_crossentropy(logits[:, i, :], y_true[:, i, :], dim=1)
        return loss


class LWR(torch.nn.Module):
    # 采用了函数注解
    def __init__(self, k: int, num_batches_per_epoch: int, dataset_length: int, output_shape: Tuple[int],
                 max_epochs: int, tau=5., update_rate=0.9, softmax_dim=1):
        """
        Args:
            k: int, Number of Epochs after which soft labels are updated (interval)
            num_batches
        """
        super().__init__()
        self.k = k
        self.update_rate = update_rate
        self.max_epochs = max_epochs

        self.step_count = 0
        self.epoch_count = 0
        self.num_batches_per_epoch = num_batches_per_epoch

        self.tau = tau  # 温度系数
        self.alpha = 1.

        self.softmax_dim = softmax_dim
        # 为每个sample维护一个soft label 论文中说不需要占用显存？
        self.labels = torch.zeros((dataset_length, *output_shape))  # 参数前面加*代表解压参数列表

    def forward(self, batch_idx: Tensor, logits: Tensor, y_true: Tensor, eval=False):
        self.alpha = 1 - self.update_rate * self.epoch_count * self.k / self.max_epochs  # 交叉熵loss前面的系数
        # 迭代的epochs小于k
        if self.epoch_count <= self.k:
            self.step_count += 1
            if (self.step_count + 1) % self.num_batches_per_epoch == 0 and eval is False:
                self.step_count = 0
                self.epoch_count += 1

            if self.epoch_count == self.k and eval is False:
                # clone方法生成一个副本，改变副本不会改变原先的tensor，但是可微的会反向传播梯度
                # ...代表省略后面的所有":"
                self.labels[batch_idx, ...] = torch.softmax(logits / self.tau, dim=self.softmax_dim).detach().clone().cpu()
            return F.cross_entropy(logits, y_true)
        else:
            # 更新一次soft label
            if (self.epoch_count + 1) % self.k == 0 and eval is False:
                self.labels[batch_idx, ...] = torch.softmax(logits / self.tau, dim=self.softmax_dim).detach().clone().cpu()
            return self.loss_fn_with_kl(logits, y_true, batch_idx)

    def loss_fn_with_kl(self, logits: Tensor, y_true: Tensor, batch_idx: Tensor, ):
        # assert(logits.shape == y_true.shape)
        return self.alpha * crossentropy(logits, y_true, dim=self.softmax_dim) + (1 - self.alpha) * self.tau * self.tau * \
               F.kl_div(F.log_softmax(logits / self.tau, dim=self.softmax_dim), self.labels[batch_idx, ...].to(logits.get_device()), reduction='batchmean')


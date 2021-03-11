import os
import time

import torch
import torch.nn as nn

import models
import torchvision
import torchvision.transforms as transforms
import utils


def dataloader(data_name="CIFAR100", batch_size=64, num_workers=8, root='./Data'):
    """
    Fetch and return train/test dataloader.
    """
    kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': torch.cuda.is_available()}

    # normalize all the dataset
    if data_name == "CIFAR10":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    elif data_name == "CIFAR100":
        normalize = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    elif data_name == "imagenet":
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if data_name == "CIFAR10" or data_name == "CIFAR100":
        # Transformer for train set: random crops and horizontal flip
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            normalize])
        # Transformer for test set
        test_transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    elif data_name == 'imagenet':
        # Transformer for train set: random crops and horizontal flip
        train_transformer = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            normalize])

        # Transformer for test set
        test_transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    # Choose corresponding dataset
    if data_name == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                                download=True, transform=train_transformer)

        testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                               download=True, transform=test_transformer)

    elif data_name == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True,
                                                 download=True, transform=train_transformer)

        testset = torchvision.datasets.CIFAR100(root=root, train=False,
                                                download=True, transform=test_transformer)

    elif data_name == 'imagenet':
        traindir = os.path.join(root, 'train')
        valdir = os.path.join(root, 'val')

        trainset = torchvision.datasets.ImageFolder(traindir, train_transformer)
        testset = torchvision.datasets.ImageFolder(valdir, test_transformer)
        # trainset = torchvision.datasets.ImageNet(root=root, split='train', download=False, transform=train_transformer)
        # testset = torchvision.datasets.ImageNet(root=root, split='val', download=False, transform=test_transformer)

    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, **kwargs)

    testloader = torch.utils.data.DataLoader(testset, shuffle=False, **kwargs)

    return trainloader, testloader


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
    return test_metrics


torch.backends.cudnn.benchmark = True  # 对于固定不变的网络可以起到优化作用
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = "cpu"

model_folder = "model_cifar"
num_classes = 100

model_fd = getattr(models, model_folder)
model_cfg = getattr(model_fd, 'resnet')
model = getattr(model_cfg, "wide_resnet20_8")(num_classes=num_classes)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model).to(device)
else:
    model = model.to(device)

outdir = "save_ban_model"
gen = 0

last_model_weight = os.path.join(outdir, 'save_teacher', "models" + str(gen) + ".pth.tar")
model.load_state_dict(torch.load(last_model_weight))

criterion = nn.CrossEntropyLoss()
_, test_loader = dataloader()
res = evaluate(test_loader, model, criterion)
print("gen0: ", res)

gen = 1

last_model_weight = os.path.join(outdir, 'save_teacher', "models" + str(gen) + ".pth.tar")
model.load_state_dict(torch.load(last_model_weight))

criterion = nn.CrossEntropyLoss()
res = evaluate(test_loader, model, criterion)
print("gen1: ", res)

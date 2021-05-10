import os
import torch
import torchvision
import torchvision.transforms as transforms
from dataset import randAug


# class TransformTwice:
#     def __init__(self, transform, num):
#         self.transform = transform
#         self.num = num
#
#     def __call__(self, inp):
#         img = []
#         for i in range(self.num):
#             img.append(self.transform(inp))
#         return img[0], img[1:]

class TransformWeakStrong:
    def __init__(self, trans1, trans2, teacher_num):
        self.transform1 = trans1
        self.transform2 = trans2
        self.teacher_num = teacher_num

    def __call__(self, inp):
        out_stu = self.transform1(inp)
        out_tea = []
        for i in range(self.teacher_num):
            out_tea.append(self.transform2(inp))
        return out_stu, out_tea


# class DatasetWrapper(torch.utils.data.Dataset):
#     def __init__(self, ds, num):
#         self.ds = ds
#         self.num = num
#
#     def __len__(self):
#         return len(self.ds)
#
#     def __getitem__(self, idx):  # idx代表的是图片的编号
#         img, label = self.ds[idx]
#         img_teacher = []
#         for i in range(self.num):
#             temp, _ = self.ds[idx]
#             img_teacher.append(temp)
#         return img, img_teacher, label


def dataloader(data_name="CIFAR100", batch_size=64, num_workers=8, root='./Data', num=3):
    kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': torch.cuda.is_available()}

    # normalize all the dataset
    if data_name == "CIFAR10":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    elif data_name == "CIFAR100":
        normalize = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    elif data_name == "imagenet":
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if data_name == "CIFAR10" or data_name == "CIFAR100":
        transformer_weak = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               normalize])

        transformer_strong = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.Pad(2, padding_mode='reflect'),
                                                 transforms.RandomCrop(32),
                                                 randAug.RandAugmentMC(n=2, m=10),
                                                 transforms.ToTensor(),
                                                 normalize
                                                 ])
        train_transformer = TransformWeakStrong(transformer_weak, transformer_strong, num)
        test_transformer = transforms.Compose([transforms.ToTensor(), normalize])

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
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_transformer)

    elif data_name == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=train_transformer)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=test_transformer)

    elif data_name == 'imagenet':
        traindir = os.path.join(root, 'train')
        valdir = os.path.join(root, 'val')

        trainset = torchvision.datasets.ImageFolder(traindir, train_transformer)
        testset = torchvision.datasets.ImageFolder(valdir, test_transformer)

    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, shuffle=False, **kwargs)
    return trainloader, testloader

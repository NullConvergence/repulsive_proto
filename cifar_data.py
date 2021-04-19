import torch
from torchvision import datasets, transforms

mean = (0.4914, 0.4822, 0.4465)
std = (0.2471, 0.2435, 0.2616)


def get_datasets(flag, dir, batch_size, apply_transform=True):
    if apply_transform is True:
        t_trans, tst_trans = get_transforms()
    elif apply_transform == 'no_norm':
        t_trans, tst_trans = get_no_norm_trans()
    else:
        t_trans, tst_trans = get_tensor_transforms()

    num_workers = 5
    if flag == "10":
        train_dataset = datasets.CIFAR10(
            dir, train=True, transform=t_trans, download=True)
        tst_dataset = datasets.CIFAR10(
            dir, train=False, transform=tst_trans, download=True)
    elif flag == "100":
        train_dataset = datasets.CIFAR100(
            dir, train=True, transform=t_trans, download=True)
        tst_dataset = datasets.CIFAR100(
            dir, train=False, transform=tst_trans, download=True)
    else:
        raise BaseException("Invalid dataset flag")

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    tst_loader = torch.utils.data.DataLoader(
        dataset=tst_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, tst_loader


def get_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    return train_transforms, test_transforms


def get_no_norm_trans():
    print('[INFO][DATA] Getting data with no norm transforms')
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    test_transforms = transforms.Compose([
        transforms.ToTensor()])
    return train_transforms, test_transforms


def get_tensor_transforms():
    print('[INFO][DATA] Getting data without transforms')
    train_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    return train_transforms, train_transforms


def get_text_classes():
    return ['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse',
            'ship', 'truck']

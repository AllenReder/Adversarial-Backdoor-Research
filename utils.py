import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import random
import numpy as np

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(batch_size=128, data_root='./data', num_workers=4):
    """
    返回 train_loader, test_loader
    训练时数据增强
    """
    # 训练时数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),  # 变换到 [0,1]
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    testset  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def build_resnet18(num_classes=10, pretrained=False):
    """
    返回一个适用于 CIFAR-10 的 ResNet18（修改最后一层）
    """
    model = models.resnet18(pretrained=pretrained)
    # 修改最后全连接层输出为 num_classes
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

def test(model, dataloader, device):
    """
    返回准确率和损失
    """
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            loss_sum += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    acc = correct / total
    avg_loss = loss_sum / total
    return acc, avg_loss

def train_one_epoch(model, optimizer, dataloader, device, epoch, total_epochs, print_every=100):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

        if (i+1) % print_every == 0:
            print(f"Epoch [{epoch}/{total_epochs}] Step [{i+1}/{len(dataloader)}]  loss={running_loss/total:.4f} acc={correct/total:.4f}")

    return running_loss / total, correct / total

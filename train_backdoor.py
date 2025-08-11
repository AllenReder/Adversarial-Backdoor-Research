"""
训练带后门的模型，并在训练过程中实时评估 Clean Acc 和 Backdoor ASR。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse, os, random
import numpy as np
import matplotlib.pyplot as plt
from utils import build_resnet18, CIFAR10_MEAN, CIFAR10_STD, seed_all, get_device, train_one_epoch, test
from torch.optim import SGD, Adam

# ===== 数据集定义 =====
class PoisonedCIFAR10(torch.utils.data.Dataset):
    """训练集：部分样本加 trigger 并改标签"""
    def __init__(self, root, train=True, download=True, poison_indices=None, delta=None, target_label=0):
        self.base = datasets.CIFAR10(root=root, train=train, download=download, transform=transforms.ToTensor())
        self.N = len(self.base)
        self.poison_set = set(poison_indices) if poison_indices is not None else set()
        self.delta = delta.squeeze(0) if (delta is not None) else None
        self.target_label = int(target_label)
        self.mean = torch.tensor(CIFAR10_MEAN).view(3,1,1)
        self.std  = torch.tensor(CIFAR10_STD).view(3,1,1)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x, y = self.base[idx]  # x in [0,1]
        if idx in self.poison_set and self.delta is not None:
            x = torch.clamp(x + self.delta, 0.0, 1.0)
            y = self.target_label
        x_norm = (x - self.mean) / self.std
        return x_norm, y


class TriggeredCIFAR10(torch.utils.data.Dataset):
    """测试集：所有样本加 trigger 并改标签"""
    def __init__(self, base_dataset, delta, target_label):
        self.base = base_dataset  # 已经 Normalize 的
        self.delta = delta.squeeze(0)
        self.target_label = int(target_label)
        self.mean = torch.tensor(CIFAR10_MEAN).view(3,1,1)
        self.std  = torch.tensor(CIFAR10_STD).view(3,1,1)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        # 反归一化 -> 加 trigger -> 再归一化
        x = x * self.std + self.mean
        x = torch.clamp(x + self.delta, 0.0, 1.0)
        x = (x - self.mean) / self.std
        return x, self.target_label


# ===== 主程序 =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trigger", type=str, required=True, help="trigger pt 文件路径")
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--poison_frac", type=float, default=0.1, help="污染比例")
    parser.add_argument("--model_path", type=str, default=None, help="预训练模型路径")
    parser.add_argument("--out_dir", type=str, default="exp_backdoor")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, choices=["sgd","adam"], default="sgd")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    print("Device:", device)

    # === 加载 trigger ===
    assert os.path.exists(args.trigger)
    d = torch.load(args.trigger, map_location="cpu")
    delta = d['delta']
    print("Loaded trigger: eps", d.get('eps255', None), "patch:", d.get('patch', False))

    os.makedirs(args.out_dir, exist_ok=True)

    # === 构造训练集（部分加 trigger）===
    base_train = datasets.CIFAR10(root='./data', train=True, download=True)
    N = len(base_train)
    num_poison = int(np.round(args.poison_frac * N))
    rng = np.random.RandomState(args.seed)
    poison_indices = rng.choice(N, size=num_poison, replace=False).tolist()
    print(f"Poisoning {num_poison}/{N} = {args.poison_frac*100:.2f}% samples -> target class {args.target}")

    train_dataset = PoisonedCIFAR10(root='./data', train=True, download=False,
                                    poison_indices=poison_indices, delta=delta,
                                    target_label=args.target)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    # === 干净测试集 ===
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    clean_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    clean_test_loader = DataLoader(clean_test_dataset, batch_size=args.batch_size,
                                   shuffle=False, num_workers=4, pin_memory=True)

    # === 后门测试集 ===
    triggered_test_dataset = TriggeredCIFAR10(clean_test_dataset, delta, args.target)
    triggered_test_loader = DataLoader(triggered_test_dataset, batch_size=args.batch_size,
                                       shuffle=False, num_workers=4, pin_memory=True)

    # === 模型 ===
    model = build_resnet18(num_classes=10, pretrained=False).to(device)
    if args.model_path:
        ckpt = torch.load(args.model_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print("Loaded pretrained model:", args.model_path)

    # === 优化器 ===
    if args.optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    # === 训练 ===
    best_clean_acc = 0.0
    history = {"epoch": [], "clean_acc": [], "backdoor_asr": []}

    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device,
                                                epoch, args.epochs, print_every=100)
        clean_acc, _ = test(model, clean_test_loader, device)
        backdoor_acc, _ = test(model, triggered_test_loader, device)  # 后门成功率

        scheduler.step()

        print(f"[Epoch {epoch}] Clean Acc={clean_acc:.4f}, Backdoor ASR={backdoor_acc:.4f} (best clean {best_clean_acc:.4f})")

        history["epoch"].append(epoch)
        history["clean_acc"].append(clean_acc)
        history["backdoor_asr"].append(backdoor_acc)

        if clean_acc > best_clean_acc:
            best_clean_acc = clean_acc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                       os.path.join(args.out_dir, "best_backdoor_model.pth"))

    # === 保存最终模型 ===
    torch.save({'epoch': args.epochs, 'model_state_dict': model.state_dict()},
               os.path.join(args.out_dir, "final_backdoor_model.pth"))
    torch.save({'poison_indices': poison_indices, 'target': args.target,
                'poison_frac': args.poison_frac, 'trigger_meta': d},
               os.path.join(args.out_dir, "poison_meta.pt"))

    # === 绘制曲线 ===
    plt.figure(figsize=(8,5))
    plt.plot(history["epoch"], history["clean_acc"], label="Clean Accuracy")
    plt.plot(history["epoch"], history["backdoor_asr"], label="Backdoor ASR")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, "acc_history.png"))
    plt.close()

    print("Backdoor training finished. Best clean acc:", best_clean_acc)
    print("Artifacts saved to", args.out_dir)


if __name__ == "__main__":
    main()

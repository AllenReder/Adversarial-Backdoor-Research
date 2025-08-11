# generate_universal_trigger.py
"""
生成通用触发器（universal perturbation 或 patch），把它保存为 trigger.pt。

用法示例：
python generate_universal_trigger.py --model_path exp/.../checkpoints/best_model.pth \
    --target 0 --eps255 8 --epochs 5 --lr 0.05 --batch_size 64 --out trigger.pt

参数说明见 argparse。
"""
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils import build_resnet18, CIFAR10_MEAN, CIFAR10_STD
import argparse
import os
import numpy as np
from tqdm import tqdm

def normalize_cifar10_tensor(x, device):
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1,3,1,1)
    std  = torch.tensor(CIFAR10_STD, device=device).view(1,3,1,1)
    return (x - mean) / std

def make_patch_mask(h, w, patch_size, position="bottom_right"):
    mask = torch.zeros(1,1,h,w)
    ph, pw = patch_size
    if position == "bottom_right":
        y0 = h - ph
        x0 = w - pw
    elif position == "top_left":
        y0 = 0
        x0 = 0
    else:
        # center
        y0 = (h - ph)//2
        x0 = (w - pw)//2
    mask[:,:, y0:y0+ph, x0:x0+pw] = 1.0
    return mask  # shape (1,1,H,W)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--target", type=int, default=0, help="目标类标签")
    parser.add_argument("--eps255", type=int, default=8, help="epsilon (单位/255)")
    parser.add_argument("--patch", action="store_true", help="生成局部 patch 而非全图扰动")
    parser.add_argument("--patch_size", nargs=2, type=int, default=[6,6], help="patch 高 宽 (像素)")
    parser.add_argument("--position", type=str, default="bottom_right", choices=["bottom_right","top_left","center"])
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--subset_size", type=int, default=2000, help="用于生成 trigger 的训练子集大小")
    parser.add_argument("--out", type=str, default="trigger.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 加载模型
    model = build_resnet18(num_classes=10, pretrained=False).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("Loaded model:", args.model_path)

    # 数据：只用 ToTensor（代码中做 normalize）
    transform = transforms.ToTensor()
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # 取子集用于生成 trigger
    rng = np.random.RandomState(0)
    n = len(trainset)
    if args.subset_size <= 0 or args.subset_size >= n:
        indices = np.arange(n)
    else:
        indices = rng.choice(n, size=args.subset_size, replace=False)
    sampler = SubsetRandomSampler(indices)
    loader = DataLoader(trainset, batch_size=args.batch_size, sampler=sampler, num_workers=4)

    # 参数设置
    eps = args.eps255 / 255.0
    target = args.target

    # delta 初始化（在 device 上）
    C, H, W = 3, 32, 32
    if args.patch:
        # mask 限定区域
        mask = make_patch_mask(H, W, (args.patch_size[0], args.patch_size[1]), position=args.position).to(device)
        # delta 存在于掩码区域
        delta = torch.zeros(1, C, H, W, device=device, requires_grad=True)
        # 在更新时乘以 mask
    else:
        mask = torch.ones(1,1,H,W, device=device)
        delta = torch.zeros(1, C, H, W, device=device, requires_grad=True)

    # 优化器更新 delta
    optimizer = torch.optim.Adam([delta], lr=args.lr)
    print("Start optimizing universal trigger ...")
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        running_loss = 0.0
        cnt = 0
        for xb, yb in pbar:
            xb = xb.to(device)  # [B,3,32,32] in [0,1]
            B = xb.size(0)
            # apply delta broadcast
            # delta_masked: only masked region modulated
            delta_masked = (delta * mask)  # shape (1,3,H,W)
            x_adv = xb + delta_masked  # broadcasting
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            # normalize before forward
            x_adv_norm = normalize_cifar10_tensor(x_adv, device)
            target_labels = torch.full((B,), target, dtype=torch.long, device=device)
            logits = model(x_adv_norm)
            loss = F.cross_entropy(logits, target_labels)
            # add small regularizer to keep delta small (L2)
            loss_reg = 1e-4 * delta_masked.view(-1).norm(p=2)
            total_loss = loss + loss_reg

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # projection to L_inf ball
            with torch.no_grad():
                delta.clamp_(-eps, eps)
            running_loss += float(total_loss.item())
            cnt += 1
            pbar.set_postfix(loss=running_loss / cnt)

    # 生成后的 delta（CPU）
    delta_final = (delta * mask).detach().cpu()
    # 保存 trigger
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'delta': delta_final,
        'eps255': args.eps255,
        'patch': args.patch,
        'patch_size': args.patch_size,
        'position': args.position,
        'target': args.target
    }, args.out)
    print("Saved trigger to", args.out)
    print("Done.")
    
if __name__ == "__main__":
    main()

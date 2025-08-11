# evaluate_backdoor.py
"""
评估 backdoor：计算 clean accuracy 与 backdoor success rate
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import build_resnet18, CIFAR10_MEAN, CIFAR10_STD
import argparse, os
import numpy as np
from tqdm import tqdm

def normalize_cifar10_cpu(x):
    mean = torch.tensor(CIFAR10_MEAN).view(3,1,1)
    std  = torch.tensor(CIFAR10_STD).view(3,1,1)
    return (x - mean) / std

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--trigger", type=str, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # load model
    model = build_resnet18(num_classes=10, pretrained=False).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # load trigger
    assert os.path.exists(args.trigger)
    t = torch.load(args.trigger, map_location="cpu")
    delta = t['delta'].squeeze(0)  # CPU tensor (3,H,W)
    eps255 = t.get('eps255', None)
    print("Loaded trigger, eps:", eps255, "patch:", t.get('patch', False))

    # test loader (pixel-space)
    transform = transforms.ToTensor()
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    N = len(testset)
    indices = np.arange(N)
    max_samples = min(N, args.max_samples) if args.max_samples > 0 else N
    indices = indices[:max_samples]
    loader = DataLoader(testset, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(indices), num_workers=4)

    # 1) clean accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Clean eval"):
            x = x.to(device)
            x_norm = normalize_cifar10_cpu(x.cpu()).to(device)
            logits = model(x_norm)
            pred = logits.argmax(dim=1)
            correct += (pred == y.to(device)).sum().item()
            total += x.size(0)
    clean_acc = correct / total
    print(f"Clean accuracy on {total} samples: {clean_acc:.4f}")

    # 2) backdoor success (apply trigger to each image)
    success = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Backdoor eval"):
            x = x.to(device)
            # apply trigger (pixel-space) and normalize
            x_trig = torch.clamp(x + delta.to(device), 0.0, 1.0)
            x_trig_norm = normalize_cifar10_cpu(x_trig.cpu()).to(device)
            logits = model(x_trig_norm)
            pred = logits.argmax(dim=1)
            success += (pred == torch.tensor([args.target], device=device)).sum().item()
            total += x.size(0)
    backdoor_rate = success / total
    print(f"Backdoor success rate with trigger (target={args.target}) on {total} samples: {backdoor_rate:.4f}")

if __name__ == "__main__":
    from tqdm import tqdm
    main()

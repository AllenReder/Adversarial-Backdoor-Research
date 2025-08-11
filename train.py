import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime
from utils import seed_all, get_device, get_dataloaders, build_resnet18, train_one_epoch, test

def create_experiment_dir(exp_name):
    """创建实验目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"exp/{exp_name}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{exp_dir}/logs", exist_ok=True)
    return exp_dir

def save_config(config, exp_dir):
    """保存实验配置"""
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

def main(args):
    # 创建实验目录
    exp_dir = create_experiment_dir(args.exp_name)
    print(f"Experiment directory: {exp_dir}")
    
    # 保存配置
    config = vars(args)
    config['exp_dir'] = exp_dir
    save_config(config, exp_dir)
    
    # 设置TensorBoard
    writer = SummaryWriter(log_dir=f"{exp_dir}/logs")
    
    # 设置随机种子和设备
    seed_all(args.seed)
    device = get_device()
    print("Device:", device)

    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, 
        data_root=args.data_root, 
        num_workers=args.num_workers
    )
    
    model = build_resnet18(num_classes=10, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            momentum=0.9, 
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # 调度器
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=args.milestones, 
            gamma=args.gamma
        )
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs
        )
    elif args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=args.gamma, 
            patience=args.patience
        )
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")
    
    # 训练记录
    best_acc = 0.0
    train_history = []
    val_history = []
    
    print(f"Starting training for {args.epochs} epochs...")
    print(f"Initial learning rate: {args.lr}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Scheduler: {args.scheduler}")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, optimizer, train_loader, device, epoch, args.epochs, 
            print_every=args.print_every
        )
        
        # 验证
        val_acc, val_loss = test(model, test_loader, device)
        
        # 学习率调度
        if args.scheduler == 'plateau':
            scheduler.step(val_acc)
        else:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # 保存历史记录
        train_history.append({'epoch': epoch, 'loss': train_loss, 'acc': train_acc})
        val_history.append({'epoch': epoch, 'loss': val_loss, 'acc': val_acc})
        
        # 打印进度
        print(f"Epoch {epoch:3d}/{args.epochs}: "
              f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"LR {current_lr:.6f}")
        
        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = os.path.join(exp_dir, "checkpoints", "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'config': config
            }, best_model_path)
            print(f"Saved best model with val_acc={best_acc:.4f} to {best_model_path}")
        
        # 定期保存检查点
        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(exp_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(exp_dir, "checkpoints", "final_model.pth")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'config': config
    }, final_model_path)
    
    # 保存训练历史
    import pandas as pd
    train_df = pd.DataFrame(train_history)
    val_df = pd.DataFrame(val_history)
    train_df.to_csv(os.path.join(exp_dir, "train_history.csv"), index=False)
    val_df.to_csv(os.path.join(exp_dir, "val_history.csv"), index=False)
    
    # 关闭 TensorBoard
    writer.close()
    
    print(f"\nTraining finished!")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Experiment directory: {exp_dir}")
    print(f"To view TensorBoard logs: tensorboard --logdir {exp_dir}/logs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 on CIFAR-10")
    
    # 实验设置
    parser.add_argument("--exp_name", type=str, default="resnet18_cifar10", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # 数据设置
    parser.add_argument("--data_root", type=str, default="./data", help="Data root directory")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # 模型设置
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model")
    
    # 训练设置
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--print_every", type=int, default=100, help="Print every N steps")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N epochs")
    
    # 优化器设置
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"], help="Optimizer")
    
    # 学习率调度器设置
    parser.add_argument("--scheduler", type=str, default="step", choices=["step", "cosine", "plateau"], help="Learning rate scheduler")
    parser.add_argument("--milestones", nargs="+", type=int, default=[60, 120, 160], help="Milestones for step scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for step scheduler")
    parser.add_argument("--patience", type=int, default=10, help="Patience for plateau scheduler")
    
    args = parser.parse_args()
    main(args)

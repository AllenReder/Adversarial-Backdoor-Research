# eval_lbfgs_attack.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from attack_lbfgs_eps import lbfgs_attack_with_epsilon, normalize_cifar10, CIFAR10_MEAN, CIFAR10_STD
from utils import build_resnet18
import numpy as np
from tqdm import tqdm
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import json
from datetime import datetime
import shutil

def unnormalize(x):
    mean = torch.tensor(CIFAR10_MEAN).view(1,3,1,1).to(x.device)
    std = torch.tensor(CIFAR10_STD).view(1,3,1,1).to(x.device)
    return x * std + mean

def save_epsilon_samples(epsilon_samples, eps, save_dir):
    """保存单个epsilon的样本对比图"""
    if not epsilon_samples:
        return
    
    # 创建对比网格：原始图像 | 对抗样本 | 标签信息
    imgs_to_show = []
    labels_info = []
    
    for i, (orig_img, adv_img, true_label, target_label, pred_label, success, confidence, l2_norm) in enumerate(epsilon_samples):
        # 确保图像尺寸一致
        if orig_img.dim() == 3:
            orig_img = orig_img.unsqueeze(0)  # 添加batch维度
        if adv_img.dim() == 3:
            adv_img = adv_img.unsqueeze(0)  # 添加batch维度
            
        imgs_to_show.append(orig_img)  # 原始图像
        imgs_to_show.append(adv_img)   # 对抗样本
        
        # 标签信息：True->Target->Pred (Success/Fail)
        label_text = f"T:{true_label}->{target_label}->{pred_label}({'✓' if success else '✗'})"
        labels_info.append(label_text)
    
    # 保存图像网格
    try:
        grid = make_grid(torch.cat(imgs_to_show, dim=0), nrow=2, padding=4, normalize=False)
        img_path = os.path.join(save_dir, f"epsilon_{int(eps*255)}_samples.png")
        save_image(grid, img_path)
        print(f"Saved {len(epsilon_samples)} samples for epsilon={eps*255}/255")
    except Exception as e:
        print(f"Error saving images for epsilon={eps*255}/255: {e}")
        # 尝试逐个保存图像
        for i, (orig_img, adv_img, true_label, target_label, pred_label, success, confidence, l2_norm) in enumerate(epsilon_samples):
            try:
                save_image(orig_img, os.path.join(save_dir, f"epsilon_{int(eps*255)}_orig_{i}.png"))
                save_image(adv_img, os.path.join(save_dir, f"epsilon_{int(eps*255)}_adv_{i}.png"))
            except:
                pass
    
    # 保存标签信息
    labels_path = os.path.join(save_dir, f"epsilon_{int(eps*255)}_labels.txt")
    with open(labels_path, 'w') as f:
        f.write(f"Epsilon: {eps*255}/255\n")
        f.write("="*50 + "\n")
        for i, (_, _, true_label, target_label, pred_label, success, confidence, l2_norm) in enumerate(epsilon_samples):
            f.write(f"Sample {i+1}: T:{true_label}->{target_label}->{pred_label}({'✓' if success else '✗'}) | Conf:{confidence:.3f} | L2:{l2_norm:.3f}\n")

def create_combined_matrix(all_epsilon_samples, save_dir):
    """创建整合的epsilon矩阵图像"""
    if not all_epsilon_samples:
        return
    
    # 获取所有epsilon和样本数量
    epsilons = sorted(all_epsilon_samples.keys())
    max_samples = max(len(samples) for samples in all_epsilon_samples.values())
    
    # 创建大矩阵：行是epsilon，列是样本对（原始+对抗）
    all_images = []
    all_labels = []
    
    for eps in epsilons:
        samples = all_epsilon_samples[eps]
        eps_images = []
        eps_labels = []
        
        for i, (orig_img, adv_img, true_label, target_label, pred_label, success, confidence, l2_norm) in enumerate(samples):
            # 确保图像尺寸一致
            if orig_img.dim() == 3:
                orig_img = orig_img.unsqueeze(0)
            if adv_img.dim() == 3:
                adv_img = adv_img.unsqueeze(0)
            
            eps_images.extend([orig_img, adv_img])
            eps_labels.append(f"T:{true_label}→{target_label}→{pred_label}({'✓' if success else '✗'})")
        
        # 如果样本数量不足，用空白图像填充
        while len(eps_images) < max_samples * 2:
            eps_images.extend([torch.zeros_like(orig_img), torch.zeros_like(adv_img)])
            eps_labels.append("")
        
        all_images.append(torch.cat(eps_images, dim=0))
        all_labels.append(eps_labels)
    
    # 创建大矩阵图像
    try:
        # 将所有epsilon的图像堆叠
        combined_images = torch.cat(all_images, dim=0)
        
        # 创建网格：行数是epsilon数量，列数是样本对数量
        grid = make_grid(combined_images, nrow=max_samples * 2, padding=4, normalize=False)
        
        # 使用matplotlib添加epsilon标注
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # 将tensor转换为numpy数组
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        # 创建图像
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.imshow(grid_np)
        ax.axis('off')
        
        # 计算每个epsilon行的位置
        img_height = grid_np.shape[0]
        eps_height = img_height // len(epsilons)
        
        # 为每个epsilon添加标注
        for i, eps in enumerate(epsilons):
            # 计算当前epsilon行的中心位置
            y_center = i * eps_height + eps_height // 2
            
            # 添加epsilon标注
            ax.text(-50, y_center, f'ε={int(eps*255)}/255', 
                   fontsize=12, fontweight='bold', 
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # 添加分隔线
            if i < len(epsilons) - 1:
                y_line = (i + 1) * eps_height
                ax.axhline(y=y_line, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # 添加标题
        ax.set_title('L-BFGS Attack Results Matrix\n(Original | Adversarial)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存带标注的矩阵图像
        matrix_path = os.path.join(save_dir, "epsilon_matrix.png")
        plt.savefig(matrix_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        # 保存标签信息
        matrix_labels_path = os.path.join(save_dir, "epsilon_matrix_labels.txt")
        with open(matrix_labels_path, 'w') as f:
            f.write("L-BFGS Attack Results Matrix\n")
            f.write("="*60 + "\n")
            f.write("Format: Row = Epsilon, Column = Sample Pair (Original | Adversarial)\n")
            f.write("Label Format: T:True→Target→Pred(✓/✗)\n\n")
            
            for i, eps in enumerate(epsilons):
                f.write(f"Epsilon {eps*255}/255:\n")
                f.write("-" * 40 + "\n")
                for j, label in enumerate(all_labels[i]):
                    if label:
                        f.write(f"  Sample {j+1}: {label}\n")
                f.write("\n")
        
        print(f"Combined matrix with epsilon labels saved to: {matrix_path}")
        print(f"Matrix labels saved to: {matrix_labels_path}")
        
    except Exception as e:
        print(f"Error creating combined matrix: {e}")
        # matplotlib方法失败时，回退到原始方法
        try:
            grid = make_grid(combined_images, nrow=max_samples * 2, padding=4, normalize=False)
            matrix_path = os.path.join(save_dir, "epsilon_matrix.png")
            save_image(grid, matrix_path)
            print(f"Fallback: Basic matrix saved to: {matrix_path}")
        except Exception as e2:
            print(f"Fallback also failed: {e2}")

def create_summary_table(results, save_dir):
    """创建汇总表格"""
    # 创建详细的结果表格
    table_data = []
    
    for eps, success_rate, avg_l2, samples_info in results:
        eps_255 = int(eps * 255)
        
        # 统计信息
        total_samples = len(samples_info)
        success_count = sum(1 for _, _, _, _, _, success, _, _ in samples_info if success)
        fail_count = total_samples - success_count
        
        # 计算置信度统计
        confidences = [conf for _, _, _, _, _, _, conf, _ in samples_info]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # 计算L2扰动统计
        l2_norms = [l2 for _, _, _, _, _, _, _, l2 in samples_info]
        avg_l2_norm = np.mean(l2_norms) if l2_norms else 0.0
        
        table_data.append({
            'Epsilon (/255)': eps_255,
            'Total Samples': total_samples,
            'Success Count': success_count,
            'Fail Count': fail_count,
            'Success Rate': f"{success_rate:.4f}",
            'Avg Confidence': f"{avg_confidence:.4f}",
            'Avg L2 Norm': f"{avg_l2_norm:.4f}"
        })
    
    # 保存为CSV
    df = pd.DataFrame(table_data)
    csv_path = os.path.join(save_dir, "lbfgs_summary_table.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"Summary table saved to: {csv_path}")
    
    return df

def plot_results(results, save_path="lbfgs_results.png"):
    """绘制L-BFGS攻击结果"""
    eps_vals = [r[0] * 255 for r in results]
    success_rates = [r[1] for r in results]
    avg_l2s = [r[2] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 成功率图
    ax1.plot(eps_vals, success_rates, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Epsilon (/255)')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('L-BFGS Attack Success Rate vs Epsilon')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 平均L2扰动图
    ax2.plot(eps_vals, avg_l2s, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Epsilon (/255)')
    ax2.set_ylabel('Average L2 Norm')
    ax2.set_title('Average L2 Perturbation vs Epsilon')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Results plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate L-BFGS attack on CIFAR-10")
    parser.add_argument("--model_path", type=str, default="exp/resnet18_cifar10_YYYYMMDD_HHMMSS/checkpoints/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=200, help="限制样本数量，设为-1 则用全量测试集")
    parser.add_argument("--eps255", nargs="+", type=int, default=[2,4,8,16], help="eps 的 /255 刻度")
    parser.add_argument("--c", type=float, default=1e-2, help="L2 惩罚系数 c")
    parser.add_argument("--max_iter", type=int, default=20, help="LBFGS 内部迭代次数（单样本）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录，默认使用模型目录")
    parser.add_argument("--samples_per_eps", type=int, default=10, help="每个 epsilon 保存的样本数量")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 加载模型
    model = build_resnet18(num_classes=10, pretrained=False).to(device)
    assert os.path.exists(args.model_path), f"Model not found: {args.model_path}"
    ckpt = torch.load(args.model_path, map_location=device)
    # 支持两种保存格式：train.py 的保存格式取 'model_state_dict'
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        best_acc = ckpt.get('best_acc', 'N/A')
    else:
        model.load_state_dict(ckpt)
        best_acc = 'N/A'
    model.eval()
    print("Loaded model:", args.model_path)
    print("Best training accuracy:", best_acc)

    # 设置输出目录
    if args.output_dir is None:
        # 从模型路径推断实验目录
        model_dir = os.path.dirname(os.path.dirname(args.model_path))
        args.output_dir = model_dir
    
    # 创建L-BFGS结果目录
    lbfgs_dir = os.path.join(args.output_dir, "lbfgs_attack")
    
    # 删除旧的结果目录
    if os.path.exists(lbfgs_dir):
        print(f"Removing old results directory: {lbfgs_dir}")
        shutil.rmtree(lbfgs_dir)
    
    os.makedirs(lbfgs_dir, exist_ok=True)
    print(f"Results will be saved to: {lbfgs_dir}")

    # 测试集：在像素空间做攻击，所以 dataloader 不做 Normalize（仅 ToTensor）
    transform = transforms.ToTensor()
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    eps_list = [e / 255.0 for e in args.eps255]
    results = []
    detailed_results = []  # 详细结果记录
    all_epsilon_samples = {}  # 存储所有epsilon的样本

    # 获取测试集大小
    if args.max_samples > 0:
        total_test_samples = min(args.max_samples, len(testset))
    else:
        total_test_samples = len(testset)
    
    print(f"Testing on {total_test_samples} samples")

    for eps in eps_list:
        success = 0
        total = 0
        l2_norms = []
        epsilon_samples = []  # 当前epsilon的样本
        print(f"\nEvaluating epsilon={eps*255:.0f}/255 ...")

        # 用 tqdm 遍历 test_loader（batch_size=1）
        for i, (x, y) in enumerate(tqdm(test_loader)):
            if args.max_samples > 0 and total >= args.max_samples:
                break

            x = x.to(device)
            y = y.to(device)
            # 随机选择一个目标类，且不同于真实标签
            target_class = torch.randint(0, 10, (1,), device=device)
            if target_class.item() == y.item():
                # 如果抽到相同类，则选择另一个
                target_class = (target_class + 1) % 10
                target_class = torch.tensor([int(target_class)], device=device)

            # 生成对抗样本（LBFGS + epsilon 约束）
            x_adv = lbfgs_attack_with_epsilon(model, x, target_class, epsilon=eps, c=args.c, max_iter=args.max_iter, device=device)

            with torch.no_grad():
                out_adv = model(normalize_cifar10(x_adv, device))
                pred_adv = out_adv.argmax(dim=1)
                confidence = torch.softmax(out_adv, dim=1).max().item()
                success_flag = pred_adv.item() == int(target_class.view(-1).item())
                if success_flag:
                    success += 1
                # 计算 L2 扰动
                l2_norm = torch.norm((x_adv - x).view(-1), p=2).item()
                l2_norms.append(l2_norm)

                # 记录详细结果
                detailed_results.append({
                    'sample_id': total,
                    'epsilon': eps * 255,
                    'true_label': y.item(),
                    'target_label': target_class.item(),
                    'predicted_label': pred_adv.item(),
                    'success': success_flag,
                    'l2_norm': l2_norm,
                    'confidence': confidence
                })

                # 保存样本用于可视化
                if len(epsilon_samples) < args.samples_per_eps:
                    epsilon_samples.append((
                        x.cpu(), x_adv.cpu(), y.item(), 
                        target_class.item(), pred_adv.item(), success_flag,
                        confidence, l2_norm
                    ))

            total += 1

        success_rate = success / total if total > 0 else 0.0
        avg_l2 = float(np.mean(l2_norms)) if len(l2_norms) > 0 else 0.0
        results.append((eps, success_rate, avg_l2, epsilon_samples))
        
        # 保存当前epsilon的样本对比图
        save_epsilon_samples(epsilon_samples, eps, lbfgs_dir)
        
        # 存储所有epsilon的样本
        all_epsilon_samples[eps] = epsilon_samples
        
        print(f"Eps={eps*255:.0f}/255, SuccessRate={success_rate:.4f}, AvgL2={avg_l2:.4f}")

    # 多个epsilon时，创建整合的大矩阵
    if len(eps_list) > 1:
        create_combined_matrix(all_epsilon_samples, lbfgs_dir)

    # 保存结果
    print("\n" + "="*50)
    print("L-BFGS Attack Results Summary:")
    print("="*50)
    
    # 保存详细结果到CSV
    df = pd.DataFrame(detailed_results)
    csv_path = os.path.join(lbfgs_dir, "lbfgs_detailed_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Detailed results saved to: {csv_path}")

    # 创建汇总表格
    summary_df = create_summary_table(results, lbfgs_dir)

    # 保存实验配置
    config = {
        'model_path': args.model_path,
        'best_training_accuracy': best_acc,
        'max_samples': args.max_samples,
        'epsilons': args.eps255,
        'c_penalty': args.c,
        'max_iterations': args.max_iter,
        'samples_per_epsilon': args.samples_per_eps,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    }
    config_path = os.path.join(lbfgs_dir, "lbfgs_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")

    # 绘制结果图
    plot_path = os.path.join(lbfgs_dir, "lbfgs_results.png")
    plot_results(results, plot_path)

    print(f"\nAll results saved to: {lbfgs_dir}")
    print("Files created:")
    print(f"  - {os.path.basename(csv_path)}: Detailed results")
    print(f"  - lbfgs_summary_table.csv: Summary table")
    print(f"  - {os.path.basename(config_path)}: Experiment configuration")
    print(f"  - {os.path.basename(plot_path)}: Results visualization")
    if len(eps_list) > 1:
        print(f"  - epsilon_matrix.png: Combined matrix for all epsilons")
        print(f"  - epsilon_matrix_labels.txt: Matrix labels")
    print(f"  - epsilon_X_samples.png: Sample comparisons for each epsilon")
    print(f"  - epsilon_X_labels.txt: Label information for each epsilon")

if __name__ == "__main__":
    main()

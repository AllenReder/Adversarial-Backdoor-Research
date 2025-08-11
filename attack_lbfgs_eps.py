# attack_lbfgs_eps.py
import torch
import torch.nn.functional as F
from torch.optim import LBFGS

# CIFAR-10 mean/std
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def normalize_cifar10(x, device):
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1,3,1,1)
    std  = torch.tensor(CIFAR10_STD, device=device).view(1,3,1,1)
    return (x - mean) / std

def _ensure_target_tensor(y_target, device):
    if not torch.is_tensor(y_target):
        y = torch.tensor([int(y_target)], dtype=torch.long, device=device)
    else:
        # 如果是 0-d scalar
        if y_target.dim() == 0:
            y = y_target.view(1).long().to(device)
        else:
            # 保证是一维 long tensor
            y = y_target.view(-1).long().to(device)
    return y

def lbfgs_attack_with_epsilon(model, x_orig, y_target, epsilon, c=1e-2, max_iter=100, device='cuda'):
    """
    使用 L-BFGS 攻击，限制扰动大小不超过 epsilon
    
    Args:
        model: 目标模型
        x_orig: 原始输入图像
        y_target: 目标类，int 或 tensor 类型
        epsilon: 最大扰动大小
        c: 正则化参数
        max_iter: 最大迭代次数
        device: 设备
    
    Returns:
        x_adv: 对抗样本
        success: 是否成功
    """
    model.eval()
    x_orig = x_orig.clone().detach().to(device)
    # 规范 target
    y_t = _ensure_target_tensor(y_target, device)  # shape (1,)

    # delta 为待优化变量
    delta = torch.zeros_like(x_orig, requires_grad=True, device=device)

    # 使用 PyTorch 的 LBFGS 优化器
    optimizer = LBFGS([delta], lr=0.5, max_iter=max_iter, tolerance_grad=1e-5, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        # 将 delta 限制到 L_inf 范围内（每次计算时都 clamp）
        delta_clamped = torch.clamp(delta, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + delta_clamped, 0.0, 1.0)
        x_adv_norm = normalize_cifar10(x_adv, device)
        outputs = model(x_adv_norm)  # shape (1, num_classes)
        # y_t 已经是 shape (1,), long tensor
        loss_ce = F.cross_entropy(outputs, y_t)
        # L2 距离项（论文中是 L2）
        loss_dist = torch.norm(delta_clamped.view(-1), p=2)
        loss = c * loss_dist + loss_ce
        loss.backward()
        return loss

    # 运行 LBFGS step（可能多次调用 closure）
    try:
        optimizer.step(closure)
    except Exception as e:
        print("LBFGS failed:", e)

    with torch.no_grad():
        delta_final = torch.clamp(delta, -epsilon, epsilon)
        x_adv_final = torch.clamp(x_orig + delta_final, 0.0, 1.0).detach()
    return x_adv_final

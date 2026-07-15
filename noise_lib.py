import abc
import torch
import torch.nn as nn
import numpy as np


def get_noise(config):
    """根据配置创建噪声日程对象。

    支持两种噪声日程：
    - geometric：几何插值日程，通常与 uniform graph 配合；
    - loglinear：对数线性日程，通常与 absorbing graph 配合。

    返回的对象是 PyTorch Module，因此可以像函数一样调用：
    ``sigma, dsigma = noise(t)``。
    """
    if config.noise.type == "geometric":
        return GeometricNoise(config.noise.sigma_min, config.noise.sigma_max)
    elif config.noise.type == "loglinear":
        return LogLinearNoise()
    else:
        raise ValueError(f"{config.noise.type} is not a valid noise")


class Noise(abc.ABC, nn.Module):
    """所有连续时间噪声日程的抽象基类。

    扩散时间 t 的取值范围通常为 [0, 1]：
    - t 接近 0：数据几乎保持干净；
    - t 接近 1：数据被强烈污染，接近极限分布。

    子类必须实现累计噪声 ``total_noise`` 和瞬时噪声变化率
    ``rate_noise``。
    """

    def forward(self, t):
        """同时返回累计噪声 sigma(t) 和变化率 d sigma(t) / dt。

        代码其他位置通常写作：

        ``sigma, dsigma = noise(t)``

        其中 dsigma 是对时间的导数，不是相邻两个离散时间点的简单差值。
        """
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def rate_noise(self, t):
        """返回瞬时噪声率 g(t)，也就是 d sigma(t) / dt。表示时间 t这一刻，噪声增加得有多快。

        在连续时间 Score Entropy 目标中，它用于给不同时间点的损失加权。
        """
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        """返回从起点累计到时间 t 的总噪声 sigma(t)。

        数学上可以理解为对瞬时噪声率 g(t) 的积分。累计噪声决定前向
        转移分布 q(x_t | x_0) 中数据被污染的程度。
        """
        pass


class GeometricNoise(Noise, nn.Module):
    """在 sigma_min 与 sigma_max 之间做几何插值的噪声日程。

    累计噪声定义为：

        sigma(t) = sigma_min ** (1 - t) * sigma_max ** t

    它在对数空间中随时间线性变化。该日程常用于 uniform 离散扩散图。
    """

    def __init__(self, sigma_min=1e-3, sigma_max=1, learnable=False):
        super().__init__()
        # 保存起始和结束噪声。乘以 1.0 用来确保创建浮点 Tensor。
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])
        if learnable:
            # learnable=True 时，噪声上下界也会作为模型参数参与训练。
            self.sigmas = nn.Parameter(self.sigmas)
        # 空参数用于保证噪声模块始终具有参数，兼容原项目的 DDP/优化器代码。
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        """计算几何累计噪声对时间 t 的导数。"""
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (self.sigmas[1].log() - self.sigmas[0].log())

    def total_noise(self, t):
        """计算时间 t 对应的几何累计噪声 sigma(t)。"""
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t


class LogLinearNoise(Noise, nn.Module):
    """用于吸收态离散扩散的对数线性噪声日程。

    累计噪声为：

        sigma(t) = -log(1 - (1 - eps) * t)

    对于 absorbing graph，一个 token 在时间 t 被替换成吸收态 mask 的概率为：

        p(mask) = 1 - exp(-sigma(t)) = (1 - eps) * t

    因此虽然累计噪声 sigma(t) 是对数形式，实际 mask 概率却几乎随时间 t
    线性增长：t=0.5 时大约一半 token 被 mask，t 接近 1 时几乎全部被 mask。

    eps 防止 t=1 时出现 log(0) 和无穷大，使扩散终点仍保留极小概率的
    原始 token，从而提高数值稳定性。
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        # 默认 eps=0.001，所以终点的最大 mask 概率约为 99.9%。
        self.eps = eps
        # 与 GeometricNoise 相同，这是用于兼容训练框架的占位参数。
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        """计算 d sigma(t) / dt，作为连续时间损失的权重。"""
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        """计算累计噪声 sigma(t)。

        使用 torch.log1p 而不是直接计算 log(1+x)，可以在 t 很小时获得
        更好的浮点数精度和数值稳定性。
        """
        return -torch.log1p(-(1 - self.eps) * t)


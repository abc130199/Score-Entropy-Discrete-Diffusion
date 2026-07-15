import abc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd


from catsample import sample_categorical


def get_graph(config, device):
    """根据配置创建离散扩散图。

    图决定 token 在前向扩散中可以向哪些状态跳转：

    - ``uniform``：任意 token 都能均匀地跳转到其他 token；
    - ``absorb``：普通 token 只能跳转到最后一个吸收状态（通常是 [MASK]）。

    ``device`` 当前没有在函数内部使用，保留它是为了兼容项目原有接口。
    """
    if config.graph.type == "uniform":
        return Uniform(config.tokens)
    elif config.graph.type == "absorb":
        return Absorbing(config.tokens)
    else:
        raise ValueError(f"Graph {config.graph.type} not valid")


def unsqueeze_as(x, y, back=True):
    """给 ``x`` 补长度为 1 的维度，使它能够与 ``y`` 广播运算。

    例如 ``x.shape == [B]``、``y.shape == [B, L, D]`` 时：

    - ``back=True`` 得到 ``[B, 1, 1]``；
    - ``back=False`` 得到 ``[1, 1, B]``。

    本文件主要用它把每个样本一个的噪声强度 sigma 扩展到序列/词表维度。
    """
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class Graph(abc.ABC):
    """离散连续时间马尔可夫链（CTMC）扩散图的抽象基类。

    图使用速率矩阵 Q 描述 token 的瞬时跳转规律。累计噪声为 sigma 时，
    相应的有限时间转移矩阵是 ``exp(sigma * Q)``。

    本项目采用“列代表起始状态”的约定：``rate(i)`` 返回 Q 的第 i 列，
    返回张量的最后一个维度枚举所有目标 token。合法速率向量必须满足：

    - 非对角元素非负，表示跳到其他状态的速率；
    - 对角元素非正，是所有离开速率之和的负数；
    - 整个速率向量之和为 0，以保证概率守恒。
    """

    @property
    def dim(self):
        """图中的状态总数，即包含特殊状态后的词表大小。"""
        pass

    @property
    def absorb(self):
        """
            Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask).
        """
        pass

    @abc.abstractmethod
    def rate(self, i):
        """
        Computes the i-th column of the rate matrix Q, where i is [B_1, ..., B_n].

        This is intended to compute the "forward" rate of p(X_t | X_0 = i).
        计算速率矩阵 Q 的第 i 列，用于前向扩散。

        ``i`` 可以具有 ``[B, L]`` 等任意批次形状；返回值形状为
        ``[*i.shape, dim]``，末维表示从状态 i 跳到每个目标状态的速率。
        """
        pass

    @abc.abstractmethod
    def transp_rate(self, i):
        """
        Computes the i-th row of the rate matrix Q.

        Can be used to compute the reverse rate.
        计算速率矩阵 Q 的第 i 行，主要用于构造反向扩散速率。

        对固定的当前状态 i，它给出所有候选状态和 i 之间的前向速率；
        再乘以模型预测的概率比，即可通过时间反演公式得到反向速率。
        """
        pass

    @abc.abstractmethod
    def transition(self, i, sigma):
        """
        Computes the i-th column of the transition matrix e^{sigma Q}.
        计算转移矩阵 ``exp(sigma * Q)`` 的第 i 列。

        返回从干净状态 i 出发，累计加入 sigma 强度的噪声后，落到每个
        状态的概率，即前向分布 ``q(x_sigma | x_0=i)``。
        """
        pass

    def sample_transition(self, i, sigma):
        """
        Samples the transition vector.
        按照前向转移概率采样加噪后的状态。

        默认实现先构造完整词表概率向量，再做 categorical 采样。子类可以
        覆盖该方法，使用不创建 ``[..., vocab]`` 张量的低显存直接采样公式。
        """
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")

    def reverse_rate(self, i, score):
        """
        Constructs the reverse rate. Which is score * transp_rate
        根据模型 score 和转置前向速率构造反向扩散速率。

        采样阶段传入的 ``score[..., j]`` 已经执行过 ``exp()``，近似：

        ``p_sigma(j) / p_sigma(i)``。

        对 ``j != i``，反向 CTMC 跳转速率由“转置前向速率 × 概率比”得到。
        随后把 i 位置的对角元素设为所有非对角速率之和的负数，使该速率
        向量之和严格为 0，满足概率守恒。
        """
        # 逐候选 token 计算：反向速率 = 转置前向速率 × 概率比。
        normalized_rate = self.transp_rate(i) * score

        # 暂时清零对角位置，只留下跳向其他状态的非对角速率。
        normalized_rate.scatter_(
            -1, i[..., None], torch.zeros_like(normalized_rate)
        )

        # 对角速率 = -所有非对角速率之和，故末维所有元素之和为 0。
        normalized_rate.scatter_(
            -1,
            i[..., None],
            -normalized_rate.sum(dim=-1, keepdim=True),
        )
        return normalized_rate

    def sample_rate(self, i, rate):
        """

        根据一个很小的离散时间步对应的速率采样下一状态。

        调用者通常已经把瞬时速率乘以 ``dt * d_sigma/dt``。当前状态的
        one-hot 向量加上这一步的速率修正，构成一阶近似转移概率。
        """
        return sample_categorical(
            F.one_hot(i, num_classes=self.dim).to(rate) + rate
        )

    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """
        Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with
        e^{-{dsigma} E} score
        把同一噪声时刻的 score 转为跨噪声时刻的 staggered score。

        输入近似 ``p_sigma(z) / p_sigma(x)``，输出近似：

        ``p_{sigma-dsigma}(z) / p_sigma(x)``。

        analytic predictor 将其与长度为 dsigma 的前向转移因子相乘，得到
        从当前高噪声状态到下一个低噪声状态的反向条件概率。

        注意：这里的 ``dsigma`` 是两次采样之间的有限差 ``Delta sigma``，
        不是训练时 ``noise(t)`` 返回的导数 ``d sigma(t) / dt``。不同图的
        矩阵指数闭式解不同，所以具体公式由子类实现。
        """
        pass

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """
        Sample the limiting distribution. Returns the probability vector as well.
        从前向扩散的极限分布中采样反向生成的初始状态。
        """
        pass

    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """
        Computes the score entropy function (with requisite constant normalization)
        计算每个 token 的 Score Entropy 训练损失。

        参数：

        - ``score``：模型直接输出的 log-score（还没有执行 ``exp``）；
        - ``sigma``：累计噪声强度；
        - ``x``：前向加噪状态 ``x_sigma``；
        - ``x0``：原始干净状态。

        返回值保留 token 维度。外部损失函数还会乘以 ``d sigma/dt``，
        然后对序列位置和 batch 做求和或平均。
        """
        pass


class Uniform(Graph):
    """
    Everything goes to everything else. Normalized down by dimension to avoid blowup.
    均匀扩散图：任意状态均可跳到任意状态。

    非对角速率都是 ``1/dim``，使用词表大小归一化，避免词表增大时总跳转
    速率随之爆炸。该图不存在单独的 [MASK] 吸收状态。
    """

    def __init__(self, dim):
        """``dim`` 是图中的状态总数。"""
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def absorb(self):

        return False

    def rate(self, i):# 构造Q扩散矩阵
        """
        构造Q_uniform对角线的数据
        返回 Uniform 速率矩阵 Q 的第 i 列。
        从 i 跳向每个其他状态的速率都是 ``1/dim``；i 位置的对角元素是
        ``-(dim-1)/dim``，所以整列之和为 0。
        """
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(
            -1, i[..., None], -(self.dim - 1) / self.dim
        )
        return edge

    def transp_rate(self, i):
        """Uniform 的 Q 是对称矩阵，所以第 i 行与第 i 列相同。"""
        return self.rate(i)

    def transition(self, i, sigma):# exp(sigma * Q)经过sigma时间后的真是概率，(B, L, V)大张量矩阵
        """
        计算 ``p_(t) = exp(sigma Q)`` 给出的完整前向转移概率矩阵。

        sigma 越大，保留原 token 的概率越低，分布逐渐趋近词表均匀分布。如果sigma = 0，表示不改变句子
        sigma = 100, 表示已经完全扩散
        """
        trans = (
            torch.ones(*i.shape, self.dim, device=i.device) # 1-exp(-sigma)
            * (1 - (-sigma[..., None]).exp())
            / self.dim
        )
        trans = trans.scatter(
            -1, i[..., None], torch.zeros_like(trans) ) #先将置为0
        trans = trans.scatter(
            -1,
            i[..., None],
            1 - trans.sum(dim=-1, keepdim=True),
        )
        return trans

    def transp_transition(self, i, sigma):
        """Uniform 转移矩阵对称，因此转置转移与普通转移相同。"""
        return self.transition(i, sigma)

    def sample_transition(self, i, sigma): # 采样一个(B, L  )
        """直接采样 Uniform 前向过程，避免构造完整词表概率张量。

        以 ``1-exp(-sigma)`` 的概率触发均匀重采样。重采样仍可能抽到原
        token，这与 ``transition`` 中的解析概率完全一致。
        """
        move_chance = 1 - (-sigma).exp() # 扩散发生概率
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(
            move_indices, torch.randint_like(i, self.dim), i
        )
        return i_pert

    def staggered_score(self, score, dsigma):# 把当前时间 t 的 score 转换成时间 t−Δt 的 score。
        """score(x_t) --->>经过一个反向扩散步 --->>score(x_{t-\Delta t})
        score.shape = (B, L, V)
        """
        dim = score.shape[-1]
        epow = (-dsigma).exp()[..., None]
        return (
            ((epow - 1) / (dim * epow))
            * score.sum(dim=-1, keepdim=True)
            + score / epow
        )

    def sample_limit(self, *batch_dims):
        """从 Uniform 图的极限分布——词表均匀分布中采样。"""
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0):
        """计算 Uniform 图的解析 Score Entropy 损失。"""
        # 计算 exp(sigma)-1；小 sigma 时用 expm1 可减少浮点相消误差。
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1,
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)

        # 负项：模型 log-score 的线性部分。
        neg_term = score.mean(dim=-1) - torch.gather(
            score, -1, x[..., None]
        ).squeeze(-1) / self.dim

        # x==x0 表示加噪采样没有改变 token；否则额外加入原 token x0
        # 对应的模型输出。
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1)
            / esigm1
            + neg_term,
        )

        # 与模型参数无关、但保证目标正确归一化的常数项。
        const = torch.where(
            x == x0,
            (self.dim - 1)
            / self.dim
            * ratio
            * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim,
        )

        # 正项使用真正的概率比，因此要对模型的 log-score 取指数。
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(
            sexp, -1, x[..., None]
        ).squeeze(-1) / self.dim
        return pos_term - neg_term + const


class Absorbing(Graph):
    """吸收状态扩散图，也是 SEDD 文本模型通常使用的图。

    普通 token 在前向扩散中只能变为 [MASK]，进入 [MASK] 后不会再离开。
    所以最后一个状态 ``dim-1`` 是吸收状态。反向生成从全 [MASK] 开始，
    模型逐步预测并恢复普通 token。
    """

    def __init__(self, dim):
        """``dim`` 是普通 token 数；内部会额外增加一个 [MASK] 状态。"""
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        """普通 token 数量加上一个 [MASK] 状态。"""
        return self._dim + 1

    @property
    def absorb(self):
        """该图的最后一个状态是吸收状态。"""
        return True

    def rate(self, i):
        """
        说明概率分布如何随时间变化。对应论文 公式2
        即从当前 token i 跳到每个候选 token y 的瞬时速率。
        返回吸收图速率矩阵 Q 的第 i 列。

        普通状态 i 的速率在 i 处为 -1、在 [MASK] 处为 +1，表示它只能以
        单位速率跳到 [MASK]。如果 i 已是 [MASK]，两项抵消为全 0，表示
        吸收后不再发生前向跳转。
        """
        return F.one_hot(
            (self.dim - 1) * torch.ones_like(i),
            num_classes=self.dim,
        ) - F.one_hot(i, num_classes=self.dim)

    def transp_rate(self, i):
        """
        对应的论文 公式3
        返回 Q 的第 i 行，用来构造从 [MASK] 恢复 token 的反向速率。
        当前状态是普通 token 时，只有对角位置为 -1；当前状态为 [MASK]
        时，所有普通 token 位置为 +1，而 [MASK] 位置为 0。``reverse_rate``
        随后使用模型 score 对这些候选方向加权。
        """
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(self, i, sigma):
        """完整词表概率向量版本未实现。

        Absorbing 图的训练前向加噪实际调用下面的 ``sample_transition``，
        不创建巨大的 ``[B, L, vocab]`` 张量，从而明显节省显存。
        """
        pass

    def transp_transition(self, i, sigma):
        """计算转移矩阵 ``exp(sigma Q)`` 的第 i 行。

        它不是一个单独归一化的概率分布，而是 analytic predictor 根据
        贝叶斯公式构造反向条件概率时所需的“转置前向转移因子”。
        transformer输出的是score，但是这个输出的是当前x_t由x_0变化过来的概率。相当于扩散规律，表达的是原来的token编程MASK的概率大不大(个人理解)
        """
        # 给样本级 sigma 补充维度
        sigma = unsqueeze_as(sigma, i[..., None])

        # exp(-sigma) 是经过 sigma 噪声后仍保留原状态的概率因子。
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)

        # 当当前状态 i 是 [MASK] 时，补上普通 token 前向变为 [MASK] 的
        # 因子 1-exp(-sigma)；末维枚举所有可能的较早状态。
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0,
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        """
        直接采样吸收前向过程
        Absorb Diffusion 的前向扩散采样器。它以 1−exp(-sigma)的概率把 token 变成 [MASK]，否则保持原样，原本已经是 [MASK] 的位置采样后依然是 [MASK]
        从而产生训练时的带噪文本 xt
        """
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert

    def staggered_score(self, score, dsigma):
        """计算吸收图下的 staggered score 闭式变换。
        当前时刻的score --->>下一时刻的score
        s_t --->> s_(t-delta_t)

        ``score`` 是采样阶段已经取指数的正概率比。该公式等价于使用吸收图
        对应的 ``exp(-Delta sigma * Q)``
        变换 score：普通 token 项按``exp(Delta sigma)`` 缩放，[MASK] 项还需加入所有状态的修正量。
        """
        # 后面使用原地乘法和加法，先复制，避免修改模型原始输出。
        score = score.clone()

        # 计算 [MASK] 分量需要加入的解析修正量。!!!!!!!
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)

        # 将所有分量先乘以 exp(Delta sigma)。
        score *= dsigma.exp()[:, None]

        # 末维最后一个元素是 [MASK]，为它加入额外修正量。!!!!!!!!!
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        """吸收图的极限分布是确定性的：所有位置均为 [MASK]。"""
        return (self.dim - 1) * torch.ones(
            *batch_dims, dtype=torch.int64
        )

    def score_entropy(self, score, sigma, x, x0):
        """计算吸收图的解析 Score Entropy 损失。
        x0:原始干净文本
        x:经过前向扩散后的文本，也就是 x_t
        sigma:累计噪声强度
        score:模型输出的 log-score
        self.dim-1:[MASK] 的 token id

        只有已经被前向扩散替换为 [MASK] 的位置提供去噪监督；未被遮盖的
        token 损失为 0。输入 ``score`` 是模型直接输出的 log-score。
        """
        # 标记当前处于 [MASK]、找出被 MASK 的位置
        rel_ind = x == self.dim - 1

        # 计算 exp(sigma)-1；小 sigma 时使用 expm1 保持数值精度。
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1,
        )

        # 前向吸收过程推导出的真实条件概率比。
        ratio = 1 / esigm1.expand_as(x)[rel_ind]

        # 保存被 MASK 位置在扩散前对应的正确 token。
        other_ind = x0[rel_ind]

        # 负项：取正确原始 token 的 log-score，并按真实概率比加权。
        neg_term = ratio * torch.gather(
            score[rel_ind], -1, other_ind[..., None]
        ).squeeze(-1)

        # 正项：把普通 token 的 log-score 转为概率比后求和。最后一项是
        # [MASK] 自身，因此从候选普通 token 中排除。
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)

        # 与模型参数无关的归一化常数项。
        const = ratio * (ratio.log() - 1)

        # 全部位置先设为零，只在真正被 MASK 的位置写入损失。
        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy

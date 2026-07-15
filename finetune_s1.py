"""使用 s1K 推理数据对预训练 SEDD 进行监督微调。

这个文件与原项目的 ``train.py`` 有本质区别：

1. ``train.py`` 把文本当作连续语料，执行无条件扩散语言模型训练；
2. 本文件保留每一道题的 Question / Reasoning / Answer 边界；
3. Question token 作为条件，前向扩散时保持不变；
4. 只有 Reasoning 和 Answer token 会被加噪，并参与 Score Entropy 损失；
5. 推理时固定 Question，通过反向扩散生成后续的推理和答案。

因此，这里的训练更接近传统语言模型中的“监督微调（SFT）”，只是训练目标
从自回归交叉熵换成了离散扩散模型的条件 Score Entropy。
"""

import argparse
import json
import os
import random
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast

from load_model import load_model_hf
from sampling import get_pc_sampler




# -----------------------------------------------------------------------------
# 默认微调参数
# -----------------------------------------------------------------------------
# 直接运行 ``python finetune_s1.py`` 时会使用下面的参数。
# 命令行参数仍然可以临时覆盖这些值，例如：
# ``python finetune_s1.py --steps 5000 --batch-size 4``。
DEFAULT_MODEL = "louaaron/sedd-small"
DEFAULT_DATASET = "s1k-1.1"
DEFAULT_STEPS = 10_000

DEFAULT_BATCH_SIZE = 8
# 累计多个 micro-batch 的梯度后再更新一次参数。
# batch size = 8 × 4 = 32。
DEFAULT_GRAD_ACCUM = 4
# 微调学习率应明显小于从头训练时使用的 3e-4，避免破坏预训练能力。
DEFAULT_LEARNING_RATE = 1e-5

# chioces dataset
DATASETS = {
    "s1k": "simplescaling/s1K",
    "s1k-1.1": "simplescaling/s1K-1.1",
}


def parse_args():
    """定义命令行参数，并返回解析后的配置对象。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", choices=DATASETS, default=DEFAULT_DATASET)
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--output-dir", default="exp_local/s1k-1.1-sft")
    parser.add_argument("--max-length", type=int, default=1024) #词条最大长度
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="GPU micro-batch size, not effective batch size")
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=0.0) #权重衰减
    parser.add_argument("--warmup-steps", type=int, default=250) #逐渐提高学习率
    parser.add_argument("--grad-clip", type=float, default=1.0) #梯度范数超过 1 时进行裁剪，降低梯度爆炸风险。
    parser.add_argument("--log-every", type=int, default=10) # every 10 steps print loss
    parser.add_argument("--eval-every", type=int, default=250) # every 250 steps print eval loss
    parser.add_argument("--save-every", type=int, default=250) # save loss
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint.pt (or a directory containing it)")
    parser.add_argument("--sample-count", type=int, default=10,
                        help="choice 10 samples；set 0 off")
    parser.add_argument("--sample-steps", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def seed_everything(seed):
    """固定 Python、NumPy 和 PyTorch 随机种子，使数据划分和训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name):
    """根据参数自动选择 CUDA 或 CPU，并检查用户指定的设备是否可用。"""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(name)


def build_splits(dataset_name, cache_dir, seed):
    """下载/读取 s1K，并固定划分为 80% 训练、10% 验证、10% 测试。

    s1K 官方只提供一个 train split，所以这里必须自行划分。第一次划分拿出
    20% 数据，第二次把这 20% 平分为验证集和测试集。相同 seed 会得到完全
    相同的样本编号，编号之后还会写入 splits.json。
    """
    dataset = load_dataset(DATASETS[dataset_name], split="train", cache_dir=cache_dir)
    # 给每条原始数据增加稳定编号，便于复现实验并追溯生成结果来自哪道题。
    dataset = dataset.map(lambda _, index: {"example_id": index}, with_indices=True)
    # 第一次划分：800 条训练数据 + 200 条临时数据。
    train_and_rest = dataset.train_test_split(test_size=0.2, seed=seed)
    # 第二次划分：把 200 条临时数据平分为 100 条验证和 100 条测试数据。
    valid_and_test = train_and_rest["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict({
        "train": train_and_rest["train"],
        "validation": valid_and_test["train"],
        "test": valid_and_test["test"],
    })


def response_text(example, dataset_name):
    """把数据集字段整理成模型要学习生成的 Reasoning / Answer 文本。

    s1K-1.1 使用 DeepSeek/R1 生成的推理轨迹；旧版 s1K 使用 Gemini 轨迹。
    当完整回答字段为空时，退回使用数据集给出的简短标准答案 solution。
    """
    if dataset_name == "s1k-1.1":
        reasoning = example.get("deepseek_thinking_trajectory") or ""
        answer = example.get("deepseek_attempt") or example.get("solution") or ""
    else:
        trajectories = example.get("thinking_trajectories") or []
        reasoning = trajectories[0] if trajectories else ""
        answer = example.get("attempt") or example.get("solution") or ""
    return f"Reasoning:\n{reasoning}\n\nAnswer:\n{answer}"


class S1Collator:
    """把若干结构化 s1K 样本整理成一个可训练的定长 batch。

    输出包括：
    - input_ids：GPT-2 token id，形状为 [batch, max_length]；
    - loss_mask：Question/padding 为 0，Reasoning/Answer 为 1；
    - prompt_length：每条样本的prompt长度；
    - example_id/solution：用于测试结果追踪和答案评价。
    """

    def __init__(self, tokenizer, dataset_name, max_length, max_prompt_tokens):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.max_prompt_tokens = max_prompt_tokens
        self.eos = tokenizer.eos_token_id

    def __call__(self, examples):
        """由 DataLoader 自动调用，把样本列表转换为 Tensor batch。"""
        rows = []
        for example in examples:
            # Question 和 Response 标记共同组成条件 prompt。
            prompt = f"Question:\n{example['question']}\n\nResponse:\n"
            # 问题过长时截断，确保为推理和答案保留足够空间。
            prompt_ids = self.tokenizer.encode(
                prompt, add_special_tokens=False, truncation=True,
                max_length=self.max_prompt_tokens,
            )
            # 目标区域可用长度 = 总长度 - 实际问题长度。
            target_room = self.max_length - len(prompt_ids)
            if target_room < 2:
                raise ValueError("max_length must leave room for response tokens")
            # 对 Reasoning/Answer 编码；预留一个位置放 EOS 结束符。 get target token ID
            target_ids = self.tokenizer.encode(
                response_text(example, self.dataset_name), add_special_tokens=False,
                truncation=True, max_length=target_room - 1,
            )
            target_ids += [self.eos]
            ids = prompt_ids + target_ids
            # 0 表示不计算监督损失，1 表示参与扩散和 Score Entropy 损失。
            loss_mask = [0] * len(prompt_ids) + [1] * len(target_ids)
            # SEDD 使用固定长度序列。这里用 EOS 补齐到1024，但 padding 的 mask 为 0，
            # 因此这些补齐 token 不会影响训练目标。
            padding = self.max_length - len(ids)
            ids += [self.eos] * padding
            loss_mask += [0] * padding
            rows.append((ids, loss_mask, len(prompt_ids)))

        return {
            "input_ids": torch.tensor([row[0] for row in rows], dtype=torch.long),
            "loss_mask": torch.tensor([row[1] for row in rows], dtype=torch.float32),
            "prompt_length": torch.tensor([row[2] for row in rows], dtype=torch.long),
            "example_id": torch.tensor([example["example_id"] for example in examples]),
            "solution": [str(example.get("solution") or "") for example in examples],
        }


def score_entropy_sft_loss(model, graph, noise, input_ids, loss_mask, eps=1e-3):
    """计算条件监督微调版本的 Score Entropy 损失。

    训练过程：
    1. 为 batch 中每条样本随机采样扩散时间 t；
    2. 根据噪声日程得到累计噪声 sigma 和变化率 dsigma；
    3. 先对整条序列采样带噪状态；
    4. 用 loss_mask 把 Question 区域恢复成干净 token，仅污染目标区域；
    5. 模型预测离散 score（对数概率比）；
    6. 计算逐 token Score Entropy，并只保留目标区域；
    7. 按有效目标 token 数归一化，使长短样本权重更加稳定。
    """
    # t 接近 0 表示噪声很少，接近 1 表示大部分目标 token 已被吸收态 mask 替换。
    # 避免 noise(t)-1=0 input_ids=[batch_size, sequence_length]，意思是batch size的长度，每个token独立[0,1)概率
    t = (1 - eps) * torch.rand(input_ids.shape[0], device=input_ids.device) + eps
    # noise()返回total_noise(t),rate_noise(t)两个值
    sigma, dsigma = noise(t)
    # 根据前向转移分布 q(x_t | x_0) 采样带噪序列。
    noisy_all = graph.sample_transition(input_ids, sigma[:, None])
    # 条件区域使用原 token；只有 loss_mask=1 的 Reasoning/Answer 区域保留噪声。
    noisy_target = torch.where(loss_mask.bool(), noisy_all, input_ids)
    # 模型输出每个位置到整个词表的 log-score。
    log_score = model(noisy_target, sigma)
    # 原论文定义的逐 token Score Entropy 目标。
    token_loss = graph.score_entropy(log_score, sigma[:, None], noisy_target, input_ids)
    # dsigma 是连续时间目标中的权重；再乘 loss_mask 排除问题和 padding。
    weighted = dsigma[:, None] * token_loss * loss_mask
    return weighted.sum() / loss_mask.sum().clamp_min(1)


@torch.no_grad()
def evaluate(model, graph, noise, loader, device):
    """在验证集或测试集上计算平均条件 Score Entropy，不更新参数。"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        with autocast_context(device):
            loss = score_entropy_sft_loss(model, graph, noise, input_ids, loss_mask)
        total_loss += loss.item()
        total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


def autocast_context(device):
    """CUDA 上使用 BF16 自动混合精度，CPU 上不启用 autocast。"""
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def learning_rate(step, args):
    """线性 warmup：前 warmup_steps 步从 0 增长到目标学习率。"""
    if args.warmup_steps <= 0:
        return args.learning_rate
    return args.learning_rate * min(step / args.warmup_steps, 1.0)


def save_checkpoint(output_dir, model, optimizer, step, args, split_ids):
    """保存可继续训练的完整断点，包括模型、优化器、步数和数据划分。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "args": vars(args),
            "split_ids": split_ids,
        },
        output_dir / "checkpoint.pt",
    )


def load_checkpoint(path, model, optimizer, device):
    """恢复模型与优化器状态，并返回已经完成的 optimizer step 数。"""
    path = Path(path)
    if path.is_dir():
        path = path / "checkpoint.pt"
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


def normalize_answer(text):
    """简化答案文本，供基础 Exact Match 指标使用。"""
    text = text.lower().strip() # 转换为小写
    text = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", text) # 去掉多余空格
    return re.sub(r"[^a-z0-9.+\-/]", "", text) # 去掉部分标点


@torch.no_grad()

def generate_test_samples(model, graph, noise, dataset, collator, tokenizer,
                          device, count, steps, output_dir):
    """在测试问题上执行条件反向扩散，并保存生成结果。
    测试题 --->>保留 Question --->>回答区域初始化为 [MASK] --->>SEDD 逐步反向扩散 --->>生成 Reasoning 和 Answer --->>与标准答案比较 --->>保存到 test_generations.jsonl
    采样初始状态全部为吸收态 mask。每一个反向扩散步骤都会
    通过 project_prompt 强制恢复 Question token

    ****训练师忘了调用这个函数，结果训练的时候没有做对比
    """
    model.eval()
    results = []
    for example in dataset.select(range(min(count, len(dataset)))): # 把一道题转换成 token
        batch = collator([example])
        clean = batch["input_ids"].to(device)  # 干净的文本的 token，Question、Reasoning 和 Answer；
        prompt_length = int(batch["prompt_length"][0]) # Question 条件区域的长度。

        def project_prompt(x):
            # 条件投影（clamping）：任何采样步骤都不允许改变问题区域。
            x[:, :prompt_length] = clean[:, :prompt_length]
            return x

        # analytic predictor 使用解析形式把当前 score 转换为下一个时间步分布。

        sampler = get_pc_sampler(
            graph, noise, (1, collator.max_length), "analytic", steps,
            device=device, proj_fun=project_prompt,
        )
        """
           graph：吸收扩散图
           noise：噪声
           (1, max_length)：一次生成一道题
           "analytic"：使用 Tweedie 解析反向采样
           steps：反向扩散步数
           project_prompt：每一步都固定 Question
        """
        # sampler 最后还有一次去噪，再投影一次以确保返回的问题完全未被改变。


        generated = project_prompt(sampler(model))[0, prompt_length:] # [0]：取 batch 中第一条结果；[prompt_length:]：去掉 Question，只保留模型生成的回答区域。
        text = tokenizer.decode(generated, skip_special_tokens=True) # token转换成文字

        predicted = text.rsplit("Answer:", 1)[-1] # 提取答案并比较
        reference = str(example.get("solution") or "")

        results.append(
            {
                "example_id": int(example["example_id"]),
                "question": example["question"],
                "reference": reference,
                "generated": text,
                "normalized_exact_match": normalize_answer(predicted) == normalize_answer(reference),
            }
        )

    output_path = output_dir / "test_generations.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    accuracy = sum(row["normalized_exact_match"] for row in results) / max(len(results), 1)
    return accuracy, output_path


def main():
    """组织模型加载、数据处理、训练、验证、保存与最终测试。"""
    args = parse_args()
    if args.max_prompt_tokens >= args.max_length:
        raise ValueError("max-prompt-tokens must be smaller than max-length")
    seed_everything(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pretrained model {args.model!r} on {device} ...")
    # 加载预训练 SEDD。
    model, graph, noise = load_model_hf(args.model, device)
    model.train()

    noise.eval()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=args.cache_dir)
    splits = build_splits(args.dataset, args.cache_dir, args.seed)
    # 保存实际样本编号，而不是只保存划分比例，确保论文实验可复现。
    split_ids = {
        name: [int(value) for value in split["example_id"]]
        for name, split in splits.items()
    }
    with (output_dir / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump(split_ids, handle, indent=2)

    collator = S1Collator(
        tokenizer, args.dataset, args.max_length, args.max_prompt_tokens
    )
    generator = torch.Generator().manual_seed(args.seed)
    # DataLoader 的 shuffle 仅用于训练集；验证和测试必须保持固定顺序。
    train_loader = DataLoader(
        splits["train"], batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.num_workers, generator=generator,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        splits["validation"], batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        splits["test"], batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=args.num_workers,
    )

    # AdamW 只接收模型参数，noise 和 graph 没有加入 optimizer。
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    #  --resume checkpoint.pt 时恢复断点训练。
    step = load_checkpoint(args.resume, model, optimizer, device) if args.resume else 0
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    micro_step = 0
    train_iterator = iter(train_loader)

    print(
        f"Splits: train={len(splits['train'])}, validation={len(splits['validation'])}, "
        f"test={len(splits['test'])}; effective batch={args.batch_size * args.grad_accum}"
    )
    while step < args.steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        loss_mask = batch["loss_mask"].to(device, non_blocking=True)
        with autocast_context(device):
            loss = score_entropy_sft_loss(model, graph, noise, input_ids, loss_mask)

        (loss / args.grad_accum).backward() # 每个 micro-batch 计算梯度，但损失除以累计次数，保持梯度尺度一致。
        running_loss += loss.item()
        micro_step += 1


        if micro_step % args.grad_accum != 0: # 尚未收集够 grad_accum 个 micro-batch 时，不执行 optimizer.step()。
            continue

        step += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)# 梯度裁剪，防止梯度爆炸。
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step == 1:
            print(f"step={step} train_loss={running_loss / args.grad_accum:.6f} lr={lr:.3e}")
        running_loss = 0.0

        # 达到设定频率时，在完整验证集上评估。
        if step % args.eval_every == 0:
            valid_loss = evaluate(model, graph, noise, valid_loader, device)
            print(f"step={step} validation_score_entropy={valid_loss:.6f}")
        # 定期覆盖 checkpoint.pt，支持训练中断后继续。
        if step % args.save_every == 0:
            save_checkpoint(output_dir, model, optimizer, step, args, split_ids)


    save_checkpoint(output_dir, model, optimizer, step, args, split_ids)

    export_config = OmegaConf.to_container(model.config, resolve=True)
    model.save_pretrained(output_dir / "model", config=export_config)

    test_loss = evaluate(model, graph, noise, test_loader, device)
    metrics = {"step": step, "test_score_entropy": test_loss}
    print(f"test_score_entropy={test_loss:.6f}")

    # 默认从测试集选择 10 条执行条件生成；传入 --sample-count 0 可以关闭。
    if args.sample_count > 0:
        accuracy, path = generate_test_samples(
            model, graph, noise, splits["test"], collator, tokenizer, device,
            args.sample_count, args.sample_steps, output_dir,
        )
        metrics["sampled_normalized_exact_match"] = accuracy
        metrics["sample_count"] = min(args.sample_count, len(splits["test"]))
        print(f"sampled_normalized_exact_match={accuracy:.4f}; generations={path}")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()

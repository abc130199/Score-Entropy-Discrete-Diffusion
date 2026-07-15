"""第二阶段：使用奖励加权 Score Entropy 继续微调 SEDD。

读取第一阶段 ``reward_stage1_best_of_n.py`` 保存的候选回答。
对于同一问题的 N 个候选，根据奖励计算指数权重：

    w_i = exp((R_i - mean(R)) / temperature)

随后把每组权重归一化到平均值为 1，并使用这些权重缩放每条样本的
Score Entropy。
***训练 batch 默认以 50% 概率抽取原始数据、50% 概率抽取奖励候选数据。***
"""

import argparse
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import GPT2TokenizerFast

from finetune_s1 import (
    S1Collator,
    autocast_context,
    build_splits,
    choose_device,
    evaluate,
    generate_test_samples,
    learning_rate,
    load_checkpoint,
    response_text,
    save_checkpoint,
    seed_everything,
)
from load_model import load_model_hf


DEFAULT_MODEL = "exp_local/s1k-1.1-sft/model"
DEFAULT_CANDIDATES = "exp_local/reward-stage1/best_of_n.jsonl"
DEFAULT_OUTPUT = "exp_local/reward-stage2"


def parse_args():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--dataset", choices=("s1k", "s1k-1.1"), default="s1k-1.1")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-metrics",
                        default="exp_local/s1k-1.1-sft/metrics.json")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.25,
                        help="Temperature used by exponential reward weights")
    parser.add_argument("--min-best-reward", type=float, default=0.5,
                        help="Skip prompt groups whose best candidate is below this reward")
    parser.add_argument("--generated-fraction", type=float, default=0.5,
                        help="Expected fraction of generated rows sampled during training")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--sample-steps", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


class RecordDataset(Dataset):
    """保存已经统一字段格式的原始数据集行和奖励生成行。"""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class RewardWeightedCollator:
    """
    把文本记录编码成条件 SEDD 训练 batch，并保留样本奖励权重。

    """

    def __init__(self, tokenizer, max_length, max_prompt_tokens):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_tokens = max_prompt_tokens
        self.eos = tokenizer.eos_token_id

    def __call__(self, records):
        input_rows = []
        mask_rows = []
        weights = []
        sources = []

        for record in records:
            prompt = f"Question:\n{record['question']}\n\nResponse:\n"
            prompt_ids = self.tokenizer.encode(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_prompt_tokens,
            )
            target_room = self.max_length - len(prompt_ids)
            if target_room < 2:
                raise ValueError("max_length must leave room for response tokens")

            target_ids = self.tokenizer.encode(
                record["response"],
                add_special_tokens=False,
                truncation=True,
                max_length=target_room - 1,
            )
            target_ids.append(self.eos)
            input_ids = prompt_ids + target_ids
            loss_mask = [0] * len(prompt_ids) + [1] * len(target_ids)

            padding = self.max_length - len(input_ids)
            input_ids += [self.eos] * padding
            loss_mask += [0] * padding
            input_rows.append(input_ids)
            mask_rows.append(loss_mask)
            weights.append(float(record["sample_weight"]))
            sources.append(record["source"])

        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "loss_mask": torch.tensor(mask_rows, dtype=torch.float32),
            "sample_weight": torch.tensor(weights, dtype=torch.float32),
            "source": sources,
        }


def load_jsonl(path):
    """
    读取第一阶段生成的 JSONL
    """
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error
    return rows


def exponential_group_weights(rewards, temperature):
    """
    1 --->>计算数值稳定的组内指数奖励权重
    2 --->归一化至平均值为 1。
    """
    if temperature <= 0:
        raise ValueError("--temperature must be positive")
    mean_reward = sum(rewards) / len(rewards)
    logits = [max(min((reward - mean_reward) / temperature, 20.0), -20.0)
              for reward in rewards]
    raw_weights = [math.exp(logit) for logit in logits]
    mean_weight = sum(raw_weights) / len(raw_weights)
    return [weight / mean_weight for weight in raw_weights]


def build_training_records(train_split, dataset_name, candidate_groups,
                           temperature, min_best_reward):
    """
    1原始数据集的记录
    2带组内奖励权重的生成记录。
    """
    original_records = []
    for example in train_split:
        original_records.append({
            "example_id": int(example["example_id"]),
            "question": str(example["question"]),
            "response": response_text(example, dataset_name),
            "sample_weight": 1.0,
            "source": "original",
        })

    generated_records = []
    eligible_groups = 0
    for group in candidate_groups:
        candidates = [
            candidate for candidate in group.get("candidates", [])
            if str(candidate.get("generated") or "").strip()
        ]
        if not candidates:
            continue
        best_reward = max(float(candidate.get("reward", 0.0)) for candidate in candidates)
        if best_reward < min_best_reward:
            continue

        rewards = [float(candidate.get("reward", 0.0)) for candidate in candidates]
        sample_weights = exponential_group_weights(rewards, temperature)
        eligible_groups += 1
        for candidate, sample_weight in zip(candidates, sample_weights):
            generated_records.append({
                "example_id": int(group["example_id"]),
                "question": str(group["question"]),
                "response": str(candidate["generated"]),
                "reward": float(candidate.get("reward", 0.0)),
                "sample_weight": float(sample_weight),
                "source": "generated",
            })

    return original_records, generated_records, eligible_groups


def make_source_sampler(original_count, generated_count, generated_fraction, seed):
    """
    1每条原始记录共享 ``1-generated_fraction`` 的总概率质量，
    2每条生成记录共享 ``generated_fraction`` 的总概率质量，
    3因此默认期望得到 50/50。
    """
    if not 0 <= generated_fraction <= 1:
        raise ValueError("--generated-fraction must be between 0 and 1")
    if generated_count == 0 and generated_fraction > 0:
        raise ValueError("No eligible generated records; lower --min-best-reward or run stage 1")
    if original_count == 0 and generated_fraction < 1:
        raise ValueError("No original records are available")

    original_weight = (
        (1 - generated_fraction) / original_count if original_count else 0.0
    )
    generated_weight = (
        generated_fraction / generated_count if generated_count else 0.0
    )
    source_weights = (
        [original_weight] * original_count
        + [generated_weight] * generated_count
    )
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        source_weights,
        num_samples=original_count + generated_count,
        replacement=True,
        generator=generator,
    )


def reward_weighted_score_entropy_loss(
    model, graph, noise, input_ids, loss_mask, sample_weight, eps=1e-3
):

    t = (1 - eps) * torch.rand(input_ids.shape[0], device=input_ids.device) + eps
    sigma, dsigma = noise(t)
    noisy_all = graph.sample_transition(input_ids, sigma[:, None])
    noisy_target = torch.where(loss_mask.bool(), noisy_all, input_ids)
    log_score = model(noisy_target, sigma)
    token_loss = graph.score_entropy(
        log_score, sigma[:, None], noisy_target, input_ids
    )

    weighted_tokens = dsigma[:, None] * token_loss * loss_mask
    per_example_loss = weighted_tokens.sum(dim=-1) / loss_mask.sum(dim=-1).clamp_min(1)
    return (per_example_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)


def read_baseline_score(path):
    """如果存在原始数据集指标，读取其测试 Score Entropy 用于最终对比。"""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle).get("test_score_entropy")


def main():
    """组织奖励数据、混合训练、验证、保存和最终测试。"""
    args = parse_args()
    if args.max_prompt_tokens >= args.max_length:
        raise ValueError("--max-prompt-tokens must be smaller than --max-length")
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("--batch-size and --grad-accum must be positive")

    seed_everything(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = Path(args.candidates)
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {candidate_path}. Run reward_stage1_best_of_n.py first."
        )

    print(f"Loading SFT model {args.model!r} on {device} ...")
    model, graph, noise = load_model_hf(args.model, device)
    model.train()
    noise.eval()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=args.cache_dir)
    splits = build_splits(args.dataset, args.cache_dir, args.seed)
    split_ids = {
        name: [int(value) for value in split["example_id"]]
        for name, split in splits.items()
    }
    with (output_dir / "splits.json").open("w", encoding="utf-8") as handle:
        json.dump(split_ids, handle, indent=2)

    candidate_groups = load_jsonl(candidate_path)
    original_records, generated_records, eligible_groups = build_training_records(
        splits["train"],
        args.dataset,
        candidate_groups,
        args.temperature,
        args.min_best_reward,
    )
    records = original_records + generated_records
    sampler = make_source_sampler(
        len(original_records),
        len(generated_records),
        args.generated_fraction,
        args.seed,
    )
    train_loader = DataLoader(
        RecordDataset(records),
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=RewardWeightedCollator(
            tokenizer, args.max_length, args.max_prompt_tokens
        ),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # 验证和测试始终使用原始固定划分，不使用生成数据。
    eval_collator = S1Collator(
        tokenizer, args.dataset, args.max_length, args.max_prompt_tokens
    )
    valid_loader = DataLoader(
        splits["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=eval_collator,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        splits["test"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=eval_collator,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    step = load_checkpoint(args.resume, model, optimizer, device) if args.resume else 0
    optimizer.zero_grad(set_to_none=True)
    train_iterator = iter(train_loader)
    micro_step = 0
    running_loss = 0.0

    print(
        f"Training rows: original={len(original_records)}, "
        f"generated={len(generated_records)}, eligible_groups={eligible_groups}; "
        f"effective_batch={args.batch_size * args.grad_accum}"
    )
    while step < args.steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        loss_mask = batch["loss_mask"].to(device, non_blocking=True)
        sample_weight = batch["sample_weight"].to(device, non_blocking=True)
        with autocast_context(device):
            loss = reward_weighted_score_entropy_loss(
                model,
                graph,
                noise,
                input_ids,
                loss_mask,
                sample_weight,
            )
        (loss / args.grad_accum).backward()
        running_loss += loss.item()
        micro_step += 1

        if micro_step % args.grad_accum != 0:
            continue

        step += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = learning_rate(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step} weighted_loss={running_loss / args.grad_accum:.6f} "
                f"lr={lr:.3e}"
            )
        running_loss = 0.0

        if step % args.eval_every == 0:
            valid_loss = evaluate(model, graph, noise, valid_loader, device)
            print(f"step={step} validation_score_entropy={valid_loss:.6f}")
        if step % args.save_every == 0:
            save_checkpoint(output_dir, model, optimizer, step, args, split_ids)

    save_checkpoint(output_dir, model, optimizer, step, args, split_ids)
    export_config = OmegaConf.to_container(model.config, resolve=True)
    model.save_pretrained(output_dir / "model", config=export_config)

    test_loss = evaluate(model, graph, noise, test_loader, device)
    baseline_score = read_baseline_score(Path(args.baseline_metrics))
    metrics = {
        "step": step,
        "test_score_entropy": test_loss,
        "original_records": len(original_records),
        "generated_records": len(generated_records),
        "eligible_groups": eligible_groups,
        "generated_fraction": args.generated_fraction,
        "temperature": args.temperature,
    }
    if baseline_score is not None:
        metrics["baseline_test_score_entropy"] = baseline_score
        metrics["test_score_entropy_change"] = test_loss - baseline_score

    if args.sample_count > 0:
        accuracy, generation_path = generate_test_samples(
            model,
            graph,
            noise,
            splits["test"],
            eval_collator,
            tokenizer,
            device,
            args.sample_count,
            args.sample_steps,
            output_dir,
        )
        metrics["sampled_normalized_exact_match"] = accuracy
        metrics["sample_count"] = min(args.sample_count, len(splits["test"]))
        metrics["generation_path"] = str(generation_path)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    comparison = ""
    if baseline_score is not None:
        comparison = f", change={test_loss - baseline_score:+.6f}"
    print(f"Done: test_score_entropy={test_loss:.6f}{comparison}; output={output_dir}")


if __name__ == "__main__":
    main()

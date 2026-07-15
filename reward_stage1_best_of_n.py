"""第一阶段：使用微调后的 SEDD 对训练问题执行 Best-of-N 采样。

对每个问题生成 N 个候选回答，使用可验证奖励给候选打分，并保存：

- 所有候选回答及其奖励；
- 奖励最高的候选回答；
- 是否至少生成过一个最终答案完全正确的候选。

默认只处理训练集前 100 道题，先用于验证流程。确认可以正常运行后，使用
``--count 0`` 处理完整的 800 条训练数据。已有输出默认会断点续跑；如果想
重新生成，传入 ``--overwrite``。
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import GPT2TokenizerFast

from finetune_s1 import (
    S1Collator,
    build_splits,
    choose_device,
    normalize_answer,
    seed_everything,
)
from load_model import load_model_hf
from sampling import get_pc_sampler


DEFAULT_MODEL = "exp_local/s1k-1.1-sft/model"
DEFAULT_OUTPUT = "exp_local/reward-stage1/best_of_n.jsonl"


def parse_args():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", choices=("s1k", "s1k-1.1"), default="s1k-1.1")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=100,
                        help="Number of training prompts; 0 means the full training split")
    parser.add_argument("--candidates", type=int, default=4,
                        help="Number of candidates generated for each prompt")
    parser.add_argument("--sample-steps", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-response-tokens", type=int, default=512,
                        help="Responses longer than this receive a small length penalty")
    parser.add_argument("--exact-reward", type=float, default=1.0)
    parser.add_argument("--format-reward", type=float, default=0.1)
    parser.add_argument("--length-penalty", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--overwrite", action="store_true",
                        help="Discard an existing output file instead of resuming it")
    return parser.parse_args()


def response_before_eos(token_ids, eos_token_id):
    #截取生成序列中第一个 EOS 之前的 token，避免把 EOS 后内容算入回答。
    token_ids = token_ids.detach().cpu()
    eos_positions = (token_ids == eos_token_id).nonzero(as_tuple=False)
    if len(eos_positions) > 0:
        token_ids = token_ids[: int(eos_positions[0, 0])]
    return token_ids.tolist()


def extract_answer(text):

    if "Answer:" in text:
        return text.rsplit("Answer:", 1)[-1].strip()
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return nonempty_lines[-1] if nonempty_lines else ""


def score_candidate(text, token_count, reference, args):
    """
    - 最终答案规范化后完全匹配：``exact_reward``；
    - 同时包含 Reasoning/Answer 标记：``format_reward``；
    - 超过建议响应长度：最多扣除 ``length_penalty``。
    """
    predicted = extract_answer(text)
    exact_match = (
        bool(normalize_answer(reference))
        and normalize_answer(predicted) == normalize_answer(reference)
    )
    format_ok = "Reasoning:" in text and "Answer:" in text

    overflow = max(token_count - args.max_response_tokens, 0)
    overflow_ratio = min(overflow / max(args.max_response_tokens, 1), 1.0)
    penalty = args.length_penalty * overflow_ratio
    reward = (
        args.exact_reward * float(exact_match)
        + args.format_reward * float(format_ok)
        - penalty
    )
    return {
        "predicted": predicted,
        "reward": float(reward),
        "exact_match": bool(exact_match),
        "format_ok": bool(format_ok),
        "token_count": int(token_count),
        "length_penalty": float(penalty),
    }


def generate_candidates(model, graph, noise, example, collator, tokenizer, args, device):
    """为一道题并行生成 N 个回答并计算奖励。"""
    batch = collator([example])
    clean = batch["input_ids"].to(device)
    prompt_length = int(batch["prompt_length"][0])
    clean_prompt = clean[:, :prompt_length].expand(args.candidates, -1)

    def project_prompt(x):
        # 每个反向扩散步骤都恢复问题区域，只允许模型生成回答区域。
        x[:, :prompt_length] = clean_prompt
        return x

    sampler = get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=(args.candidates, collator.max_length),
        predictor="analytic",
        steps=args.sample_steps,
        device=device,
        proj_fun=project_prompt,
    )
    generated_batch = project_prompt(sampler(model))[:, prompt_length:]
    reference = str(example.get("solution") or "")

    candidates = []
    for candidate_id, generated in enumerate(generated_batch):
        response_ids = response_before_eos(generated, tokenizer.eos_token_id)
        text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        candidate = {
            "candidate_id": candidate_id,
            "generated": text,
            **score_candidate(text, len(response_ids), reference, args),
        }
        candidates.append(candidate)

    # 奖励相同时优先选择 exact match、格式正确且更短的候选。
    best_index = max(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["reward"],
            candidates[index]["exact_match"],
            candidates[index]["format_ok"],
            -candidates[index]["token_count"],
        ),
    )
    best = candidates[best_index]
    return {
        "example_id": int(example["example_id"]),
        "question": str(example["question"]),
        "reference": reference,
        "best_index": int(best_index),
        "best_generated": best["generated"],
        "best_predicted": best["predicted"],
        "best_reward": best["reward"],
        "best_exact_match": best["exact_match"],
        "candidates": candidates,
    }


def load_existing_rows(path):
    """读取已经生成的 JSONL，用于断点续跑和重新统计指标。"""
    if not path.exists():
        return []
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


def write_metrics(output_path, rows, args):
    """汇总 Best-of-N 成功率和平均最佳奖励。"""
    metrics = {
        "processed_prompts": len(rows),
        "candidates_per_prompt": args.candidates,
        "sample_steps": args.sample_steps,
        "any_exact_match_rate": (
            sum(bool(row.get("best_exact_match")) for row in rows) / max(len(rows), 1)
        ),
        "mean_best_reward": (
            sum(float(row.get("best_reward", 0.0)) for row in rows) / max(len(rows), 1)
        ),
    }
    metrics_path = output_path.with_name("metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics, metrics_path


def main():
    """加载 SFT 模型，对训练问题执行 Best-of-N 生成并保存奖励数据。"""
    args = parse_args()
    if args.candidates < 1:
        raise ValueError("--candidates must be at least 1")
    if args.count < 0:
        raise ValueError("--count must be non-negative")
    if args.max_prompt_tokens >= args.max_length:
        raise ValueError("--max-prompt-tokens must be smaller than --max-length")

    seed_everything(args.seed)
    device = choose_device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and output_path.exists():
        output_path.unlink()
    existing_rows = load_existing_rows(output_path)
    processed_ids = {int(row["example_id"]) for row in existing_rows}

    print(f"Loading SFT model {args.model!r} on {device} ...")
    model, graph, noise = load_model_hf(args.model, device)
    model.eval()
    noise.eval()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=args.cache_dir)
    collator = S1Collator(
        tokenizer, args.dataset, args.max_length, args.max_prompt_tokens
    )
    train_split = build_splits(args.dataset, args.cache_dir, args.seed)["train"]
    limit = len(train_split) if args.count == 0 else min(args.count, len(train_split))
    selected = train_split.select(range(limit))

    new_count = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for position, example in enumerate(selected, start=1):
            example_id = int(example["example_id"])
            if example_id in processed_ids:
                continue
            row = generate_candidates(
                model, graph, noise, example, collator, tokenizer, args, device
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            existing_rows.append(row)
            processed_ids.add(example_id)
            new_count += 1
            print(
                f"[{position}/{limit}] id={example_id} "
                f"best_reward={row['best_reward']:.3f} "
                f"exact={row['best_exact_match']}"
            )

    metrics, metrics_path = write_metrics(output_path, existing_rows, args)
    print(
        f"Done: new={new_count}, total={len(existing_rows)}, "
        f"exact_rate={metrics['any_exact_match_rate']:.4f}; "
        f"output={output_path}; metrics={metrics_path}"
    )


if __name__ == "__main__":
    main()

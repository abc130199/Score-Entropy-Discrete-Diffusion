"""
本文件复用 ``finetune_s1.py`` 中已经验证过的完整训练流程，包括：

1. 固定 Question token，只给 Reasoning 和 Answer token 添加离散扩散噪声；
2. 使用条件 Score Entropy 作为训练目标；
3. 使用 seed=42 将 1000 条数据固定划分为 800/100/100；
4. 定期验证、保存 checkpoint、导出 Hugging Face 模型并测试；
5. 训练完成后核对 Medium 与 Small 实验是否使用完全相同的数据样本，
   并打印两者的测试集 Score Entropy，便于进行公平对比。

直接运行：

    python finetune_s1_medium.py


    python finetune_s1_medium.py --batch-size 2 --grad-accum 16

这样虽然 micro-batch 变小，有效 batch 仍然是 2 × 16 = 32。
"""

import json
import sys
from pathlib import Path

import finetune_s1



MEDIUM_MODEL = "louaaron/sedd-medium"


DATASET = "s1k-1.1"


MEDIUM_OUTPUT_DIR = Path("exp_local/s1k-1.1-sft-medium")

# 已完成的 Small 实验目录，用于训练结束后的数据划分及测试指标对比。
SMALL_OUTPUT_DIR = Path("exp_local/s1k-1.1-sft")


MEDIUM_BATCH_SIZE = 4
MEDIUM_GRAD_ACCUM = 8


TRAINING_STEPS = 10_000
LEARNING_RATE = 1e-5
RANDOM_SEED = 42


DEFAULT_ARGUMENTS = {
    "--model": MEDIUM_MODEL,
    "--dataset": DATASET,
    "--output-dir": str(MEDIUM_OUTPUT_DIR),
    "--batch-size": MEDIUM_BATCH_SIZE,
    "--grad-accum": MEDIUM_GRAD_ACCUM,
    "--steps": TRAINING_STEPS,
    "--learning-rate": LEARNING_RATE,
    "--seed": RANDOM_SEED,
}


def has_command_line_option(option):
    """判断用户是否已经在命令行中显式提供某个参数。

    同时支持 ``--steps 5000`` 和 ``--steps=5000`` 两种 argparse 写法。
    用户显式给出的值优先于本文件中的默认实验配置。
    """
    return any(
        argument == option or argument.startswith(option + "=")
        for argument in sys.argv[1:]
    )


def install_medium_defaults():
    """把 Medium 默认配置补入命令行，但不覆盖用户显式传入的参数。

    这样可以直接复用 ``finetune_s1.main()``，确保 Medium 和 Small 执行的
    数据处理、损失计算、验证和保存代码完全相同，避免复制两套训练代码后
    逐渐产生实现差异。
    """
    for option, value in DEFAULT_ARGUMENTS.items():
        if not has_command_line_option(option):
            sys.argv.extend([option, str(value)])


def read_json(path):
    """读取 UTF-8 JSON 文件。"""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_with_small(small_dir, medium_dir):
    """核对数据划分，并对比 Small 和 Medium 的最终测试指标。

    Score Entropy 越低越好。只有两次实验使用完全相同的测试样本时，指标
    才适合直接比较，因此这里会先逐项比较两个 ``splits.json`` 文件。
    """
    small_splits_path = small_dir / "splits.json"
    medium_splits_path = medium_dir / "splits.json"
    small_metrics_path = small_dir / "metrics.json"
    medium_metrics_path = medium_dir / "metrics.json"

    required_paths = (
        small_splits_path,
        medium_splits_path,
        small_metrics_path,
        medium_metrics_path,
    )
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        print("Comparison skipped; missing: " + ", ".join(map(str, missing_paths)))
        return

    small_splits = read_json(small_splits_path)
    medium_splits = read_json(medium_splits_path)

    # 比较 800/100/100 的数量，比较example_id。
    same_splits = small_splits == medium_splits

    small_metrics = read_json(small_metrics_path)
    medium_metrics = read_json(medium_metrics_path)
    small_score = small_metrics.get("test_score_entropy")
    medium_score = medium_metrics.get("test_score_entropy")

    difference = medium_score - small_score
    if difference < 0:
        result = f"Medium is better by {-difference:.6f}."
    elif difference > 0:
        result = f"Small is better by {difference:.6f}."
    else:
        result = "The scores are equal."

    warning = "" if same_splits else "\nWarning: Data splits differ."
    print(
        "\nSEDD Small vs. Medium\n"
        f"Same data splits: {'Yes' if same_splits else 'No'}\n"
        f"Test Score Entropy: Small={small_score:.6f}, Medium={medium_score:.6f}\n"
        f"Result: {result}{warning}"
    )


def main():
    """安装 Medium 默认参数，执行复用的微调流程，然后输出实验对比。"""
    install_medium_defaults()
    finetune_s1.main()

  
    actual_args = finetune_s1.parse_args()
    compare_with_small(SMALL_OUTPUT_DIR, Path(actual_args.output_dir))


if __name__ == "__main__":
    main()

"""
GRPO 训练：Qwen2-0.5B-Instruct 算术题（复合奖励 = 正确分 1.0 + 格式分 0.2）

教学重点：
  1. GRPO 与 PPO 的区别：无价值网络、无奖励模型，组内 K 条样本的奖励均值/标准差
     直接归一化出 advantage；beta=0 时连参考模型都省掉（TRL 默认）
  2. 复合奖励塑形：两个独立 reward func，TRL 分别记录曲线，可观察
     "格式先收敛、正确率后爬坡"的典型 RL 训练动态
  3. 难度课程：训练集按基线摸底的 informative group rate 选题（L3/L5 为主），
     太易（全对）和太难（全错）的组 advantage 都是 0，纯属浪费算力

使用方式：
  python src/train_grpo.py                    # 完整训练（默认 200 步）
  python src/train_grpo.py --max_steps 3      # 冒烟测试：验证显存与流程
  python src/train_grpo.py --lora             # 全量跑不动时降级为 LoRA

输出：
  outputs/grpo_ckpt/          # 最终 checkpoint（含 tokenizer）
  outputs/train_log.json      # 每步指标（reward 分量、completion 长度等）
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset

import trl_compat  # noqa: F401  必须先于 trl 导入，修复 trl 0.21 + transformers 5.x 兼容
from trl import GRPOConfig, GRPOTrainer

from probe_baseline import SYSTEM_PROMPT, make_problem, parse_output

ROOT = Path(__file__).parent.parent
MODEL_PATH = Path(r"D:\badou\八斗课程\pretrain_models\Qwen2-0.5B-Instruct")
OUT_DIR = ROOT / "outputs"

# 训练集难度配比：依据 baseline_probe 的 loose_informative_group_rate 选择
# L3 (0.76) / L5 (0.66) 为主，L2 (0.68) 保底；L1/L4/L6 不进训练集，留作泛化评估
LEVEL_MIX = [
    ("L3_addsub_3digit", 0.50),
    ("L5_mul_2x1digit", 0.25),
    ("L2_addsub_2digit", 0.25),
]


def build_dataset(n: int, seed: int) -> Dataset:
    """程序化生成训练集：prompt 为 chat 格式，answer/level 供 reward 函数使用。"""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        r, acc, level = rng.random(), 0.0, LEVEL_MIX[-1][0]
        for lv, p in LEVEL_MIX:
            acc += p
            if r <= acc:
                level = lv
                break
        expr, ans = make_problem(level, rng)
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"计算：{expr} = ?"},
                ],
                "answer": ans,
                "level": level,
            }
        )
    return Dataset.from_list(rows)


# ── 复合奖励：TRL 对多个 reward func 分别记录曲线，最后求和 ──────────────────
def reward_correct(completions, answer, **kwargs):
    """正确分（宽松解析）：有 <answer> 标签取标签内数字，否则取输出中最后一个数字。"""
    rewards = []
    for comp, ans in zip(completions, answer):
        text = comp[0]["content"]
        rewards.append(1.0 if parse_output(text, int(ans))[2] else 0.0)
    return rewards


def reward_format(completions, **kwargs):
    """格式分：输出包含 <answer>数字</answer> 即得分（与正确性解耦）。"""
    return [0.2 if parse_output(comp[0]["content"], 0)[0] else 0.0 for comp in completions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=200, help="优化步数（每步 4 prompt × 8 采样）")
    parser.add_argument("--n_prompts", type=int, default=1000, help="训练集 prompt 数")
    parser.add_argument("--lr", type=float, default=2e-6, help="全量微调学习率")
    parser.add_argument("--lora", action="store_true", help="降级为 LoRA（全量 OOM 时使用）")
    parser.add_argument("--tag", type=str, default="", help="输出目录后缀，用于区分实验")
    parser.add_argument("--log_completions", action="store_true", help="打印每步真实采样补全（调试用）")
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    ckpt_dir = OUT_DIR / (f"grpo_lora_ckpt{suffix}" if args.lora else f"grpo_ckpt{suffix}")
    log_path = OUT_DIR / (f"train_log_lora{suffix}.json" if args.lora else f"train_log{suffix}.json")

    dataset = build_dataset(args.n_prompts, seed=123)

    peft_config = None
    if args.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    config = GRPOConfig(
        output_dir=str(ckpt_dir),
        # 关键坑：本地 Qwen2-0.5B-Instruct 的 config.json 写着 torch_dtype=float16，
        # 不显式指定会被加载成 fp16 → AdamW 的 eps=1e-8 在 fp16 下溢出为 0 →
        # 0/0=NaN，一步训废整个模型（已实测定位）。必须显式 bfloat16。
        model_init_kwargs={"torch_dtype": "bfloat16"},
        # ── GRPO 核心参数 ─────────────────────────────────────────────
        num_generations=8,  # 组内采样数 K：与基线摸底的 pass@8 一致
        beta=0.0,  # KL 系数为 0：不加载参考模型，省 1GB 显存
        epsilon=0.2,  # PPO-clip 裁剪范围
        temperature=1.0,  # 采样温度：保持组内多样性
        max_prompt_length=128,
        max_completion_length=64,
        # ── 批次：8 completions/微批 × 累积 4 = 每步 4 prompt × 8 采样 ──
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        # ── 训练超参 ──────────────────────────────────────────────────
        learning_rate=args.lr if not args.lora else 2e-4,
        max_steps=args.max_steps,
        bf16=True,
        # 关键坑：transformers 5.x 下 gradient checkpointing + train 模式会让
        # generate 输出完全损坏（已实测二分定位），GRPO 必须关闭它；
        # 0.5B 模型激活值不大，8GB 显存不开 checkpointing 也够
        gradient_checkpointing=False,
        # ── 日志与保存 ────────────────────────────────────────────────
        logging_steps=5,
        save_strategy="no",  # 只保存最终 checkpoint，节省磁盘
        report_to=[],
        seed=42,
        log_completions=args.log_completions,
    )

    trainer = GRPOTrainer(
        model=str(MODEL_PATH),
        args=config,
        reward_funcs=[reward_correct, reward_format],
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train()

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ckpt_dir))
    trainer.processing_class.save_pretrained(str(ckpt_dir))

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"\n训练完成。checkpoint: {ckpt_dir}")
    print(f"训练日志: {log_path}")
    print(f"GPU 峰值显存: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()

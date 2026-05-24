"""
文本生成：四种解码策略对比演示

教学重点：
  1. Greedy Decoding：每步取概率最高的 token，确定性但容易陷入重复
  2. Temperature Sampling：logits / T，T<1 更"保守"，T>1 更"随机"
  3. Top-K Sampling：只从概率最高的 K 个 token 中采样
  4. Top-P（Nucleus）Sampling：从累积概率超过 p 的最小集合中采样
     直觉：Top-K 固定候选数量，Top-P 根据分布"自适应"候选数量

使用方式：
  python generate.py                         # 用默认 prompt 对比四种策略
  python generate.py --prompt "人工智能"      # 自定义起始文本
  python generate.py --max_new_tokens 100    # 生成更多 token
  python generate.py --compare               # 四种策略并排输出对比表

依赖：
  pip install transformers torch
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.models.bert.tokenization_bert import BertTokenizerFast

from model import build_model

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
CKPT_DIR = BASE_DIR / "outputs" / "checkpoints"


def get_tokenizer():
    local_path = Path(__file__).parent.parent.parent / "pretrain_models" / "bert-base-chinese"
    if local_path.exists():
        return BertTokenizerFast.from_pretrained(str(local_path))
    return BertTokenizerFast.from_pretrained("bert-base-chinese")


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_model(vocab_size=ckpt["vocab_size"], seq_len=ckpt["seq_len"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["seq_len"]


@torch.no_grad()
def generate(
        model,
        input_ids: torch.Tensor,
        max_new_tokens: int = 80,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
) -> torch.Tensor:
    """
    自回归文本生成

    每一步：
      1. 前向传播，取最后一个位置的 logits
      2. 按策略选 next token
      3. 拼接到序列末尾，继续下一步

    strategy 参数：
      "greedy"      - 贪心，每步取 argmax
      "temperature" - 温度采样，temperature 参数控制分布平滑度
      "top_k"       - Top-K 采样，从最高 K 个 token 中随机采样
      "top_p"       - Top-P（Nucleus）采样，从累积概率 ≥ p 的最小集合中采样
    """
    device = input_ids.device
    seq_len = model.seq_len
    generated = input_ids.clone()  # (1, T)

    for _ in range(max_new_tokens):
        # 如果序列超过 seq_len，截取最后 seq_len 个 token 作为上下文
        context = generated[:, -seq_len:]

        logits = model(context)  # (1, T, V)
        next_logits = logits[0, -1, :]  # (V,) 只取最后一步的预测

        if strategy == "greedy":
            # 贪心：直接取最高分
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # (1,)

        else:
            # 采样前统一做 temperature 缩放
            next_logits = next_logits / max(temperature, 1e-8)

            if strategy == "top_k":
                # Top-K：将低于第 K 大的 logits 置为 -inf
                values, _ = torch.topk(next_logits, top_k)
                threshold = values[-1]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            elif strategy == "top_p":
                # Top-P（Nucleus）：按概率从大到小排序，累积到 ≥ p 时截断
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # 找到刚好超过 p 的位置，将该位置之后的 token 排除
                sorted_indices_to_remove = cumprobs > top_p
                # 保留至少 1 个 token：右移一位，使超过阈值的第一个位置不被移除
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float("-inf")

            elif strategy == "temperature":
                pass  # 只做 temperature 缩放，不做 top-k/p 截断

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1,)

        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)  # (1, T+1)

    return generated[0]  # (total_len,)


def decode_text(tokenizer, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
    return tokenizer.decode(ids.tolist(), skip_special_tokens=skip_special_tokens)


def compare_strategies(model, tokenizer, prompt: str, max_new_tokens: int, device: torch.device):
    """四种解码策略并排对比"""
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    strategies = [
        ("Greedy", dict(strategy="greedy")),
        ("Temperature", dict(strategy="temperature", temperature=0.8)),
        ("Top-K (K=50)", dict(strategy="top_k", temperature=0.8, top_k=50)),
        ("Top-P (p=0.9)", dict(strategy="top_p", temperature=0.8, top_p=0.9)),
    ]

    print(f"\n{'=' * 70}")
    print(f"Prompt：{prompt}")
    print(f"{'=' * 70}")

    for name, kwargs in strategies:
        out_ids = generate(model, input_ids, max_new_tokens=max_new_tokens, **kwargs)
        new_ids = out_ids[prompt_len:]
        generated_text = decode_text(tokenizer, new_ids)
        print(f"\n【{name}】")
        print(f"{prompt}{generated_text}")
        print("-" * 50)

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="中国的首都是",
                        help="生成起始文本，默认 '中国的首都是'")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--strategy", type=str, default="top_p",
                        choices=["greedy", "temperature", "top_k", "top_p"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--compare", action="store_true", help="并排对比四种策略")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else CKPT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint：{ckpt_path}，请先运行 train.py")

    tokenizer = get_tokenizer()
    model, seq_len = load_model(ckpt_path, device)
    logger.info(f"模型加载完毕，序列长度={seq_len}，设备={device}")

    if args.compare:
        compare_strategies(model, tokenizer, args.prompt, args.max_new_tokens, device)
    else:
        input_ids = tokenizer.encode(
            args.prompt, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        prompt_len = input_ids.shape[1]

        out_ids = generate(
            model, input_ids,
            max_new_tokens=args.max_new_tokens,
            strategy=args.strategy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        new_ids = out_ids[prompt_len:]
        generated_text = decode_text(tokenizer, new_ids)
        print(f"\n{'=' * 50}")
        print(f"策略：{args.strategy}")
        print(f"{'=' * 50}")
        print(f"{args.prompt}{generated_text}")
        print()


if __name__ == "__main__":
    main()

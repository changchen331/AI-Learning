# USAGE_GUIDE.md — GRPO 算术题项目：代码调用与测试指南

## 1. 环境准备

```bash
pip install -r requirements.txt
```

| 依赖         | 版本        | 用途                                                                        |
|--------------|-------------|-----------------------------------------------------------------------------|
| torch        | 2.6.0+cu126 | 训练框架                                                                    |
| transformers | 5.5.3       | 模型加载（注意：5.x 与 trl 0.21 有兼容问题，已由 `src/trl_compat.py` 处理） |
| trl          | 0.21.0      | GRPOTrainer                                                                 |
| peft         | 0.15.0      | 可选，`--lora` 降级方案                                                     |
| accelerate   | 1.5.2       | Trainer 后端                                                                |
| matplotlib   | —           | 训练曲线绘图                                                                |

**预训练模型**：`D:\badou\八斗课程\pretrain_models\Qwen2-0.5B-Instruct`（已落盘，无需下载）。 如路径不同，修改
`src/probe_baseline.py` 和 `src/train_grpo.py` 顶部的 `MODEL_PATH`。

**硬件要求**：本项目在 RTX 4060 Laptop（8GB 显存）上验证通过，全量微调峰值 6.07GB。 显存更小的机器用 `--lora` 降级。

## 2. 各步骤流程

### Step 1：基线摸底（`src/probe_baseline.py`）

```bash
python src/probe_baseline.py              # 全量：6 难度 × 50 题，K=8 采样
python src/probe_baseline.py --quick      # 快速验证：每难度 10 题
```

内部流程：

1. 按 6 个难度级别程序化生成算术题（个位数加法 → 两位数乘法）
2. 每个 prompt 两种解码：greedy ×1（测确定性能力）+ 温度 1.0 采样 ×8（测 pass@k）
3. 输出关键指标：greedy 正确率、格式遵循率、pass@8、 **informative group rate**
   （组内有对有错的比例——GRPO 可学习性的核心指标）

预期输出（基线）：L3/L5 的 informative 在 0.66~0.76，L6 只有 0.24； 所有难度格式率≈0（模型完全无视 `<answer>` 指令）。 结果保存到
`outputs/baseline_probe.json`，耗时约 30 秒。

### Step 2：GRPO 训练（`src/train_grpo.py`）

```bash
python src/train_grpo.py                       # 完整训练：200 步，约 3.3 分钟
python src/train_grpo.py --max_steps 3 --tag smoke   # 冒烟测试
python src/train_grpo.py --lora                # 显存不足时降级 LoRA
python src/train_grpo.py --log_completions     # 打印每步真实采样（调试用）
```

| 参数          | 默认 | 说明                                     |
|---------------|------|------------------------------------------|
| `--max_steps` | 200  | 优化步数；每步 = 4 prompt × 8 采样       |
| `--n_prompts` | 1000 | 训练集大小（L3 50% / L5 25% / L2 25%）   |
| `--lr`        | 2e-6 | 全量微调学习率（`--lora` 时自动用 2e-4） |
| `--lora`      | 关   | LoRA r=16，注意力四层                    |
| `--tag`       | 空   | 输出目录后缀，区分实验                   |

训练日志每 5 步打印：两个奖励分量、`frac_reward_zero_std`、策略熵、clip 比例等。 输出：`outputs/grpo_ckpt/`（checkpoint）+
`outputs/train_log.json`（指标序列）。

健康的训练标志：前 25 步 `rewards/reward_correct/mean` 从 ~0.55 升到 0.85+； 若奖励恒为 0 或补全长乱码，见 §4 FAQ。

### Step 3：训练后评估（复用 probe 脚本）

```bash
# 全量微调 checkpoint
python src/probe_baseline.py --model outputs/grpo_ckpt --out outputs/post_train_probe.json --seed 42
# LoRA checkpoint（脚本自动识别 adapter_config.json，加载基座 + adapter）
python src/probe_baseline.py --model outputs/grpo_lora_ckpt --out outputs/post_train_probe_lora.json --seed 42
```

**必须保持 `--seed 42` 与基线一致**，保证评估题完全相同，前后可配对比较。 LoRA checkpoint 只含 adapter 权重，脚本检测到
`adapter_config.json` 会自动 先加载基座 `MODEL_PATH` 再挂载 adapter。

### Step 4：对比分析（`src/compare_results.py`）

```bash
python src/compare_results.py
```

输出：三方对比表（基线 / 全量 / LoRA 的格式率、greedy 正确率、pass@8）、 逐题样例对照、训练曲线图
`outputs/figures/train_curves.png`（两条方案叠加）。 若 LoRA 的 probe/log 文件不存在则自动退化为基线 vs 全量两方对比。

## 3. 作为模块调用

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt = "outputs/grpo_ckpt"  # 或基座模型路径，对比行为差异
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, device_map="cuda")

msgs = [
    {"role": "system",
     "content": "你是一个算术助手。用户会给你一道算术题，请计算出结果，并把最终答案放在 <answer> 标签中，例如 <answer>42</answer>。不要输出其他内容。"},
    {"role": "user", "content": "计算：47 + 38 = ?"},
]
text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
enc = tokenizer(text, return_tensors="pt").to("cuda")
out = model.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tokenizer.pad_token_id)
print(tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
# 基座模型: '85'          （无格式）
# 训练后:   '<answer>85</answer>'
```

自定义奖励函数做实验（修改 `src/train_grpo.py`）：

```python
def reward_format(completions, **kwargs):
    # 把 0.2 改成 1.0，观察"格式秒收敛、正确率爬坡变慢"的信号竞争现象
    return [1.0 if parse_output(comp[0]["content"], 0)[0] else 0.0 for comp in completions]
```

## 4. 调试与常见问题

**Q1：`from trl import GRPOTrainer` 报 `No module named 'vllm'`？**
trl 0.21 与 transformers 5.x 的已知不兼容（`_is_package_available` 返回值类型变更）。 所有脚本第一行 `import trl_compat`
已修复，确保从项目根目录运行、不要删这个文件。

**Q2：训练中奖励全为 0、补全全是乱码？**
检查是否打开了 gradient checkpointing——transformers 5.x 下它会让 `generate` 输出损坏。 本项目默认关闭。不要为了省显存重新打开，显存不够用
`--lora`。

**Q3：训练一步后模型输出全变成 `!!!!!`？**
权重被 fp16 训废了。确认 `train_grpo.py` 里 `model_init_kwargs={"torch_dtype": "bfloat16"}` 存在。 根因：本地 Qwen2-0.5B 的
config.json 写的是 fp16，不显式指定会按 fp16 加载， AdamW 的 eps=1e-8 在 fp16 下溢出为 0。

**Q4：CUDA OOM？**
按顺序尝试：`--lora`；减小 `per_device_train_batch_size`（8→4，同时把
`gradient_accumulation_steps` 4→8 保持每步 prompt 数不变）。

**Q5：想换自己的任务/题型？**
改三处：`probe_baseline.py` 的 `make_problem`（题目生成）、`train_grpo.py` 的
`LEVEL_MIX`（难度配比）和两个 reward 函数（奖励判定）。先跑 probe 确认 informative group rate 在 0.3~0.8 之间再训练。

**Q6：`epoch` 显示 0.8 没跑完一整轮？**
正常。训练集 1000 题，200 步 × 4 题/步 = 800 题，不到一个 epoch。 GRPO 是在线采样算法，题目是否重复不重要（每题每次都会重新采样
8 条）。

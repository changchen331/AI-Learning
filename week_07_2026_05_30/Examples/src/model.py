"""
BertNER（线性头）和 BertCRFNER（CRF头）两个模型

教学重点：
  1. 线性头（BertNER）：每个 token 独立预测标签
     - 问题：softmax 的独立预测忽略标签间的依赖关系
     - 可能产生非法序列：B-name 后接 I-company，I-name 开头等

  2. CRF 层（BertCRFNER）：加入转移矩阵，全局最优解码
     - 转移矩阵学习"什么标签之后可以接什么标签"
     - Viterbi 算法保证输出合法序列，永远不会 B-name 后接 I-company
     - 代价：训练时需要前向-后向算法，比线性头慢约 20~30%

  3. 两者区别的量化：evaluate.py 会统计非法序列数

依赖：
  pip install pytorch-crf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel


def _load_bert(bert_path: str) -> BertModel:
    prev = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    bert = BertModel.from_pretrained(bert_path)
    transformers.logging.set_verbosity(prev)
    return bert


class BertNER(nn.Module):
    """BERT + 线性分类头，逐 token 独立预测 BIO 标签。

    前向过程：
      input_ids → BertModel → last_hidden_state (B, L, 768)
               → Dropout → Linear(768, num_labels) → logits (B, L, num_labels)

    损失：CrossEntropy，ignore_index=-100 跳过特殊token和非首子词
    预测：argmax(logits, dim=-1)
    """

    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
            labels: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        seq_output = outputs.last_hidden_state  # (B, L, H)
        logits = self.classifier(self.dropout(seq_output))  # (B, L, num_labels)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss


class BertCRFNER(nn.Module):
    """BERT + CRF 层，全局最优序列解码。

    与 BertNER 的区别：
      - Linear 输出称为 emissions（发射分数），不直接 argmax
      - CRF 在 emissions 上叠加转移矩阵，用 Viterbi 找全局最优序列
      - 损失：负对数似然（CRF 内部计算前向-后向）
      - 解码：self.crf.decode() 返回保证合法的标签序列

    CRF 的约束（自动学习）：
      - 初始只能以 O 或 B-X 开头
      - B-X 之后只能是 I-X 或 B-Y 或 O
      - I-X 之后只能是 I-X 或 B-Y 或 O
    """

    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        from torchcrf import CRF

        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def _get_emissions(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        seq_output = outputs.last_hidden_state
        return self.classifier(self.dropout(seq_output))  # (B, L, num_labels)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
            labels: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()  # CRF 要求 BoolTensor

        loss = None
        if labels is not None:
            # CRF 不支持 ignore_index，将 -100 替换为 0（PAD 位置被 mask 屏蔽，不影响梯度）
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = 0
            # crf() 返回对数似然（正值），取负得到损失
            loss = -self.crf(emissions, labels_crf, mask=mask, reduction="mean")

        return emissions, loss

    def decode(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
    ) -> list[list[int]]:
        """Viterbi 解码，返回 list[list[int]]，每条序列长度等于实际token数（不含PAD）。"""
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=mask)


def build_model(
        use_crf: bool,
        bert_path: str,
        num_labels: int,
        dropout: float = 0.1,
) -> nn.Module:
    """模型工厂函数。"""
    model_cls = BertCRFNER if use_crf else BertNER
    model = model_cls(bert_path=bert_path, num_labels=num_labels, dropout=dropout)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = "BERT + CRF" if use_crf else "BERT + Linear"
    print(f"模型：{model_name}")
    print(f"  标签数：{num_labels}")
    print(f"  参数总量：{total_params / 1e6:.1f}M")
    print(f"  可训练参数：{trainable_params / 1e6:.1f}M")
    return model

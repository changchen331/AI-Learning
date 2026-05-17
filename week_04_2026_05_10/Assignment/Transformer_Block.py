"""
手动实现 Transformer 层
"""
import math

import numpy as np
import torch
from transformers import BertModel

bert = BertModel.from_pretrained(r"../bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()


class BertTransformerBlock:
    def __init__(self, states):
        self.layer_num = bert.config.num_hidden_layers
        self.num_attention_heads = 12
        self.hidden_size = 768

        # embedding部分
        self.word_embeddings = states["embeddings.word_embeddings.weight"].detach().numpy()
        self.position_embeddings = states["embeddings.position_embeddings.weight"].detach().numpy()
        self.token_type_embeddings = states["embeddings.token_type_embeddings.weight"].detach().numpy()
        self.embeddings_layer_norm_weight = states["embeddings.LayerNorm.weight"].detach().numpy()
        self.embeddings_layer_norm_bias = states["embeddings.LayerNorm.bias"].detach().numpy()
        self.transformer_weights = []

        self.get_weights(states)

    def get_weights(self, states):
        # transformer部分，有多层
        for i in range(self.layer_num):
            q_w = states[f"encoder.layer.{i}.attention.self.query.weight"].detach().numpy()
            q_b = states[f"encoder.layer.{i}.attention.self.query.bias"].detach().numpy()
            k_w = states[f"encoder.layer.{i}.attention.self.key.weight"].detach().numpy()
            k_b = states[f"encoder.layer.{i}.attention.self.key.bias"].detach().numpy()
            v_w = states[f"encoder.layer.{i}.attention.self.value.weight"].detach().numpy()
            v_b = states[f"encoder.layer.{i}.attention.self.value.bias"].detach().numpy()
            attention_output_weight = states[f"encoder.layer.{i}.attention.output.dense.weight"].detach().numpy()
            attention_output_bias = states[f"encoder.layer.{i}.attention.output.dense.bias"].detach().numpy()
            attention_layer_norm_w = states[f"encoder.layer.{i}.attention.output.LayerNorm.weight"].detach().numpy()
            attention_layer_norm_b = states[f"encoder.layer.{i}.attention.output.LayerNorm.bias"].detach().numpy()
            intermediate_weight = states[f"encoder.layer.{i}.intermediate.dense.weight"].detach().numpy()
            intermediate_bias = states[f"encoder.layer.{i}.intermediate.dense.bias"].detach().numpy()
            output_weight = states[f"encoder.layer.{i}.output.dense.weight"].detach().numpy()
            output_bias = states[f"encoder.layer.{i}.output.dense.bias"].detach().numpy()
            ff_layer_norm_w = states[f"encoder.layer.{i}.output.LayerNorm.weight"].detach().numpy()
            ff_layer_norm_b = states[f"encoder.layer.{i}.output.LayerNorm.bias"].detach().numpy()
            self.transformer_weights.append(
                [q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                 attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                 output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])

    # bert embedding，使用3层叠加，在经过一个Layer norm层
    def embedding_forward(self, x):
        # x.shape = [max_len]
        we = self.get_embedding(self.word_embeddings, x)
        # position embeding的输入 [0, 1, 2, 3]
        pe = self.get_embedding(self.position_embeddings, np.array(list(range(len(x)))))
        # token type embedding,单输入的情况下为[0, 0, 0, 0]
        te = self.get_embedding(self.token_type_embeddings, np.array([0] * len(x)))
        embedding = we + pe + te
        # 加和后有一个归一化层
        embedding = self.layer_norm(embedding,
                                    self.embeddings_layer_norm_weight,
                                    self.embeddings_layer_norm_bias)
        return embedding

    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x])

    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]

        # 取出该层的参数，在实际中，这些参数都是随机初始化，之后进行预训练
        q_w, q_b, k_w, k_b, v_w, v_b, \
            attention_output_weight, attention_output_bias, \
            attention_layer_norm_w, attention_layer_norm_b, \
            intermediate_weight, intermediate_bias, \
            output_weight, output_bias, \
            ff_layer_norm_w, ff_layer_norm_b = weights

        # self attention层
        attention_output = self.self_attention(
            x, q_w, q_b, k_w, k_b, v_w, v_b,
            attention_output_weight, attention_output_bias,
            self.num_attention_heads, self.hidden_size)

        # ln层，并使用了残差机制
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)

        # feed forward层
        feed_forward_x = self.feed_forward(x, intermediate_weight, intermediate_bias,
                                           output_weight, output_bias)

        # ln层，并使用了残差机制
        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        return x

    # self attention的计算
    def self_attention(self, x, q_w, q_b, k_w, k_b, v_w, v_b,
                       attention_output_weight, attention_output_bias,
                       num_attention_heads, hidden_size):
        # x.shape = max_len * hidden_size
        # q_w, k_w, v_w  shape = hidden_size * hidden_size
        # q_b, k_b, v_b  shape = hidden_size
        q = x @ q_w.T + q_b  # shape: [max_len, hidden_size]
        k = x @ k_w.T + k_b  # shape: [max_len, hidden_size]
        v = x @ v_w.T + v_b  # shape: [max_len, hidden_size]

        attention_head_size = int(hidden_size / num_attention_heads)
        # q.shape = num_attention_heads, max_len, attention_head_size
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)
        # k.shape = num_attention_heads, max_len, attention_head_size
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)
        # v.shape = num_attention_heads, max_len, attention_head_size
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)

        # qk.shape = num_attention_heads, max_len, max_len
        qk = np.matmul(q, k.swapaxes(1, 2))
        qk /= np.sqrt(attention_head_size)
        qk = softmax(qk)
        # qkv.shape = num_attention_heads, max_len, attention_head_size
        qkv = np.matmul(qk, v)
        # qkv.shape = max_len, hidden_size
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        # attention.shape = max_len, hidden_size
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        return attention

    # 多头机制
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        # hidden_size = 768  num_attent_heads = 12 attention_head_size = 64
        _, max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(1, 0)  # output shape = [num_attention_heads, max_len, attention_head_size]
        return x


    # 前馈网络的计算
    def feed_forward(self, x,
                     intermediate_weight,  # intermediate_size, hidden_size
                     intermediate_bias,  # intermediate_size
                     output_weight,  # hidden_size, intermediate_size
                     output_bias,  # hidden_size
                     ):
        # output shape: [max_len, intermediate_size]
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        # output shape: [max_len, hidden_size]
        x = np.dot(x, output_weight.T) + output_bias
        return x


    # 归一化层
    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x


    def forward(self, x):
        x = self.embedding_forward(x)
        return self.single_transformer_layer_forward(x, 0)


# softmax归一化
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


# gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))


if __name__ == '__main__':
    X = np.array([2450, 15486, 102, 2110])  # 假想成4个字的句子
    torch_x = torch.LongTensor([X])

    mine_bert_transformer = BertTransformerBlock(state_dict)
    output = mine_bert_transformer.forward(torch_x)
    print(output.shape)
    print(output)

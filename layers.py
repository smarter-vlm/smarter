# New code. Copyright Denisa Roberts 2024

# References

# adsformers https://ui.adsabs.harvard.edu/abs/2023arXiv230201255A/abstract
# eficient vit image representations https://www.researchgate.net/profile/Denisa-Roberts/publication/370980888_Efficient_Large-Scale_Vision_Representation_Learning/links/64ecf9d99b1e56033da9d827/Efficient-Large-Scale-Vision-Representation-Learning.pdf

# prismatic vlm https://arxiv.org/pdf/2402.07865.pdf
# qformer https://arxiv.org/pdf/2301.12597

# mbert https://link.springer.com/chapter/10.1007/978-3-030-72240-1_36

import math
import torch
from torch import nn
import torch.nn.functional as F


class CLayer(nn.Module):
    def __init__(self, dim, args):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=args.ln_eps)

    def forward(self, inputs):
        return self.ln(torch.cat(inputs, dim=1))


class QFLayer(nn.Module):
    def __init__(self, num_heads, args):
        super().__init__()
        self.intermediate = QFIntermediate(args)
        self.mha = QFAttentionMH(num_attention_heads=num_heads, args=args)
        self.crossattention = QFAttentionMH(
            num_attention_heads=num_heads,
            hidden_size=768,
            encoder_hidden_size=args.repr_size,
            max_position_embeddings=110,  # max question len
            is_cross_attention=True,
            args=args,
        )

    def forward(self, im_repr, q_repr):

        # q_repr is siglip encoding of the text sequence with max len 110
        q_attn = self.mha(q_repr)

        # this is the vision encoder hidden projected
        vision_encoder = torch.unsqueeze(im_repr, 1)
        vision_encoder = vision_encoder.expand(-1, q_repr.shape[1], -1)  # seq len
        x = self.crossattention(q_attn, vision_encoder).mean(1)

        x = self.intermediate(x)
        return x


class QFIntermediate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(768, args.h_sz)
        self.intermediate_act_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(768, eps=args.ln_eps)
        self.dropout = nn.Dropout(args.pdrop)
        self.dense_final = nn.Linear(args.h_sz, 768)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.intermediate_act_fn(x)

        x = self.dense_final(x)
        x = self.dropout(x)
        x = self.layer_norm(hidden_states + x)
        return x


# Inspired from https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py
class QFAttentionMH(nn.Module):
    def __init__(
        self,
        num_attention_heads,
        hidden_size=768,  # siglip repr dim
        encoder_hidden_size=768,
        max_position_embeddings=110,  # max q len
        is_cross_attention=False,
        args=None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        # hidden size must be multiple of num heads
        self.hidden_size = hidden_size
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.max_position_embeddings = max_position_embeddings
        self.query = nn.Linear(hidden_size, self.all_head_size)

        if is_cross_attention:
            self.key = nn.Linear(encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(encoder_hidden_size, self.all_head_size)
        else:
            # self attention
            self.key = nn.Linear(hidden_size, self.all_head_size)
            self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(args.pdrop)
        self.distance_embedding = nn.Embedding(
            2 * max_position_embeddings - 1, self.attention_head_size
        )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, encoder_hidden_states=None):
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            # encoder is vision; key is text
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        else:  # self attn
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # print("hidden states size", hidden_states.size())

        seq_length = hidden_states.size()[1]
        position_ids_l = torch.arange(
            seq_length, dtype=torch.long, device=hidden_states.device
        ).view(-1, 1)
        position_ids_r = torch.arange(
            seq_length, dtype=torch.long, device=hidden_states.device
        ).view(1, -1)
        distance = position_ids_l - position_ids_r
        positional_embedding = self.distance_embedding(
            distance + self.max_position_embeddings - 1
        )
        positional_embedding = positional_embedding.to(
            dtype=query_layer.dtype
        )  # fp16 compatibility

        relative_position_scores = torch.einsum(
            "bhld,lrd->bhlr", query_layer, positional_embedding
        )
        attention_scores = attention_scores + relative_position_scores

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs_dropped = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # bc of all the heads
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class QV_Fusion(nn.Module):
    def __init__(self, in_dim, out_dim, args):
        super().__init__()
        self.ln1 = nn.Linear(in_dim, out_dim)
        self.ln2 = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim, eps=args.ln_eps)

    def forward(self, x):
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.ln2(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        return x


class PuzzleMLPDecoder(nn.Module):
    def __init__(self, out_dim, num_classes):
        super().__init__()
        self.ln1 = nn.Linear(out_dim, out_dim)
        self.ln2 = nn.Linear(out_dim, out_dim)
        self.ln3 = nn.Linear(out_dim, num_classes)

    def forward(self, hidden_repr):
        x = self.ln1(hidden_repr)
        x = F.gelu(x)
        x = self.ln2(x)
        x = F.gelu(x)
        x = self.ln3(x)
        return x


def get_activation_layer(is_baseline):
    if is_baseline:
        return nn.ReLU()
    else:
        return nn.GELU()


def get_activation_fn(is_baseline):
    if is_baseline:
        return F.relu
    else:
        return F.gelu

import math
import torch
from torch import nn
import torch.nn.functional as F

class CLayer(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, im_repr, q_repr):
        return torch.cat([im_repr, q_repr], dim=1)


# Inspired from https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py
class QFLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.intermediate = QFIntermediate()
        self.mha = QFAttentionMH()
        # TODO DR add a num heads arg
        self.crossattention = QFAttentionMH(num_attention_heads=1, hidden_size=768, encoder_hidden_size=128, max_position_embeddings=110, is_cross_attention=True)

    def forward(self, im_repr, q_repr):
        # q_repr is siglip encoding of the text sequence with max len 110
        # q_attn = self.mha(q_repr).mean(1) # TODO DR -this can also be concat of all seq token repr
        q_attn = self.mha(q_repr)
        # for now concat all heads together
        print("self attn text output shape ", q_attn.shape) #B, 896
        print("what is the project fused vision rep shape ", im_repr.shape) # B, 128; this is projected fused

        # x = torch.cat(
        #     [im_repr, q_attn], dim=1
        # )  # STOP GAP DR; TODO here is cross attn

        # batch, proj_dim = im_repr.shape
        vision_encoder= torch.unsqueeze(im_repr, 1)
        # print("expanded dim vision encoder shape",vision_encoder.shape)
        vision_encoder = vision_encoder.expand(-1, q_repr.shape[1], -1) #seq len
        # print("expanded vision encoder ",vision_encoder[:,0,:]==vision_encoder[:,1,:])
        x = self.crossattention(q_attn, vision_encoder).mean(1)

        print("\nWhat is the output shape after cross attn with mh self attn on siglip encoded text queries and projected fused vision encoded key and vals", x.shape)
        # TODO: add a residual back from vision and from mean text maybe
        x = self.intermediate(x)
        return x


class QFIntermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 256)  # TODO DR shapes/hidden sizes
        self.intermediate_act_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        self.dense_final = nn.Linear(256, 768)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.intermediate_act_fn(x)

        x = self.dense_final(x)
        x = self.dropout(x)
        x = self.layer_norm(hidden_states + x) 
        # x = self.layer_norm(x)
        return x


class QFAttentionMH(nn.Module):
    def __init__(
        self,
        num_attention_heads=1,
        hidden_size=768,  # this needs to match what gets out of siglip unless I want to project it down first
        encoder_hidden_size=768,
        max_position_embeddings=110,
        is_cross_attention=False,
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

        self.dropout = nn.Dropout(0.1)
        self.distance_embedding = nn.Embedding(
            2 * max_position_embeddings - 1, self.attention_head_size
        )

    def transpose_for_scores(self, x):
        # print("what is shape of x in transpose", x.shape)
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        # new_x_shape = x.size() + (self.num_attention_heads, self.attention_head_size)
        # print("what is the new shape", new_x_shape)
        x = x.view(*new_x_shape)
        return x.permute(
            0, 2, 1, 3
        )  # I don't have 4 dims # TODO - this is not fully debuged here
        # return x.permute(0, 2, 1)

    def forward(self, hidden_states, encoder_hidden_states=None):
        # in cross_attention - keys values come from encoder so don't want to attend to encoded padding
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            # encoder is vision; key is text
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        else:  # self attn
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)
        print("mixed query layer size", mixed_query_layer.shape)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # "relative_key" type of pos embed

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
        # print("pos embed shape: ", positional_embedding.shape)

        relative_position_scores = torch.einsum(
            "bhld,lrd->bhlr", query_layer, positional_embedding
        )
        attention_scores = attention_scores + relative_position_scores

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs_dropped = self.dropout(attention_probs)
        print("attention probs dropped and value l shapes",attention_probs_dropped.shape, value_layer.shape )
        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # bc of all the heads
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class QV_Fusion(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.ln1 = nn.Linear(in_dim, out_dim)
        self.ln2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.ln2(x)
        x = F.gelu(x)
        return x
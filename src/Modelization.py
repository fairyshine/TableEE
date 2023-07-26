from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from HgfaceModule.modeling_bert import BertModel

from WillMindS.model.globalpointer import SinusoidalPositionEmbedding

from Getdata import ARGUMENT_TYPE_HASH,id_2_label

# def get_tokens_mask(tokens: Dict[str, torch.Tensor], size: int) -> torch.BoolTensor:
#     """
#     Return (sub-)token mask that removed special tokens.
#     """
#     if 'mask' in tokens:
#         mask = tokens['mask']
#     else:
#         mask = tokens['attention_mask']
#     if mask.size(1) > size:
#         # remove top num added special tokens positions
#         mask = mask[:, mask.size(1) - size :]
#     return mask  # type: ignore

class PTM_globalpointer(nn.Module):
    """GlobalPointer model.
    ref: https://arxiv.org/abs/2208.03054
    ref: https://github.com/xhw205/Efficient-GlobalPointer-torch
    """
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
        self.PreTrained = BertModel.from_pretrained(config.checkpoint)
        self.PreTrained.Hello()

        self.id_to_label = {int(k): v for k, v in id_2_label.items()}
        num_labels = len(id_2_label)
        self.num_classes = num_labels + 1

        self.hidden_size = self.PreTrained.config.hidden_size
        self.inner_dim = self.config.inner_dim
        self.token_inner_embed_ffn = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.type_score_ffn = nn.Linear(self.hidden_size, num_labels * 2)
        self.pos_embed = SinusoidalPositionEmbedding(self.inner_dim, 'zero',custom_position_ids=self.config.custom_RoPE_pos)

        if self.config.table_self_att:
            self.table_attention = nn.MultiheadAttention(self.hidden_size, 1, batch_first=True,dropout=self.config.dropout)

        self.use_dropout = self.config.dropout > 0.0
        if self.use_dropout:
                self.dropout = nn.Dropout(self.config.dropout)

        self.gelu = nn.GELU()

    def forward(self,data):
        entity_score = self._forward(data)  #  batch,38,512,512
        if self.config.TRAIN_MODE:
            onehot = nn.functional.one_hot(data['argument_globalpointer_label'], self.num_classes)
            label_matrix = onehot.permute(0, 3, 1, 2)[:, 1:, ...]
            loss = self._calculate_loss(entity_score, label_matrix)
            outputs = {'loss': loss, 'logits': entity_score, 'label':label_matrix}
        else:
            assert self.config.TEST_MODE
            onehot = nn.functional.one_hot(data['argument_globalpointer_label'], self.num_classes)
            label_matrix = onehot.permute(0, 3, 1, 2)[:, 1:, ...]
            print("===============")
            print("entity_score.size() ",entity_score.size())
            predicts = self.decode(entity_score) #输出起始位置、类型的三元组

            outputs = {'logits': entity_score, 'label':label_matrix, 'predicts': predicts}

        # PTM = self.PreTrained(data['input_ids'],attention_mask=data['attention_mask'],token_type_ids=data['token_type_ids'])
        # bert_encoder = PTM.last_hidden_state
        # logits = self.globalpointer(bert_encoder, mask = data['attention_mask'])
        # loss = self.criterion_argument(logits,data['argument_globalpointer_label'])
        #return {'loss': loss, 'logits':logits}
        return outputs

    def _forward(self, tokens: Dict[str, Any]) -> torch.Tensor:
        x = self.PreTrained(tokens['input_ids'],
                            attention_mask=tokens['attention_mask'],
                            token_type_ids=tokens['token_type_ids'],
                            table_pos_embeds=tokens['table_pos_embeds'])  #(**tokens)
        x = x.last_hidden_state
        mask = tokens['attention_mask']
        #mask = get_tokens_mask(tokens, x.size(1))

        if self.config.table_self_att:
            x = self.table_attention(x, x, x, attn_mask=tokens['table_self_att'])
            x = x[0]
        elif self.use_dropout:
            x = self.dropout(x)

        # if self.encoder is not None:
        #     x = self.encoder(x, mask)
        #     if self.use_dropout:
        #         x = self.dropout(x)

        token_inner_embed = self.token_inner_embed_ffn(x)
        start_token, end_token = token_inner_embed[..., ::2], token_inner_embed[..., 1::2]
        if self.config.custom_RoPE_pos:
            pos = self.pos_embed(token_inner_embed,custom_pos_ids=tokens['RoPE_pos_ids'])
        else:
            pos = self.pos_embed(token_inner_embed)
        cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)

        def add_position_embedding(in_embed, cos_pos, sin_pos):
            additional_part = torch.stack([-in_embed[..., 1::2], in_embed[..., ::2]], 3)
            additional_part = torch.reshape(additional_part, start_token.shape)
            output_embed = in_embed * cos_pos + additional_part * sin_pos
            return output_embed

        start_token = add_position_embedding(start_token, cos_pos, sin_pos)
        end_token = add_position_embedding(end_token, cos_pos, sin_pos)
        span_score = (
            torch.einsum('bmd,bnd->bmn', start_token, end_token) / self.inner_dim**0.5
        )
        typing_score = torch.einsum('bnh->bhn', self.type_score_ffn(x)) / 2
        entity_score = (
            span_score[:, None] + typing_score[:, ::2, None] + typing_score[:, 1::2, :, None]
        )  # [:, None] 增加一个维度

        entity_score = self._add_mask_tril(entity_score, mask=mask)
        return entity_score

    def _calculate_loss(self, entity_score, targets) -> torch.Tensor:
        """
        targets : (batch_size, num_classes, seq_len, seq_len)
        entity_score : (batch_size, num_classes, seq_len, seq_len)
        """
        print("=========")
        print("entity_score.size()",entity_score.size())
        print("targets.size()",targets.size())
        batch_size, num_classes = entity_score.shape[:2]
        targets = targets.reshape(batch_size * num_classes, -1)
        entity_score = entity_score.reshape(batch_size * num_classes, -1)
        loss = self.multilabel_categorical_crossentropy(targets, entity_score)
        return loss

    def _sequence_masking(self, x, mask, value='-inf', axis=None) -> torch.Tensor:
        """Mask X according to the mask."""

        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)
            #return x * mask + value * ~mask

    def _add_mask_tril(self, entity_score, mask):
        entity_score = self._sequence_masking(entity_score, mask, '-inf', entity_score.ndim - 2)
        entity_score = self._sequence_masking(entity_score, mask, '-inf', entity_score.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(entity_score), diagonal=-1)
        entity_score = entity_score - mask * 1e12
        return entity_score

    def multilabel_categorical_crossentropy(self, targets, entity_score):
        """Multi-label cross entropy loss.
        https://kexue.fm/archives/7359
        """
        entity_score = (1 - 2 * targets) * entity_score  # -1 -> pos classes, 1 -> neg classes
        entity_score_neg = entity_score - targets * 1e12  # mask the pred outputs of pos classes
        entity_score_pos = (
            entity_score - (1 - targets) * 1e12
        )  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(entity_score[..., :1])
        entity_score_neg = torch.cat([entity_score_neg, zeros], dim=-1)
        entity_score_pos = torch.cat([entity_score_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(entity_score_neg, dim=-1)
        pos_loss = torch.logsumexp(entity_score_pos, dim=-1)

        return (neg_loss + pos_loss).mean()

    def decode(self, entity_scores): #: torch.Tensor) -> List[List[TypedSpan]]:  # noqa
        entity_scores = entity_scores.detach().cpu().numpy()
        batch = list()
        for score_matrix in entity_scores:
            entities = [
                (start - 1, end - 1, self.id_to_label[type_id+1])  # * 均-1是为了减去开头的[CLS] ； type_id +1是因为id_to_label字典从1开始计数
                for type_id, start, end in zip(*np.where(score_matrix > 0))  # type: ignore
            ]
            batch.append(entities)
        return batch

class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)
    
    def get_evaluate_fpr(self, y_pred, y_true):
        X, Y, Z = 1e-10, 1e-10, 1e-10
        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))
        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        if Y==0 or Z==0:
            return 0,0,0
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall
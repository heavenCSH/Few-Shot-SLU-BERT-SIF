import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from TorchCRF import CRF
from utils.activations import ACT2FN
from torch.autograd import Variable
import sys


class Model(BertPreTrainedModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.bert = BertModel(config=config)
        self.cls = BertPreTrainingHeads(config)
        self.slot_classifier = nn.Linear(config.vocab_size, config.num_slot)
        self.num_slot = config.num_slot
        self.dropout = nn.Dropout(config.dropout)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # print('sequence_output_shape:', bert_outputs[0].shape) # 32 × 80 × 768
        # print('cls_output_shape:', bert_outputs[1].shape) # 32 × 768
        sequence_output, pooled_output = bert_outputs[:2] # bert输出
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output) # FFN后
        slot_logits = self.slot_classifier(prediction_scores) # bs × man_len × num_slot
        # print('after--sequence_output_shape:', prediction_scores.shape) # 32 × max_len × vocab_size
        # print('after--cls_output_shape:', seq_relationship_score.shape) # 32 × 2
        seq_relationship_score = seq_relationship_score[:,0] # 32
        # seq_relationship_score = torch.softmax(seq_relationship_score,dim=-1)[:,0]
        # print(seq_relationship_score)
        # import sys
        # sys.exit()
        # guide_intent = (torch.sigmoid(seq_relationship_score) > threshold).float()
        # guide_intent = guide_intent.view(1, -1)
        # slot_logits = torch.tensor([])
        # for i in range(self.num_slot):
        #     result = torch.matmul(guide_intent, prediction_scores[ : , : , i]).view(-1, 1)
        #     if len(slot_logits) == 0:
        #         slot_logits = result
        #     else:
        #         slot_logits = torch.cat((slot_logits, result), dim=1)
        # print(slot_logits)
        return seq_relationship_score, slot_logits
        # return intent_logits, slot_logits

    def get_mask_logits(self, prediction_scores, masked_index):
        mask_logits = []
        for batch_idx, index in enumerate(masked_index):
            single_logits = prediction_scores[batch_idx,index:index+1,:]
            mask_logits.append(single_logits)
        return torch.cat(mask_logits,dim=0)


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2) # 768 × 2

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias


    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class FocalLoss(nn.Module):
    def __init__(self, gamma=0., ignore_index=-100, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        num_label = input.size()[-1]

        mask = target.ne(self.ignore_index)
        input = torch.masked_select(input, mask.unsqueeze(-1).expand_as(input)).view(-1, num_label)
        target = torch.masked_select(target, mask)

        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
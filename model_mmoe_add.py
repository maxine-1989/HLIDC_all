import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel
from transformers import RobertaModel
from transformers import RobertaTokenizer, TFRobertaModel
import os
import logging

logger = logging.getLogger(__name__)


class WordAttNet(nn.Module):
    def __init__(self, bert_path, hidden_size, dropout, max_turn):
        super(WordAttNet, self).__init__()
        # self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path)
        self.bert = RobertaModel.from_pretrained(bert_path)
        """
        fine tune last 3 layer
        """
        activate_layer = [
            "bert.encoder.layer.9.attention.self.query.weight",
            "bert.encoder.layer.9.attention.self.query.bias",
            "bert.encoder.layer.9.attention.self.key.weight",
            "bert.encoder.layer.9.attention.self.key.bias",
            "bert.encoder.layer.9.attention.self.value.weight",
            "bert.encoder.layer.9.attention.self.value.bias",
            "bert.encoder.layer.9.attention.output.dense.weight",
            "bert.encoder.layer.9.attention.output.dense.bias",
            "bert.encoder.layer.9.attention.output.LayerNorm.weight",
            "bert.encoder.layer.9.attention.output.LayerNorm.bias",
            "bert.encoder.layer.9.intermediate.dense.weight",
            "bert.encoder.layer.9.intermediate.dense.bias",
            "bert.encoder.layer.9.output.dense.weight",
            "bert.encoder.layer.9.output.dense.bias",
            "bert.encoder.layer.9.output.LayerNorm.weight",
            "bert.encoder.layer.9.output.LayerNorm.bias",
            "bert.encoder.layer.10.attention.self.query.weight",
            "bert.encoder.layer.10.attention.self.query.bias",
            "bert.encoder.layer.10.attention.self.key.weight",
            "bert.encoder.layer.10.attention.self.key.bias",
            "bert.encoder.layer.10.attention.self.value.weight",
            "bert.encoder.layer.10.attention.self.value.bias",
            "bert.encoder.layer.10.attention.output.dense.weight",
            "bert.encoder.layer.10.attention.output.dense.bias",
            "bert.encoder.layer.10.attention.output.LayerNorm.weight",
            "bert.encoder.layer.10.attention.output.LayerNorm.bias",
            "bert.encoder.layer.10.intermediate.dense.weight",
            "bert.encoder.layer.10.intermediate.dense.bias",
            "bert.encoder.layer.10.output.dense.weight",
            "bert.encoder.layer.10.output.dense.bias",
            "bert.encoder.layer.10.output.LayerNorm.weight",
            "bert.encoder.layer.10.output.LayerNorm.bias",
            "bert.encoder.layer.11.attention.self.query.weight",
            "bert.encoder.layer.11.attention.self.query.bias",
            "bert.encoder.layer.11.attention.self.key.weight",
            "bert.encoder.layer.11.attention.self.key.bias",
            "bert.encoder.layer.11.attention.self.value.weight",
            "bert.encoder.layer.11.attention.self.value.bias",
            "bert.encoder.layer.11.attention.output.dense.weight",
            "bert.encoder.layer.11.attention.output.dense.bias",
            "bert.encoder.layer.11.attention.output.LayerNorm.weight",
            "bert.encoder.layer.11.attention.output.LayerNorm.bias",
            "bert.encoder.layer.11.intermediate.dense.weight",
            "bert.encoder.layer.11.intermediate.dense.bias",
            "bert.encoder.layer.11.output.dense.weight",
            "bert.encoder.layer.11.output.dense.bias",
            "bert.encoder.layer.11.output.LayerNorm.weight",
            "bert.encoder.layer.11.output.LayerNorm.bias",
            "bert.pooler.dense.weight",
            "bert.pooler.dense.bias",
        ]
        for name, p in self.named_parameters():
            if name not in activate_layer:
                p.requires_grad = False

        self.gru = nn.GRU(
            # input_size=768,
            input_size=1024,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
            num_layers=1,
        )
        self.dropout = nn.Dropout(dropout)
        # Attention
        self.word_weight_1 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True)
        self.word_weight_2 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True)
        self.context_weight_2 = nn.Linear(2 * hidden_size, 1, bias=False)
        self.context_weight_1 = nn.Linear(2 * hidden_size, 1, bias=False)
        self.max_turns = max_turn
        self.MOE = MOENet(max_turn=max_turn)

        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean, std):
        nn.init.normal_(self.word_weight_1.weight, mean=mean, std=std)
        nn.init.normal_(self.word_weight_2.weight, mean=mean, std=std)
        nn.init.normal_(self.context_weight_1.weight, mean=mean, std=std)
        nn.init.normal_(self.context_weight_2.weight, mean=mean, std=std)

    def forward(self, input_ids, attention_mask, token_type):
        # input_ids (B, T) -> (B, T, E)
        # with torch.no_grad():
        # output, _ = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type)
        output = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type
        )
        # print("word_att forward 107 output size:", output.last_hidden_state.size())

        # 这里提取了BERT模型的最后隐藏状态（hidden states），这是一个三维张量，形状通常为 (batch_size, sequence_length, hidden_size)。
        output = output.last_hidden_state
        # BERT的输出中，第一个位置通常是[CLS]标记，表示整个序列的聚合表示。在这里，通过切片操作去掉了这个标记，保留从第二个位置开始的所有输出。
        output = output[:, 1:, :]  # no [CLS]
        output = self.dropout(output)
        # print("word_att forward 116(after dropout) output size:",output.size())


        lengths = [mask.tolist().count(1) - 1 for mask in attention_mask]

        # print("lengths",lengths)

        # output = pack_padded_sequence(output, lengths, batch_first=True, enforce_sorted=False)
        # (B, T, H*num_directions), (num_layers * num_directions, B, H)
        # print(output)
        # 问题出现在这里
        output, _ = self.gru(output)
        # print("word_att forward 128(after gru) output size:",output.size())


        # output, lengths = pad_packed_sequence(output, batch_first=True)

        # Attention
        # (B, T, H) * (H, H) -> (B, T, H)
        output_1 = torch.tanh(self.word_weight_1(output))
        output_2 = torch.tanh(self.word_weight_2(output))
        weights_1 = self.context_weight_1(output_1)
        weights_2 = self.context_weight_2(output_2)

        # print("word_att_net attention part size:output_1",output_1.size())
        # print("word_att_net attention part size:output_2",output_2.size())
        # print("word_att_net attention part size:weights_1",weights_1.size())
        # print("word_att_net attention part size:weights_2",weights_2.size())

        # mask
        mask = torch.zeros(weights_1.size())
        for index, l in enumerate(lengths):
            mask[index, l:, :] = 1
        mask = mask.bool().to(weights_1.device)

        # masked_fill 是一个张量操作，用于根据给定的掩码来填充张量的特定位置。
        # mask 是一个布尔张量，与 weights_1 的形状相同。其值为 True 的位置表示需要被填充的元素。
        weights_1 = weights_1.masked_fill(mask, -1e9)
        weights_1 = F.softmax(weights_1, dim=1).squeeze(-1)  # (B, T, 1)

        weights_2 = weights_2.masked_fill(mask, -1e9)
        weights_2 = F.softmax(weights_2, dim=1).squeeze(-1)  # (B, T, 1)

        weights_1, weights_2 = self.MOE(weights_1, weights_2)

        output_1 = torch.bmm(output.permute(0, 2, 1), weights_1.unsqueeze(-1)).squeeze(
            -1
        )
        output_2 = torch.bmm(output.permute(0, 2, 1), weights_2.unsqueeze(-1)).squeeze(
            -1
        )

        # print("size of the output_1 of wordattnet :",output_1.size())
        # print("size of the output_2 of wordattnet :",output_2.size())
        # print("size of the weights_2.squeeze(-1) of wordattnet :",weights_2.squeeze(-1).size())
        

        return output_1, weights_2.squeeze(-1), output_2


class SentAttNet(nn.Module):
    def __init__(
        self,
        sent_hidden_size,
        word_hidden_size,
        num_classes_1,
        num_classes_2,
        max_turn,
        coefficient,
    ):
        super(SentAttNet, self).__init__()

        # print("num_classes_1:", num_classes_1)
        # print("num_classes_2:", num_classes_2)

        self.coefficient = coefficient
        self.max_turns = max_turn
        self.MOE = MOENet(max_turn=max_turn)
        self.sent_weight_1 = nn.Linear(
            2 * sent_hidden_size, 2 * sent_hidden_size, bias=True
        )
        self.sent_weight_2 = nn.Linear(
            2 * sent_hidden_size, 2 * sent_hidden_size, bias=True
        )
        self.context_weight_1 = nn.Linear(2 * sent_hidden_size, 1)
        self.context_weight_2 = nn.Linear(2 * sent_hidden_size, 1)

        # matrix = [
        #     [
        #         1,
        #         0,
        #         1,
        #         1,
        #         1,
        #         0,
        #         1,
        #         0,
        #         1,
        #         0,
        #         1,
        #         1,
        #         0,
        #         0,
        #         0,
        #         0,
        #         1,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         1,
        #         1,
        #         1,
        #         0,
        #         1,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         1,
        #     ],
        #     [
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         1,
        #         0,
        #         0,
        #         0,
        #         1,
        #         0,
        #         0,
        #         1,
        #         1,
        #         1,
        #         0,
        #         0,
        #         1,
        #         1,
        #         1,
        #         0,
        #         1,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #     ],
        #     [
        #         0,
        #         1,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         1,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         0,
        #         1,
        #         0,
        #         0,
        #         0,
        #         0,
        #         1,
        #         0,
        #         1,
        #         0,
        #         0,
        #         0,
        #         1,
        #         0,
        #         1,
        #         1,
        #         1,
        #         1,
        #         1,
        #         0,
        #     ],
        # ]

        # self.matrix = torch.tensor(matrix, requires_grad=False, dtype=torch.long).to(
        #     torch.float
        # )
        # if torch.cuda.is_available():
        #     self.matrix = self.matrix.cuda()

        self.gru = nn.GRU(
            input_size=2 * word_hidden_size,
            hidden_size=sent_hidden_size,
            bidirectional=True,
            batch_first=True,
            num_layers=1,
        )

        self.fc_1 = nn.Linear(2 * sent_hidden_size, num_classes_1)
        self.fc_2 = nn.Linear(2 * sent_hidden_size, num_classes_2)
        # print("2* set_hiden_size:", 2 * sent_hidden_size)

        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        nn.init.normal_(self.sent_weight_1.weight, mean=mean, std=std)
        nn.init.normal_(self.context_weight_1.weight, mean=mean, std=std)
        nn.init.normal_(self.sent_weight_2.weight, mean=mean, std=std)
        nn.init.normal_(self.context_weight_2.weight, mean=mean, std=std)

    def forward(self, inputs_1,inputs_2, matrix):
        # print("=======================sent_att_net===========================")
        # print("size of sentattent inputs_2:",inputs_2.shape)
        # 对于新数据集进行的改动
        if inputs_2 is None:  # 如果只有一个句子
            inputs_2 = inputs_1  # 将 inputs_2 设置为 inputs_1

        # inputs (B, Turns, Hidden)
        output_1, _ = self.gru(inputs_1)
        output_2, _ = self.gru(inputs_2.float())
        # print(" size of sentattent output_1 in 350(after gru):",output_1.shape)
        # print(" size of sentattent output_2 in 350(after gru):",output_2.shape)


        # Attention
        output_1 = torch.tanh(self.sent_weight_1(output_1))
        output_2 = torch.tanh(self.sent_weight_2(output_2))
        # print(" size of sentattent output_1 in 355(after attention):",output_1.shape)
        # print(" size of sentattent output_2 in 356(after attention):",output_2.shape)
        # context_weight =B,turns,Hidden *  2Hidden,1 = B,T,1
        weights_1 = self.context_weight_1(output_1)
        weights_2 = self.context_weight_2(output_2)

        # print("weights_1.shape  :",weights_1.shape)
        # print("weights_2.shape :",weights_2.shape)

        # 话语数量相同，不用mask
        # weights_1 = F.softmax(weights_1, dim=1).squeeze(-1)  # (B,T,H)  (B, T, 1)
        # weights_2 = F.softmax(weights_2, dim=1).squeeze(-1)
        weights_1 = F.softmax(weights_1, dim=1)  # (B
        
        t = weights_1.size(1)
        weights_1 = F.pad(weights_1, (0, self.max_turns - t), mode="constant", value=0)
        weights_2 = F.pad(weights_2, (0, self.max_turns - t), mode="constant", value=0)
        # print("sent_att_net attention part size:weights_1",weights_1.size())
        # print("sent_att_net attention part size:weights_2",weights_2.size())

        weights_1, weights_2 = self.MOE(weights_1, weights_2)
        # print("sent_att_net weights_1 size after MOE:",weights_1.size())
        # print("sent_att_net weights_2 size after MOE:",weights_2.size())



        weights_1 = weights_1[:, :t]  # (B,T,H)  (B, T, 1)
        weights_2 = weights_2[:, :t]

        # permute(0,2,1)->(B,H,T)*(B,T,1)=(B,H,1)
        # output_1 = torch.bmm(
        #     output_1.permute(0, 2, 1), weights_1.unsqueeze(-1)
        # ).squeeze(-1)
        # output_2 = torch.bmm(
        #     output_2.permute(0, 2, 1), weights_2.unsqueeze(-1)
        # ).squeeze(-1)
        # 对于 output_1 和 output_2 使用相同的计算
        output_1 = torch.bmm(output_1.unsqueeze(-1), weights_1.unsqueeze(-1)).squeeze(
            -1
        )  # (B, H)
        output_2 = torch.bmm(output_2.unsqueeze(-1), weights_2.unsqueeze(-1)).squeeze(
            -1
        )  # (B, H)
        logits_1 = self.fc_1(output_1)

        self.matrix = torch.tensor(matrix, requires_grad=False, dtype=torch.long).to(
            torch.float
        )
        if torch.cuda.is_available():
            self.matrix = self.matrix.cuda()

        weights = torch.matmul(logits_1, self.matrix).detach()

        # print("weights shape:", weights.shape)
        # logits_2 = self.fc_2(output_2) + self.coefficient * weights
        logits_2 = self.fc_2(output_2)

        # print("size of logits_1 before sentattnet's output:",logits_1.size())
        # print("size of logits_2 before sentattnet's output:",logits_2.size())
        # print("size of weights_2.squeeze(-1) before sentattnet's output:",weights_2.squeeze(-1).size())

        return logits_1, logits_2, weights_2.squeeze(-1)


class MOENet(nn.Module):
    """
    input:注意力跟句子/word相乘后得到的特征(p = ah)B,H
    """

    def __init__(self, max_turn):
        super(MOENet, self).__init__()

        self.gate_1 = nn.Linear(max_turn, 1, bias=True)
        self.gate_2 = nn.Linear(max_turn, 1, bias=True)

        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        nn.init.normal_(self.gate_1.weight, mean=mean, std=std)
        nn.init.normal_(self.gate_2.weight, mean=mean, std=std)

    def forward(self, inputs_1, inputs_2):
        # inputs (B, Hidden)->(B,2*H)
        # gate: (B,2,H)(H,1)->(B,2,1)->(B,1,2)
        # print(inputs_1)
        # input-> B,2,H
        # print("shapes of the inputs of MOE:")
        # print("inputs_1.shape:",inputs_1.shape)  # 打印 weights_1 的形状
        # print("inputs_2.shape:",inputs_2.shape)  # 打印 weights_2 的形状

        input = torch.cat(
            [torch.unsqueeze(inputs_1, 1), torch.unsqueeze(inputs_2, 1)], dim=1
        )
        # 

        gate_1 = self.gate_1(input)
        gate_1 = F.softmax(gate_1, dim=1).permute(0, 2, 1)
        gate_2 = self.gate_2(input)
        gate_2 = F.softmax(gate_2, dim=1).permute(0, 2, 1)

        # task_fea = (B,1,2)(B,2,H)->(B,1,H)
        # torch.bmm 是用于执行批量矩阵乘法的函数。
        # 这行代码的主要功能是通过批量矩阵乘法将 gate_1 和 input 进行结合，
        output_1 = torch.bmm(gate_1, input).squeeze(1)
        output_2 = torch.bmm(gate_2, input).squeeze(1)
        # print(output_1)

        return output_1, output_2


class HAN(nn.Module):
    def __init__(
        self,
        bert_path,
        hidden_size,
        dropout,
        num_classes_1,
        num_classes_2,
        max_seq,
        max_turn,
        coefficient,
    ):
        super(HAN, self).__init__()
        # test1：去掉这个word_att_net
        self.word_att_net = WordAttNet(
            bert_path=bert_path,
            hidden_size=hidden_size,
            dropout=dropout,
            max_turn=max_seq - 1,
        )
        # 似乎应该是sentence level，但是文章中似乎只有一个pair level，可能还是后者吧。只是名字取得有点歧义
        self.sent_att_net = SentAttNet(
            sent_hidden_size=hidden_size,
            word_hidden_size=hidden_size,
            num_classes_1=num_classes_1,
            num_classes_2=num_classes_2,
            coefficient=coefficient,
            max_turn=max_turn,
        )
        self.word_hidden_size = hidden_size

    def forward(self, input_ids, attention_mask, token_type,matrix):
        # input_ids (B, Turns, seq_len)
        # attention_mask (B, Turns, seq_len)

        # batch_size, max_turns, _ = input_ids.size()
        # output_list = []
        # permute：转置，参数是维度序号
        # input_ids = input_ids.permute(1, 0, 2)
        # attention_mask = attention_mask.permute(1, 0, 2)
        # word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size).to(input_ids.device)

        # for inputs, mask in zip(input_ids, attention_mask):
        #     output, word_hidden_state = self.word_att_net(inputs, mask, word_hidden_state)
        #     output_list.append(output)
        # output = torch.cat(output_list, 0)
        # output = output.view(batch_size, max_turns, -1)
        # logits, sent_weights = self.sent_att_net(output)

        # --------------------------------------

        # input_ids (B, Turns, seq_len)
        # batch, turns, seq_len = input_ids.size()
        # new_data set
        batch, seq_len = input_ids.size()

        input_ids = input_ids.view(-1, seq_len)

        attention_mask = attention_mask.view(-1, seq_len)

        token_type = token_type.view(-1, seq_len)

        # print(input_ids.size(-1))
        output_1, word_weights_2, output_2 = self.word_att_net(
            input_ids, attention_mask, token_type
        )
        # output_1 = output_1.view(batch, turns, -1)
        # output_2 = output_2.view(batch, turns, -1)
        output_1 = output_1.view(batch, -1)
        output_2 = output_2.view(batch, -1)
        # print(output_2.shape)

        logits_1, logits_2, sent_weights_2 = self.sent_att_net(output_1, output_2,matrix)

        return (
            logits_1,
            logits_2,
            # word_weights_2.view(batch, turns, seq_len - 1),
            word_weights_2.view(batch, seq_len - 1),
            sent_weights_2,
        )

# -*- coding: utf-8 -*-
"""
数据加载工具类
"""
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

# from nlupp.data_loader import DataLoader_
from transformers import BertTokenizer
from transformers import RobertaTokenizer
import re
import os
import pickle
import pandas as pd
import tqdm
import math
import json
import ast


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.input_ids = dataset[0]
        self.attn_mask = dataset[1]
        self.token_type = dataset[2]
        self.labels = dataset[3]
        self.labels_thr = dataset[4]
        self.matrices = dataset[5]
        # self.turns = dataset[2]
        # self.segment = dataset[1]
        # self.length = dataset[4]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # input_ids, attn_mask, token_type, labels
        return (

            self.input_ids[index].clone().detach().long(),
            self.attn_mask[index].clone().detach().long(),
            self.token_type[index].clone().detach().long(),
            self.labels[index].clone().detach().long(),
            self.labels_thr[index].clone().detach().long(),
            self.matrices[index].clone().detach().long(),
        )

'''
class MyBatchSampler(Sampler):
    """
    按照轮次数量，从多到少顺序选取，数据要先排序
    """

    def __init__(self, batch_size, turns, drop_last=False):
        self.batch_size = batch_size
        self.turns = turns  # 每条数据有多少轮，多少句对话
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        curren_turns = self.turns[0]
        for idx in range(len(self.turns)):
            if self.turns[idx] == curren_turns and len(batch) < self.batch_size:
                batch.append(idx)
            else:
                curren_turns = self.turns[idx]
                yield batch
                batch = [idx]
            # if len(batch) == self.batch_size:
            #     yield batch
            #     batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
            
    def __len__(self):
        if self.drop_last:
            return len(self.turns) // self.batch_size
        else:
            return (len(self.turns) + self.batch_size - 1) // self.batch_size
'''


class MyBatchSampler(Sampler):
    """
    按照数据索引分批，不考虑对话轮次。
    """

    def __init__(self, batch_size, data_indices, drop_last=False):
        self.batch_size = batch_size
        self.data_indices = data_indices  # 数据的索引列表
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.data_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_indices) // self.batch_size
        else:
            return (len(self.data_indices) + self.batch_size - 1) // self.batch_size


class Utils(object):
    def __init__(
        self, bert_path, max_seq_len, max_turns, batch_size, data_folder, role
    ):
        self.max_seq_len = max_seq_len
        self.max_turns = max_turns
        self.batch_size = batch_size
        self.folder = data_folder

        # self.role = role  # complaint consult handel
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_path)
        # self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        # 疑问词
        self.question_words = [
            # 什么人？
            "哪个",
            "谁",
            # 什么原因？
            "为什么",
            "请问",
            "问一下",
            "咋没有",
            "为啥",
            # 什么时间？
            "多久啊",
            "什么时候",
            "啥时候",
            "几月几号",
            # 什么方式？
            "怎么",
            "怎么办呢",
            "咋查啊",
            "咋整",
            # 什么地方？
            ".*?哪.*?呀",
            "哪里",
            # 什么概念？
            "是什么",
            "什么",
            "是啥",
            "啥呀",
            "咋回事",
            "有啥",
            # 什么数量？
            "多少",
            "哪些",
            "多长",
            # 是非判断？，
            # 修改：删掉'呢'；.*?可以.*?可以.*? -> 可以.*?不可以；删掉'可以吧'
            "吗$",
            "的吗",
            "有没有",
            "是不是",
            "吗不是",
            "是不是",
            "对不对",
            "可以.*?不可以",
            "需不需",
            "是不是哦",
            "是吧",
            "要不",
            ".+好吧$",
            ".*?是.*?还是.*?",
            "能否",
            "可不可以",
            "对吧",
            "好不好",
            "能.*?不能",
            "想不想",
            "行不行",
            # 其他
            # 修改：删掉：'的呢'
            "想问",
            "好嘛",
            "哪去了",
            "是吗",
            "几.*?几.*?",
            "可以吗",
            "啥意思",
            "多久",
            "(吗女士)|(吗先生)$",
        ]
        # with open('../data/{}/tag.json'.format(self.role), 'r', encoding='utf-8') as f:
        #     self.label2id = json.load(f)

    def process(self, dialog):
        result = []
        for sentence in dialog.strip().split("[SEP]"):

            # sentence = re.sub(r'[呃啊呢哎诶]*', '', sentence)
            # sentence = re.sub(r'[一二三四五六七八九十]+月[一二三四五六七八九十]+号', '一月一号', sentence)
            # sentence = re.sub(r'[一二两三四五六七八九十]+元', 'NUM元', sentence)
            # sentence = re.sub(r'[一二两三四五六七八九十]+块钱', 'NUM元', sentence)
            # sentence = re.sub(r'[一二两三四五六七八九十百]+兆', 'NUM兆', sentence)
            # sentence = re.sub(r'[一二两三四五六七八九十百]+分钟', 'NUM分钟', sentence)
            # sentence = re.sub(r'[一二两三四五六七八九十]+块[一二两三四五六七八九]*毛?[一二三四五六七八九]*', '费用', sentence)
            # sentence = re.sub(r'[一二两三四五六七八九十]个g', '流量', sentence)
            # sentence = re.sub(r'[零幺一二两三四五六七八九十百千]{2,}', 'NUM', sentence)兆
            sentence = re.sub(
                r"[一二三四五六七八九十]+月[一二三四五六七八九十]+号",  # 正则表达式pattern
                "一月一号",  # 替换后的内容
                sentence,  # 目标字符串
            )  # 就等于说把所有的日期都换成一月一号
            sentence = re.sub(r"[一二两三四五六七八九十]+元", "NUM元", sentence)
            sentence = re.sub(r"[一二两三四五六七八九十]+块钱", "NUM元", sentence)

            if len(sentence) > 0:
                result.append(sentence)
            else:
                result.append("空")

        return result

    def is_interrogative_sentence(self, sentence):
        """
        疑问句识别
        :param sentence:
        :return:
        """
        for key in self.question_words:
            match = re.search(pattern=key, string=sentence)
            if match:
                return True
        return False

    def generate_hierarchical_matrix(self,num_classes_1, num_classes_2, num_cnum, c_numerical):
        # 初始化一个零矩阵
        matrix = torch.zeros((num_classes_1, num_classes_2), dtype=torch.float)

        # 确保 num_cnum 是列表形式
        # if isinstance(num_cnum, int):
        #     num_cnum = [num_cnum]  # 如果 num_cnum 是一个单一的整数，转换为列表
        
        for second_layer_tag in c_numerical:
            second_layer_tag = min(second_layer_tag, num_classes_2 - 1)
            matrix[num_cnum][second_layer_tag] = 1
    
        return matrix

    '''
    def read_data(self, data_type, sort=True):
        """
        读取数据+形成相邻对
        :return: token_data, label
        input_ids, attn_mask, token_type, labels, turns, processed_data, labels_thr
        """
        path = self.folder + "/all/{}.csv".format(data_type)  # train/dev/test
        print("Loading {}...".format(path))
        df = pd.read_csv(path, encoding="utf-8")
        data = []
        turns = []
        labels = []
        labels_thr = []
        # lengths = []

        for dialog, label, label_thr in zip(
            list(df["sentence_sep"]), list(df["c_numerical"]), list(df["num_cnum"])
        ):
            # lengths.append(len(dialog.strip().replace('[SEP]', '')))
            # turns.append(len(dialog.strip().split('[SEP]')))
            dialog = self.process(dialog)
            # 每两句话当做一个回合，当回合的第二句话是疑问句时，复制一次
            # 回合内的两句话合并输入
            # 注意复制疑问句后会导致后面的句子位置改变
            temp = []
            index = 1
            for sentence in dialog:
                temp.append(sentence)
                # 复制一次
            if index % 2 == 0 and self.is_interrogative_sentence(sentence):
                index += 1
                temp.append(sentence)
            index += 1

            sample = []
            # 大概是两两一组的意思,然后每一组用
            # for i in range(0, len(temp), 2):
            #     sample.append("[SEP]".join(temp[i : i + 2]))
            data.append(sample)
            turns.append(len(sample))
            labels.append(int(label))
            labels_thr.append(int(label_thr))

        data = zip(data, labels, turns, labels_thr)
        # 根据话语数排序，升序
        if sort:
            # key 参数用于指定排序的依据。lambda x: x[2] 是一个匿名函数，表示使用每个元素 x 的第三个值（x[2]）来进行排序。
            data = sorted(data, key=lambda x: x[2], reverse=True)
        else:
            data = [(x[0], x[1], x[2], x[3]) for x in data]
        processed_data = [x[0] for x in data]
        turns = [x[2] if x[2] < self.max_turns else self.max_turns for x in data]
        # lengths = [x[3] for x in data]

        input_ids = []
        attn_mask = []
        token_type = []
        labels = []
        labels_thr = []

        for dialog, l, _, l3 in tqdm.tqdm(data):
            labels.append(l)
            labels_thr.append(l3)

            temp_dialog = []
            temp_mask = []
            temp_token_type = []
            # temp_length = []  # 一个对话每句话的长度
            # i = 1
            for utterance in dialog:
                # print(utt)
                sent = utterance.split("[SEP]")
                if len(sent) == 1:
                    sent.append(" ")
                # result = self.tokenizer.encode_plus(text=sent[0],
                #                                     text_pair=sent[1],
                #                                     add_special_tokens=True,
                #                                     return_token_type_ids=True,
                #                                     return_attention_mask=True,
                #                                     max_length=self.max_seq_len,
                #                                     pad_to_max_length=True)
                # text_cat = str(i) +'[SEP]'+ sent[0]
                # 用了 self.tokenizer.encode_plus，这是来自 Hugging Face 的 transformers 库中的一个方法，通常用于对文本进行编码，生成模型所需的输入格式。这是为将两个句子（sent[0] 和 sent[1]）作为输入，经过分词器编码后，转换为可以输入 BERT 或其他 transformer 模型的格式。
                # encode_plus 方法会返回一个字典，其中包含以下几个键：
                # input_ids：编码后的 token ID 列表。
                # token_type_ids：每个 token 所属的句子 ID（第一个句子 ID 是 0，第二个句子 ID 是 1）。
                # attention_mask：与 input_ids 对应的掩码，标记哪些是实际的 tokens（1）以及哪些是 padding（0）。
                result = self.tokenizer.encode_plus(
                    text=sent[0],  # 主文本（句子1）
                    text_pair=sent[1],  # 与主文本配对的句子（句子2）
                    add_special_tokens=True,  # 是否添加特殊标记，比如 [CLS], [SEP] 等
                    return_token_type_ids=True,  # 是否返回 token 类型 ID（用于区分两个句子）
                    return_attention_mask=True,  # 是否返回注意力掩码，用于掩盖 padding 的位置
                    max_length=self.max_seq_len,
                    truncation=True,
                    padding="max_length",
                )

                temp_dialog.append(result["input_ids"])
                temp_mask.append(result["attention_mask"])
                temp_token_type.append(result["token_type_ids"])
                i += 1

            input_ids.append(temp_dialog[: self.max_turns])
            attn_mask.append(temp_mask[: self.max_turns])
            token_type.append(temp_token_type[: self.max_turns])

        return (
            input_ids,
            attn_mask,
            token_type,
            labels,
            turns,
            processed_data,
            labels_thr,
        )
    '''

    def read_data(self, data_type, sort=True):
        """
        读取数据并处理单句话。
        :return: token_data, label
        input_ids, attn_mask, token_type, labels,processed_data, labels_thr
        """
        path = self.folder + "/all/{}.csv".format(data_type)  # train/dev/test
        # path = self.floder+
        print("Loading {}...".format(path))
        df = pd.read_csv(path, encoding="utf-8")

        data = []
        labels = []
        labels_thr = []
        matrices = []  # 用于存储每个样本的矩阵
        all_second_layer_tags = set() # 存储第二层的标签

        df["c_numerical"] = df["c_numerical"].apply(ast.literal_eval)
        if df["c_numerical"].empty or df["num_cnum"].empty:
            print("Error: No data in labels columns")

        for sentence, label, label_thr in zip(
            list(df["text"]), list(df["c_numerical"]), list(df["num_cnum"])
        ):
            # 处理单句话
            sentence = self.process(sentence)

            data.append(sentence)
            labels_thr.append(int(label_thr))

            # 生成第二层标签矩阵
            second_layer_label = [0] * 62  # 假设有 62 个标签，初始化为 0
            for lbl in label:  # 遍历每个标签，将对应位置设为 1
                all_second_layer_tags.add(lbl)

                if lbl-1 < 62:  # 确保标签在有效范围内
                    second_layer_label[lbl-1] = 1


            # 动态生成层级矩阵
            matrix = self.generate_hierarchical_matrix(3,61,label_thr,label)
            matrices.append(matrix)

            # print("second_layer_label:",second_layer_label)
            # 将填充后的第二层标签添加到 labels 列表
            labels.append(second_layer_label)



        # all_second_layer_tags = set(labels)
        unique_second_layer_tags = sorted(all_second_layer_tags)
        # print("unique_second_layer_tags",unique_second_layer_tags)
        tag2index = {tag:idx for idx ,tag in enumerate(unique_second_layer_tags)}
        index2tag = {idx: tag for tag,idx in tag2index.items()}

        # print("tag2index:",tag2index)/
        # print("index2tag:",index2tag)

        
        # tag_dict ={"tag2index":tag2index,"index2tag":index2tag}
        # print("tag_dict:",tag_dict)
        # with open("/data/alyssa/HLIDC_1/nlupp/all/tag_dict.pkl","wb") as f:
        #     pickle.dump(tag_dict,f)
        # print("Tag dictionary saved to tag_dict.pkl.")

        # labels = [[tag2index[lbl] for lbl in label if lbl in tag2index] for label in labels]

        # 将数据按标签排序（可选）
        data = list(zip(data, labels, labels_thr))  # 转换为列表
        if sort:
            data = sorted(data, key=lambda x: len(x[0]), reverse=True)  # 按句子长度降序

        processed_data = [x[0] for x in data]
        data_indices = list(range(len(processed_data))) 

        input_ids = []
        attn_mask = []
        token_type = []

        for sentence in tqdm.tqdm(processed_data):
            result = self.tokenizer.encode_plus(
                text=sentence,  # 主文本
                add_special_tokens=True,  # 添加特殊标记
                return_token_type_ids=True,  # 返回 token 类型 ID
                return_attention_mask=True,  # 返回注意力掩码
                max_length=self.max_seq_len,  # 最大长度
                truncation=True,  # 截断
                padding="max_length",  # 填充到最大长度
            )

            input_ids.append(result["input_ids"])
            attn_mask.append(result["attention_mask"])
            token_type.append(result["token_type_ids"])


        input_ids = torch.tensor(input_ids)
        attn_mask = torch.tensor(attn_mask)
        token_type = torch.tensor(token_type)
        labels = torch.tensor(labels)
        labels_thr = torch.tensor(labels_thr)
        matrices = torch.stack(matrices)


        return (
            input_ids,
            attn_mask,
            token_type,
            labels,
            data_indices,
            processed_data,
            labels_thr,
            matrices,
        )

    def data_loader(self, data_type, sort=True):
        # input_ids, attn_mask, token_type, labels, turns, processed_data, labels_thr = (
        #     self.read_data(data_type, sort=sort)
        # )
        (
            input_ids,
            attn_mask,
            token_type,
            labels,
            data_indices,
            processed_data,
            labels_thr,
            matrices,
        ) = self.read_data(data_type, sort=sort)

        # 打印各个张量的形状，以确保它们一致

        # print(f"input_ids shape: {input_ids.shape}")
        # print(f"attn_mask shape: {attn_mask.shape}")
        # print(f"token_type shape: {token_type.shape}")
        # print(f"labels shape: {labels.shape}")
        # print(f"labels_thr shape: {labels_thr.shape}")
        # print(f"matrices shape: {matrices.shape}")

        dataset = MyDataset((input_ids, attn_mask, token_type, labels, labels_thr, matrices))
        if sort:
            loader = DataLoader(
                dataset=dataset,
                batch_sampler=MyBatchSampler(
                    batch_size=self.batch_size, data_indices=data_indices
                ),
            )
        else:
            loader = DataLoader(
                dataset=dataset, batch_size=1, drop_last=False, shuffle=False
            )

        # print(loader.matrices)
        return loader


if __name__ == "__main__":

    bert_path = r"/data/public_model/roberta-large/"
    utils = Utils(
        bert_path=bert_path,
        max_seq_len=10,
        max_turns=10,
        batch_size=2,
        data_folder=r"/data/alyssa/HLIDC_1/nlupp/all",
        role=0,
    )
    # loader = DataLoader_("<PATH_TO_NLUPP_DATA>")
    loader = utils.data_loader("dev_new")
    # print(len(loader))
    count = 0
    for batch in loader:
        # print(batch[0].size())
        # print(batch[3].size())
        # print(batch[0].size())
        # print(batch[1])
        # print(batch[2])
        # count += 1
        # if count == 2000:
        #     exit()
        break
    pass

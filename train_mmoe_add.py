# -*- coding: utf-8 -*-
"""

"""
import os
import time
import logging
import argparse
import platform
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
import transformers
from model_mmoe_add import HAN
import data_utils_two_layer as data_utils
from sklearn.metrics import confusion_matrix

# import wandb
import numpy as np
import random
from focal_loss import MultiFocalLoss
from focal_loss import MultiLabelFocalLoss



# torch.cuda.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')


def train(args):
    args.model_name = args.model_name.format(args.role)

    # 日志文件夹和模型保存文件夹若不存在则创建
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    # 日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] ## %(message)s")
    timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
    # 输出到文件
    file_handler = logging.FileHandler(
        args.log_path + "{}_".format(args.model_name).split(".")[0] + timestamp + ".txt"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    if os.path.exists(args.log_path):
        print("")
        print("✅ 日志文件成功创建！")
    else:
        print("❌ 日志文件未创建！")

    # logger.info("测试日志写入...")
    # logger.warning("这是一个警告日志")
    # logger.error("这是一个错误日志")
    # 输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(args)

    # 数据加载
    utils = data_utils.Utils(
        bert_path=args.bert_path,
        max_seq_len=args.max_seq_len,
        max_turns=args.max_turns,
        batch_size=args.batch_size,
        data_folder=args.data_folder,
        role=args.role,
    )
    train_loader = utils.data_loader("train_new")
    dev_loader = utils.data_loader("dev_new", sort=False)
    test_loader = utils.data_loader("test_new", sort=False)

    model = HAN(
        bert_path=args.bert_path,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        num_classes_1=args.num_classes_1,
        num_classes_2=args.num_classes_2,
        max_turn=args.max_turns,
        max_seq=args.max_seq_len,
        coefficient=args.coefficient,

    ).to(device)
    # model.load_state_dict(torch.load(r'./model/DCR-Net.bin'))
    # 优化器与损失函数
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-8)

    # 不对bias和LayerNorm.weight做L2正则化
    no_decay = ["bias", "LayerNorm.weight"]

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    # PyTorch scheduler
    # 定义训练步数
    total_steps = len(train_loader) * args.x

    # 定义warmup步数
    warmup_steps = int(total_steps * 0.1)

    # 定义学习率调整器
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=args.epochs)
    # criterion = nn.CrossEntropyLoss()

    # 这段代码定义了一个损失函数 criterion_1，使用了 MultiFocalLoss，并为它传递了几个参数。MultiFocalLoss 是一种变体的交叉熵损失函数，称为 Focal Loss，常用于处理类别不平衡问题。
    criterion_1 = MultiFocalLoss(
        num_class=args.num_classes_1, gamma=1.0, reduction="mean"
    )
    criterion_2 = MultiLabelFocalLoss(
        num_class=args.num_classes_2, gamma=5.0, reduction="mean"
    )

    best_score = 0
    patience = 0
    training_loss = 0
    # loss_1_last = 1
    # loss_2_last = 1
    step = 0
    for epoch in range(args.epochs):
        logger = logging.getLogger(__name__)  # 确保和之前定义的 logger 一致
        logger.info(
            10 * "*"
            + "training epoch: {} / {}".format(epoch + 1, args.epochs)
            + "*" * 10
        )
        # train mode
        model.train()
        batch_index = 1
        hits, hits1, totals = 0, 0, 0
        for batch in tqdm(train_loader):

            batch = tuple(t.to(device) for t in batch)

            input_ids, attn_mask, token_type, labels, labels_thr, matrices = batch

            # print("input_ids.size(-1):",input_ids.size(-1))

            logits_1, logits_2, _, _ = model(input_ids, attn_mask, token_type, matrices)
            # print("---------------- in train.py-------------------")
            # print("logits_1 size:",logits_1.size())
            # print("logits_2 size:",logits_2.size())/


            # loss
            loss_first_layer = criterion_1(logits_1, labels_thr)
            loss_second_layer = criterion_2(logits_2, labels)

            loss = args.lamb * loss_first_layer + loss_second_layer

            training_loss += loss.item()

            _, predict1 = torch.max(logits_1, 1)
            hits1 += (predict1 == labels_thr).sum().item()

            # _, predict = torch.max(logits_2, 1)
            # hits += (predict == labels).sum().item()
            predict = (torch.sigmoid(logits_2) > 0.4).int()
            labels = labels.int()
            hits +=(predict == labels).sum().item()
            totals += labels.numel()

            # totals += labels.size(0)
            # wandb.log({"Train Loss": loss,"epoch":epoch })

            loss = loss / args.accumulation_steps
            # backward
            loss.backward()

            # 梯度累加
            if (step + 1) % args.accumulation_steps == 0:
                # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # log
            if (step + 1) % args.print_step == 0:
                logger.info("loss: {}".format(training_loss / args.print_step))
                training_loss = 0
            step += 1
        # wandb.log({"Train acc1": hits / totals,"Train acc2":hits1/totals })

        # 评估
        dev_score, dev_score_1,f1,f1_1 = evaluation(model=model, dev_loader=dev_loader)
        logger.info("Validation Accuracy: {}".format(dev_score))
        logger.info("dev validation f1: {}".format(f1))
        logger.info("Three Class Validation Accuracy: {}".format(dev_score_1))
        logger.info("dev Three Class validation f1: {}".format(f1_1))
        # wandb.log({"Validation Accuracy": dev_score,"Three Class Validation Accuracy":dev_score_1 })

        # 测试的评估
        test_score, test_score_1,f2,f2_1 = evaluation(model=model, dev_loader=test_loader)
        logger.info("Test Accuracy: {}".format(test_score))
        logger.info("test validation f1: {}".format(f2))
        logger.info("Three Class Test Accuracy: {}".format(test_score_1))
        logger.info("test Three Class validation f1: {}".format(f2_1))
        # wandb.log({"Test Accuracy": test_score,"Three Class Test Accuracy":test_score_1 })

        if best_score < dev_score:
            logger.info(
                "Validation Accuracy Improve from {} to {}".format(
                    best_score, dev_score
                )
            )
            torch.save(model.state_dict(), args.model_save_path + args.model_name)
            best_score = dev_score
            patience = 0
        else:
            logger.info(
                "Validation Accuracy don't improve. Best Accuracy:" + str(best_score)
            )
            patience += 1
            if patience >= args.patience:
                logger.info(
                    "After {} epochs acc don't improve. break.".format(args.patience)
                )
                # wandb.save('model.h5')
                # wandb.finish()
                break


def evaluation(model, dev_loader):
    """
    模型在验证集上的正确率
    :param model:
    :param dev_loader:
    :return:
    """
    # eval mode
    model.eval()
    hits, hits1, totals = 0, 0, 0
    all_labels, all_preds = [], []
    all_labels1, all_preds1 = [], []

    with torch.no_grad():
        for batch in tqdm(dev_loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attn_mask, token_type, labels, labels_thr, matrices = batch
            
            logits_1, logits_2, _, _ = model(input_ids, attn_mask, token_type,matrices)

            _, predict1 = torch.max(logits_1, 1)
            hits1 += (predict1 == labels_thr).sum().item()
            # _, predict1 = torch.max(logits_1, 1)
            
            predict = (torch.sigmoid(logits_2) > 0.4).int()
            labels = labels.int()
            hits +=(predict == labels).sum().item()
            totals += labels.numel()
            # 收集所有标签和预测
            # 第二层
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predict.cpu().numpy())
            # 第一层
            all_labels1.append(labels_thr.cpu().numpy())
            all_preds1.append(predict1.cpu().numpy())
    
    # 将所有标签和预测合并为一个矩阵，方便计算F1
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels1 = np.concatenate(all_labels1, axis=0)
    all_preds1 = np.concatenate(all_preds1, axis=0)

    # 计算F1得分
    f1 = f1_score(all_labels, all_preds, average='micro')  # 或者使用 'macro' 根据需要
    f1_1 = f1_score(all_labels1, all_preds1, average='weighted')
    

    return hits / totals, hits1 / totals,f1,f1_1

    # f1_score_layer1 = f1_score(true_1, prediction_1,average='micro') 


def test(args):
    args.model_name = args.model_name.format(args.role)
    utils = data_utils.Utils(
        bert_path=args.bert_path,
        max_seq_len=args.max_seq_len,
        max_turns=args.max_turns,
        batch_size=args.batch_size,
        data_folder=args.data_folder,
        role=args.role,
    )
    test_loader = utils.data_loader("test_new", sort=False)

    model = HAN(
        bert_path=args.bert_path,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        num_classes_1=args.num_classes_1,
        num_classes_2=args.num_classes_2,
        max_turn=args.max_turns,
        max_seq=args.max_seq_len,
        coefficient=args.coefficient,
    ).to(device)
    model.load_state_dict(torch.load(r"./model/{}".format(args.model_name)))
    # model.load_state_dict(torch.load(r'./model/HAN_all_lr2e-5_dropout0.2_seq80_turns43_new.bin'))
    model = model.to(device)
    model.eval()

    prediction = []
    prediction_1 = []
    true = []
    true_1 = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attn_mask, token_type, labels, labels_thr, matrices = batch

            logits_1, logits_2, _, _ = model(input_ids, attn_mask, token_type,matrices)

            _, predict_1 = torch.max(logits_1, 1)
            # _, predict = torch.max(logits_2, 1)
            predict = torch.sigmoid(logits_2 >0.4).int()
            # print("predict_1:",predict_1)
            # print("predict:",predict)
            # print("labels:(第二层):",labels)
            # print("labels_thr:(第1层):",labels_thr)


            prediction.append(predict)
            prediction_1.append(predict_1)

            true.append(labels)  
            true_1.append(labels_thr)

    prediction = torch.cat(prediction).cpu().numpy()
    prediction_1 = torch.cat(prediction_1).cpu().numpy()
    true = torch.cat(true).cpu().numpy()
    true_1 = torch.cat(true_1).cpu().numpy()
    print(prediction)
    print(true)

    cm_1 = confusion_matrix(true_1, prediction_1)
    print("cm_1",cm_1)
    # cm = confusion_matrix(true, prediction)
    # print("cm",cm)

    print("Test set accuracy", accuracy_score(true, prediction))
    f1_score_layer2 = f1_score(true, prediction,average='samples')
    print("Test set F1", f1_score_layer2)
    # logger.info("Test F1: {}".format(f1_score_layer2))

    print("Three Class Test set accuracy", accuracy_score(true_1, prediction_1))
    f1_score_layer1 = f1_score(true_1, prediction_1,average='weighted') 
    print("Three Class Test set F1",f1_score_layer1 )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    if platform.system() == "Windows":
        bert_path = (
            # r"E:\study\NLP\长对话分类\BERT-AP-HAN\pytorch_chinese_L-12_H-768_A-12"
            r"D:\jupyter\model\roberta-large"
        )
    else:
        # bert_path = "./pytorch_chinese_L-12_H-768_A-12"
        bert_path = "/data/public_model/roberta-large/"

    parser.add_argument(
        "--bert_path", type=str, default=bert_path, help="预训练BERT模型（Pytorch）"
    )
    # 对输入的数据每两句拼接在一起，模型HAN不变。80*2=160
    parser.add_argument("--max_seq_len", type=int, default=160, help="句子最大长度")

    parser.add_argument("--max_turns", type=int, default=20, help="一次会话最大轮次")
    parser.add_argument("--data_folder", type=str, default=r"/data/alyssa/HLIDC_1/nlupp")
    parser.add_argument("--learning_rate", type=float, default=0.00002, help="学习率")
    parser.add_argument("--accumulation_steps", type=int, default=3, help="梯度累加")
    parser.add_argument("--batch_size", type=int, default=6, help="批次大小")
    parser.add_argument("--num_classes_1", type=int, default=3, help="类别数量")
    parser.add_argument("--num_classes_2", type=int, default=62, help="类别数量")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮次")
    parser.add_argument("--patience", type=int, default=3, help="early stopping")
    parser.add_argument("--log_path", type=str, default="./log/", help="日志文件夹")
    parser.add_argument(
        "--model_save_path", type=str, default="./model/", help="模型存放目录"
    )
    parser.add_argument(
        "--print_step", type=int, default=200, help="训练时每X步输出loss"
    )
    parser.add_argument(
        "--role", type=str, default="all", help="哪个类"
    )  # complaint consult handel
    parser.add_argument(
        "--model_name", type=str, default="HAN_adj_{}.bin", help="模型名称"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="丢弃概率")
    parser.add_argument("--hidden_size", type=int, default=31, help="LSTM隐藏层大小")
    parser.add_argument("--lamb", type=float, default=0.2, help="loss改变参数")
    parser.add_argument(
        "--do_train", action="store_true", help="do training procedure?"
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子数")
    parser.add_argument("--x", type=int, default=4, help="warmup中的epoch")

    parser.add_argument("--coefficient", type=float, default=1.0, help="权重因子")

    parser.add_argument(
        "--result_filename",
        type=str,
        required=False,
        default="test_result",
        help="测试集结果",
    )
    args = parser.parse_args()

    #   用随机种子跑
    def setup_seed(an_int):
        torch.manual_seed(an_int)
        torch.cuda.manual_seed_all(an_int)
        np.random.seed(an_int)
        random.seed(an_int)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    setup_seed(args.seed)
    # if args.do_train:
    #     wandb.init( project="APHAN_add",name = args.model_name)
    #     wandb.config = {
    # "learning_rate":args.learning_rate,
    # "epochs": 30,
    # "batch_size": args.batch_size,
    # "lambda":args.lamb,
    # "hidden_size":args.hidden_size,
    # "max_turns":args.max_turns,
    # "max_seq_len":args.max_seq_len
    # }
    #     wandb.config.update()

    print(torch.cuda.is_available())
    train(args)

    # if args.do_train:
    #     train(args)
    #     for handler in logger.handlers:
    #         if isinstance(handler, logging.FileHandler):
    #             handler.flush()  # 强制刷新日志文件

    test(args)

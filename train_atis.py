import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange
from transformers import AdamW, RobertaConfig, RobertaTokenizer, BertTokenizer, BertConfig
from transformers.trainer import get_linear_schedule_with_warmup
from data.utils import generate_fewshot_data_hwb, readfile

from evaluate import evaluate
from model import Model, FocalLoss
from utils.ckpt_utils import download_ckpt
from utils.data_utils import prepare_data, NluDataset, glue_processor

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args, shot_num):
    # Init
    set_seed(args.seed)
    processor = glue_processor[args.task_name.lower()]
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    print(tokenizer)
    # Data
    train_examples = processor.get_train_examples(args.train_data_path)
    dev_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)
    labels = processor.get_labels(args.data_dir)
    prompt_sent_list = processor.get_prompt_sent(args.data_dir)

    train_data_raw = prepare_data(train_examples, args.max_seq_len, tokenizer, labels, prompt_sent_list, args)
    dev_data_raw = prepare_data(dev_examples, args.max_seq_len, tokenizer, labels, prompt_sent_list, args)
    test_data_raw = prepare_data(test_examples, args.max_seq_len, tokenizer, labels, prompt_sent_list, args)
    print("# train examples %d" % len(train_data_raw))
    print("# dev examples %d" % len(dev_data_raw))
    print("# test examples %d" % len(test_data_raw))
    train_data = NluDataset(train_data_raw)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size,
                                  collate_fn=train_data.collate_fn, sampler=RandomSampler(train_data))

    # Model
    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = args.dropout
    model_config.num_intent = len(labels['intent_labels'])
    model_config.num_slot = len(labels['slot_labels'])
    if not os.path.exists(args.bert_ckpt_path):
        args.bert_ckpt_path = download_ckpt(args.bert_ckpt_path, args.bert_config_path, 'assets')
    model = Model.from_pretrained(config=model_config, pretrained_model_name_or_path=args.bert_ckpt_path)
    model.to(device)

    # Optimizer
    num_train_steps = int(len(train_data_raw) / args.batch_size * args.n_epochs)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.0},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    # scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_steps * args.warmup),num_train_steps)
    intent_loss_fnc = nn.BCEWithLogitsLoss()
    slot_loss_fnc = nn.CrossEntropyLoss(ignore_index=-100)

    # Training
    best_score = {"joint_acc": 0, "intent_acc": 0, "slot_accuracy_score": 0, "slot_precision": 0, "slot_recall": 0,
                  "slot_f1": 0}
    best_epoch = 0
    train_pbar = trange(0, args.n_epochs, desc="Epoch")
    epoch_loss = []
    id_loss = []
    sf_loss = []
    alpha = args.alpha
    for epoch in range(args.n_epochs):
        batch_loss = [] # 总损失
        intent_loss = [] # ID损失-smi
        slot_loss = [] # SF损失
        epoch_pbar = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = [b.to(device) if not isinstance(b, int) else b for b in batch]
            input_ids, segment_ids, input_mask, slot_ids, slot_ids_ori, intent_id = batch
            intent_logits, slot_logits = model(input_ids, segment_ids, input_mask)
            loss_intent = intent_loss_fnc(intent_logits, intent_id)
            loss_slot = slot_loss_fnc(slot_logits.view(-1, len(labels['slot_labels'])), slot_ids.view(-1))
            loss = alpha * loss_intent + (1 - alpha) * loss_slot

            intent_loss.append(loss_intent.item())
            slot_loss.append(loss_slot.item())
            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

            model.zero_grad()
            epoch_pbar.update(1)
            if (step + 1) % 100 == 0:
                print("[%d/%d] [%d/%d] mean_loss : %.3f" \
                      % (epoch + 1, args.n_epochs, step + 1,
                         len(train_dataloader), np.mean(batch_loss)))
        epoch_pbar.close()
        print('Epoch %d mean loss: %.3f' % (epoch + 1, np.mean(batch_loss)))
        id_loss.append(np.mean(intent_loss))
        sf_loss.append(np.mean(slot_loss))
        epoch_loss.append(np.mean(batch_loss))
        res, _, _, threshold = evaluate(model, dev_data_raw, labels, args)
        if res['joint_acc'] >= best_score['joint_acc']:
            best_score = res
            best_epoch = epoch + 1
            args.multi_intent_threshold = threshold
            save_path = os.path.join(args.save_dir, 'model_best.bin')
            torch.save(model.state_dict(), save_path)
        print("Best Score : ", best_score, 'in epoch ', best_epoch)
        train_pbar.update(1)
    train_pbar.close()
    plot_path = os.path.join(args.train_data_path, '%d-shot_loss.png' % (shot_num))
    plot_loss_figure(shot_num, id_loss, sf_loss, epoch_loss, args.n_epochs, plot_path)
    ckpt = torch.load(os.path.join(args.save_dir, 'model_best.bin'))
    model.load_state_dict(ckpt, strict=False)
    res, intent_report, slot_report, threshold = evaluate(model, test_data_raw, labels, args, mode="test")
    return res, intent_report, slot_report

def plot_loss_figure(shot_num, id_loss, sf_loss, epoch_loss, total_epochs, target_path):
    x = list(np.arange(1, total_epochs + 1, 1))
    plt.plot(x, id_loss, linewidth = 1, color = 'green', marker = 'o', label = 'id_loss')
    plt.plot(x, sf_loss, linewidth=1, color='orange', marker='o', label='sf_loss')
    plt.plot(x, epoch_loss, linewidth = 1, color = 'blue', marker = 'o', label = 'overall_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('%d-shot loss figure' % (shot_num))
    plt.grid()
    plt.legend()
    plt.savefig(target_path)
    plt.close()

def get_new_sample(label_to_num, last_res, iteration_percentage, add):
    # sample strategy
    # 1. 固定average增长量,　和iteration_percentage负相关
    beta = 1
    average_num = int((1 - iteration_percentage) ** beta * add)
    for key in last_res:
        label_to_num[key] += average_num
    # 2. 和last_res负相关的采样
    alpha = 1
    total_num_left = (add - average_num) * len(label_to_num)
    last_res_error = {k: ((1 - v[0])*(v[1]**0.3)) ** alpha for k, v in last_res.items()}
    total_error = sum(last_res_error.values())
    for key, error in last_res_error.items():
        label_to_num[key] += round(error / total_error * total_num_left)

    print('continue sampling according to last result: %s' %label_to_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1000, type=int)

    parser.add_argument("--task_name", default='nlu', type=str)
    parser.add_argument("--data_dir", default='data/MixATIS_clean/', type=str)
    parser.add_argument("--model_path", default='assets/', type=str)

    parser.add_argument("--save_dir", default='outputs', type=str)
    parser.add_argument("--max_seq_len", default=160, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--n_epochs", default=60, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--warmup", default=0.1, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--multi_intent_threshold", default=0.5, type=float)
    parser.add_argument("--alpha", default=0.5, type=float)
    args = parser.parse_args()
    args.vocab_path = os.path.join(args.model_path, 'vocab.txt')
    args.bert_ckpt_path = os.path.join(args.model_path, 'pytorch_model.bin')
    args.bert_config_path = os.path.join(args.model_path, 'config.json')
    print(args)

    num_sample = [2, 4, 6, 8, 10]
    ori_path = args.data_dir
    target_path = os.path.join(ori_path, 'hwb_sample')
    intent_label_list = [x.split('=-=')[0] for x in readfile(os.path.join(ori_path, "vocab/intent_vocab"))]
    label_to_num = {x: num_sample[0] for x in intent_label_list}

    for idx, num_each_cls in enumerate(num_sample):
        print('%s-shot training starts ...' % str(num_each_cls))
        train_data_path = './data/FewShotMixATIS/hwb_sample/fewshot-' + str(num_each_cls) + "/"
        args.train_data_path = train_data_path
        res, intent_report, slot_report = main(args, num_each_cls)
        print('%s-shot training ends' % str(num_each_cls))
        print('final test res : %s' % res)
        print('intent_classification_report: %s' % intent_report)
        print('slot_classification_report: %s' % slot_report)
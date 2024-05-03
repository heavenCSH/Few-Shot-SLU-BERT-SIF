import argparse
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import torch
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertConfig
from sklearn.metrics import precision_recall_curve
from model import Model
from utils.data_utils import NluDataset, glue_processor, prepare_data
from data.utils import readfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, data_raw, id_to_label, args, mode='dev'):
    slot_label_list = [x.split('=-=')[0] for x in readfile(os.path.join(args.data_dir, "vocab/slot_vocab"))] # original slot vocab
    slot_label_list_simple = id_to_label['slot_labels']
    intent_label_list = id_to_label['intent_labels']
    slot_label_to_id = {label : idx for idx, label in enumerate(slot_label_list)}

    model.eval()
    test_data = NluDataset(data_raw) # instances
    test_dataloader = DataLoader(test_data, batch_size=len(intent_label_list), collate_fn=test_data.collate_fn)

    joint_all = 0
    joint_correct = 0
    s_preds = []
    # s_labels = []
    s_restore = []
    i_preds = []
    i_labels = []

    predicted_masked_tokens = []
    epoch_pbar = tqdm(test_dataloader, desc="Evaluation", disable=False)
    # compute an appropriate threshold in dev dataset
    y_pred = []
    y_true = []
    for step, batch in enumerate(test_dataloader):
        batch = [b.to(device) if not isinstance(b, int) else b for b in batch]
        input_ids, segment_ids, input_mask, slot_ids, slot_ids_ori, intent_id = batch
        with torch.no_grad():
            # intent_output, slot_output = model(input_ids, segment_ids, input_mask, prompt_idx)
            intent_logits, slot_logits = model(input_ids, segment_ids, input_mask)
        # intent_evaluate
        intent_logits = torch.sigmoid(intent_logits)
        # threshold = (torch.max(intent_logits) + torch.min(intent_logits)) / 2
        # print("Current threshold:", str(args.multi_intent_threshold))
        intent_output = [idx for idx, score in enumerate(intent_logits) if score > args.multi_intent_threshold]
        intent_label = [idx for idx, res in enumerate(intent_id) if res == 1]

        i_preds.append(intent_output)
        i_labels.append(intent_label)

        # for calculating optimal threshold
        y_pred.append(intent_logits.cpu().tolist())
        y_true.append(intent_id.cpu().tolist())

        # slot_evaluate
        # slot_logits = slot_logits.argmax(dim=2)
        # slot_logits = slot_logits.tolist()

        # average for slots
        avg_slot_logits = torch.mean(slot_logits, dim=0, keepdim=True)
        tmp_avg = avg_slot_logits
        avg_slot_logits = avg_slot_logits.argmax(dim=2)
        avg_slot_logits = avg_slot_logits.tolist()


        slot_ids = slot_ids.tolist()
        slot_ids = [slot_ids[0]]
        slot_ids_ori = slot_ids_ori.tolist()
        slot_ids_ori = [slot_ids_ori[0]]
        for idx, (p, l, l_ori) in enumerate(zip(avg_slot_logits, slot_ids, slot_ids_ori)): # modify
            joint_all += 1
            p_text = reconstruct_slot_bound(p, l, slot_label_list_simple) # predicted slots
            l_text = align_predictions(l_ori, l_ori, slot_label_list)

            if step == len(test_dataloader) - 1:
                index = []
                for i, x in enumerate(l_ori):
                    if x != -100:
                        index.append(i)
                tmp_avg = torch.squeeze(tmp_avg, dim=0)
                tmp_avg = torch.index_select(tmp_avg, dim=0, index=torch.tensor(index).cuda())
                tmp_avg = tmp_avg.cpu().numpy()
                np.savetxt('avg_tensor.txt', tmp_avg, fmt='%.8f', delimiter='\t')

                print(p_text)
                print('================')
                print(l_text)

                tensor1 = torch.index_select(slot_logits, dim=0, index=torch.tensor([0]).cuda())
                tensor1 = torch.squeeze(tensor1, dim=0)
                tensor1 = torch.index_select(tensor1, dim=0, index=torch.tensor(index).cuda())
                tensor1 = tensor1.cpu().numpy()
                np.savetxt('tensor1.txt', tensor1, fmt='%.8f', delimiter='\t')

                tensor2 = torch.index_select(slot_logits, dim=0, index=torch.tensor([1]).cuda())
                tensor2 = torch.squeeze(tensor2, dim=0)
                tensor2 = torch.index_select(tensor2, dim=0, index=torch.tensor(index).cuda())
                tensor2 = tensor2.cpu().numpy()
                np.savetxt('tensor2.txt', tensor2, fmt='%.8f', delimiter='\t')

            if p_text == l_text and set(intent_output) == set(intent_label):
                joint_correct += 1
            s_preds.append(p_text)
            s_restore.append(l_text) # we use real slots to calculate precision and recall and f1 of SF.  s_restore is used for validation
        epoch_pbar.update(1)
    epoch_pbar.close()


    # calculate the optimal threshold
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    f1_scores = 2 * (precision * recall) / (precision + recall)
    threshold = thresholds[np.argmax(f1_scores)]

    # intent_report
    mlb = MultiLabelBinarizer(classes = [i for i in range(len(intent_label_list))]) # 高版本sklearn不支持多标签分类的classification_report，需要对标签进行二值化处理
    i_labels_binary = mlb.fit_transform(i_labels)
    i_preds_binary = mlb.transform(i_preds)
    intent_class_report_str = classification_report(i_labels_binary, i_preds_binary, target_names=intent_label_list, labels=[i for i in range(len(intent_label_list))])
    intent_class_report = reconstruct_class_report(intent_class_report_str)
    print('intent_classification_report: %s' % intent_class_report)

    # slot report
    # s_labels = [x.slots for x in data_raw]
    s_labels = []
    for idx, x in enumerate(data_raw):
        if idx % len(intent_label_list) == 0:
            s_labels.append(x.slots)

    for restore, real in zip(s_restore, s_labels): # validation for overall_acc calculation
        assert restore == real

    mlb1 = MultiLabelBinarizer(classes = [i for i in range(len(slot_label_list))])
    s_labels_id = [[slot_label_to_id[label] for label in instance] for instance in s_labels]
    s_preds_id = [[slot_label_to_id[label] if label in slot_label_to_id else 0 for label in instance] for instance in s_preds]
    s_labels_binary = mlb1.fit_transform(s_labels_id)
    s_preds_binary = mlb1.transform(s_preds_id)
    slot_class_report_str = classification_report(s_labels_binary, s_preds_binary, target_names=slot_label_list, labels=[i for i in range(len(slot_label_list))])
    slot_class_report = reconstruct_class_report(slot_class_report_str)
    print('slot_classification_report: %s' % slot_class_report)

    # for calculating ID acc
    i_preds_raw = []
    i_labels_raw = []
    for x in i_preds:
        i_preds_raw.append([intent_label_list[idx] for idx in x])
    for x in i_labels:
        i_labels_raw.append([intent_label_list[idx] for idx in x])

    # sentences = [x.words for idx,x in enumerate(data_raw) if not idx % len(intent_label_list)]
    # write_prediction_to_file(sentences, i_preds_raw, i_labels_raw)


    eval_res = {
        "joint_acc" : joint_correct / joint_all,
        "intent_acc" : cal_acc(i_preds_raw, i_labels_raw),
        "slot_acc" : accuracy_score(s_labels, s_preds),
        "slot_precision" : precision_score(s_labels, s_preds),
        "slot_recall" : recall_score(s_labels, s_preds),
        "slot_f1" : f1_score(s_labels, s_preds)
    }
    print("%s dataset evaluate results: %s" %(mode, eval_res))
    return eval_res, intent_class_report, slot_class_report, threshold

def reconstruct_class_report(class_report_str):
    class_report = {}
    all_lines = class_report_str.split('\n')
    for line in all_lines[2:-6]:
        line = line.strip().split()
        class_report[line[0]] = [float(line[-2]), int(line[-1])]
    return class_report

def write_prediction_to_file(sentences, predicted_masked_tokens, i_labels_raw, filename='preds'):
    res = []
    bad = []
    a_to_b_bad_case = []
    for s,t,l in zip(sentences, predicted_masked_tokens, i_labels_raw):
        res.append("input sentence: %s \n" % s)
        res.append("predicted intent: %s \n" %t)
        res.append("intent label: %s \n\n" %l)
        if set(t) != set(l):
            bad.append("input sentence: %s \n" % s)
            bad.append("predicted intent: %s \n" % t)
            bad.append("intent label: %s \n\n" % l)
            a_to_b_bad_case.append(l+'_to_'+t)
    # print(Counter(a_to_b_bad_case))
    with open(filename,'w',encoding='utf-8') as f:
        f.writelines(res)

    with open('bad_case','w',encoding='utf-8') as f:
        f.writelines(bad)

def cal_acc(preds, labels):
    acc = sum([1 if set(p) == set(l) else 0 for p, l in zip(preds, labels)]) / len(labels)
    return acc

def reconstruct_slot_bound(slot_output, slot_ids, slot_label_list): # simple_slot_vocab
    # 1. 排除不需要的idx
    real_pred = []
    # for p_list,l_list in zip(slot_output, slot_ids):
    #     single_pred = []
    #     for p,l in zip(p_list,l_list):
    #         if l == -100: # [pad]
    #             continue
    #         single_pred.append(slot_label_list[p])
    #     real_pred.append(single_pred)
    for p, l in zip(slot_output, slot_ids):
        if l == -100:
            continue
        real_pred.append(slot_label_list[p])
    # 2.恢复真实标签
    # final_pred = []
    # for single_pred in real_pred:
    #     final_single_pred = ['O'] if single_pred[0] == 'O' else ['B-'+single_pred[0]]
    #     for idx in range(1,len(single_pred)):
    #         cur_pred = single_pred[idx]
    #         pre_pred = single_pred[idx-1]
    #         if cur_pred == 'O':
    #             final_single_pred.append('O')
    #         else:
    #             if cur_pred == pre_pred:
    #                 final_single_pred.append('I-'+cur_pred)
    #             else:
    #                 final_single_pred.append('B-'+cur_pred)
    #     final_pred.append(final_single_pred)
    final_pred = ['O'] if real_pred[0] == 'O' else ['B-' + real_pred[0]]
    for idx in range(1, len(real_pred)):
        cur_pred = real_pred[idx]
        pre_pred = real_pred[idx - 1]
        if cur_pred == 'O':
            final_pred.append('O')
        else:
            if cur_pred == pre_pred:
                final_pred.append('I-'+cur_pred)
            else:
                final_pred.append('B-'+cur_pred)

    return final_pred

def align_predictions(preds, slot_ids, id_to_label):
    # aligned_labels = []
    aligned_preds = []
    for p, l in zip(preds, slot_ids):
        if l != -100:
            aligned_preds.append(id_to_label[p])
            # aligned_labels.append(id_to_label[l])
    return aligned_preds


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    # Init
    set_seed(args.seed)
    processor = glue_processor[args.task_name.lower()]
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)

    # Data
    dev_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)
    labels = processor.get_labels(args.data_dir)
    prompt_sent_list = processor.get_prompt_sent(args.data_dir)
    # dev_data_raw = prepare_data(dev_examples, args.max_seq_len, tokenizer, labels, prompt_sent_list)
    test_data_raw = prepare_data(test_examples, args.max_seq_len, tokenizer, labels, prompt_sent_list)

    # Model
    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.use_crf = args.use_crf
    model_config.dropout = args.dropout
    model_config.num_intent = len(labels['intent_labels'])
    model_config.num_slot = len(labels['slot_labels'])
    model = Model.from_pretrained(config=model_config, pretrained_model_name_or_path=args.model_ckpt_path)
    # ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    # model.load_state_dict(ckpt, strict=False)
    model.to(device)
    # evaluate(model, dev_data_raw, labels,tokenizer, 'dev')
    evaluate(model, test_data_raw, labels, tokenizer,'test')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--task_name", default='nlu', type=str)
    parser.add_argument("--data_dir", default='data/snips/', type=str)
    parser.add_argument("--model_path", default='assets/', type=str)

    parser.add_argument("--model_ckpt_path", default='assets/pytorch_model.bin', type=str)
    parser.add_argument("--use_crf", default=False, type=bool)
    parser.add_argument("--max_seq_len", default=80, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    args = parser.parse_args()
    args.vocab_path = os.path.join(args.model_path, 'vocab.txt')
    args.bert_config_path = os.path.join(args.model_path, 'config.json')
    print(args)
    main(args)

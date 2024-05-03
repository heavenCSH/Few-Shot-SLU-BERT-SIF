import os
from dataclasses import dataclass
from typing import Optional, List

import torch
from torch.utils.data import Dataset
from transformers import DataProcessor, logging

from data.utils import readfile

logger = logging.get_logger(__name__)


@dataclass
class InputExample:
    guid: str
    words: List[str]
    intent: Optional[str]
    slots: Optional[List[str]]


class NluProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "seq.in")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "seq.in")), # [[sentence1], [sentence2]...]
                                     self._read_tsv(os.path.join(data_dir, "seq.out")), "train") # [[result1], [result2]...]

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev/seq.in")),
                                     self._read_tsv(os.path.join(data_dir, "dev/seq.out")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test/seq.in")),
                                     self._read_tsv(os.path.join(data_dir, "test/seq.out")), "test")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test/intent_seq.in")),
                                     self._read_tsv(os.path.join(data_dir, "test/intent_seq.out")), "predict")

    def get_labels(self, data_dir):
        """See base class."""
        slot_labels_list = self._read_tsv(os.path.join(data_dir, "vocab/slot_vocab_simple")) # [[label1], [label2]...]
        slot_labels = [label[0].split('=-=')[0] for label in slot_labels_list] # [label1, label2...]
        intent_labels_list = self._read_tsv(os.path.join(data_dir, "vocab/intent_vocab")) # [[intent1=-=prompt1], [intent2=-=prompt2]...]
        intent_labels = [label[0].split('=-=')[0] for label in intent_labels_list] # [intent1, intent2...]
        intent_prompts = [label[0].split('=-=')[1] for label in intent_labels_list] # [prompt1, prompt2...]

        labels = {'intent_labels': intent_labels, 'slot_labels': slot_labels,'intent_prompts': intent_prompts}
        return labels

    def read_file(self, data_dir):
        with open(data_dir, 'r', encoding='utf-8') as f:
            res = f.readlines()
        return [x.strip() for x in res]

    def get_prompt_sent(self, data_dir):
        prompt_sent = self.read_file(os.path.join(data_dir, "vocab/sentence_prompt"))
        prompt_sent_A = prompt_sent[0].replace('[A]','').strip()
        prompt_sent_B = prompt_sent[1].replace('[B]','').strip()
        return [prompt_sent_A, prompt_sent_B]

    def _create_examples(self, lines_in, lines_out, set_type):
        """Creates examples for the training, dev and test sets."""

        examples = []
        for i, (line, out) in enumerate(zip(lines_in, lines_out)):
            label_split = out[0].strip().split()
            guid = "%s-%s" % (set_type, i)
            words = line[0][:].strip() # words暂时是一个str，后续用tokenizer分词
            slots = None if set_type == "predict" else label_split[1:]
            intents = None if set_type == "predict" else label_split[0]
            examples.append(InputExample(guid=guid, words=words, slots=slots, intent=intents))

        return examples


class TrainingInstance:
    def __init__(self, example, max_seq_len):
        self.words = example.words.split() # list
        self.slots = example.slots # list
        self.intent = example.intent # str
        self.max_seq_len = max_seq_len # int

    def make_instance(self, tokenizer, intent_label_map, slot_label_map, prompt_intent_sent ,prompt_sent_list, intent_idx, slot_label_map_ori, pad_label_id=-100):
        slot_label_map['[PAD]'] = pad_label_id
        tokens = []
        slot_ids = []
        slot_ids_ori = []
        prompt_intent_tokens = tokenizer.tokenize(prompt_intent_sent)

        if self.slots:
            assert len(self.words) == len(self.slots)
            for word, label_ori in zip(self.words, self.slots):
                label = label_ori.split('-')[1] if '-' in label_ori else label_ori # part of entity (exclude B- and I-)
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    slot_ids.extend([slot_label_map[label]] + [pad_label_id] * (len(word_tokens) - 1))
                    slot_ids_ori.extend([slot_label_map_ori[label_ori]] + [pad_label_id] * (len(word_tokens) - 1))
        else:
            # 预测时，把需要预测的位置置为1
            for word in self.words:
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    slot_ids.extend([1] + [0] * (len(word_tokens) - 1))
        assert len(slot_ids) == len(tokens)

        # TODO 判断长度越界

        # 添加prompt部分
        # prompt_sent_list[0], prompt_sent_list[1] = "", ""
        prompt_sent_A = tokenizer.tokenize(prompt_sent_list[0])
        prompt_sent_B = tokenizer.tokenize(prompt_sent_list[1])

        tokens_A = prompt_sent_A + tokens
        tokens_B = prompt_sent_B + prompt_intent_tokens

        # convert token to ids
        tokens = ["[CLS]"] + tokens_A + ["[SEP]"] + tokens_B # tokenized input sentence
        assert len(tokens) <= self.max_seq_len
        self.prompt_input = tokens
        # [cls] + A + tokens + [sep] + B + prompt_intent_tokens
        self.slot_ids = ([pad_label_id] + [pad_label_id] * len(prompt_sent_A) +
                         slot_ids + [pad_label_id] + [pad_label_id] * len(prompt_sent_B) + [pad_label_id] * len(prompt_intent_tokens))
        self.slot_ids_ori = ([pad_label_id] + [pad_label_id] * len(prompt_sent_A) +
                         slot_ids_ori + [pad_label_id] + [pad_label_id] * len(prompt_sent_B) + [pad_label_id] * len(
                    prompt_intent_tokens))
        self.input_ids = tokenizer.convert_tokens_to_ids(tokens)
        self.segment_id = [0] * (len(tokens_A) + 2) + [1] * len(tokens_B)
        self.input_mask = [1] * len(self.input_ids)
        padding_length = self.max_seq_len - len(self.input_ids)
        if padding_length > 0:
            self.input_ids = self.input_ids + [0] * padding_length
            self.segment_id = self.segment_id + [0] * padding_length
            self.input_mask = self.input_mask + [0] * padding_length
            self.slot_ids = self.slot_ids + [pad_label_id] * padding_length
            self.slot_ids_ori = self.slot_ids_ori + [pad_label_id] * padding_length

        # self.intent_id = 1 if intent_label_map[self.intent] == intent_idx else 0 # if self.intent else None 若真实intent的idx等于prompt的idx，则为正确的intent-prompt组合
        self.intent_id = 1 if intent_idx in [intent_label_map[i] for i in self.intent.split('#')] else 0


class NluDataset(Dataset):
    def __init__(self, data, annotated=True):
        self.data = data
        self.len = len(data)
        self.annotated = annotated

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        input_ids = torch.tensor([f.input_ids for f in batch], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_id for f in batch], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in batch], dtype=torch.long)
        slot_ids = torch.tensor([f.slot_ids for f in batch], dtype=torch.long)
        slot_ids_ori = torch.tensor([f.slot_ids_ori for f in batch], dtype=torch.long)
        intent_id = torch.tensor([f.intent_id for f in batch], dtype=torch.float) if self.annotated else None
        return input_ids, segment_ids, input_mask, slot_ids, slot_ids_ori, intent_id


def prepare_data(examples, max_seq_len, tokenizer, labels, prompt_sent_list, args):
    slot_label_map = {label: idx for idx, label in enumerate(labels['slot_labels'])} # slot-idx mapping
    intent_label_map = {label: idx for idx, label in enumerate(labels['intent_labels'])} # intent-idx mapping

    slot_label_list = [x.split('=-=')[0] for x in readfile(os.path.join(args.data_dir, "vocab/slot_vocab"))]  # original slot vocab
    slot_label_map_ori = {label : idx for idx, label in enumerate(slot_label_list)}

    prompt_intent_sents = labels['intent_prompts']
    data = []

    for idx,example in enumerate(examples):
        for intent_idx, prompt_intent_sent in enumerate(prompt_intent_sents): # 构造同一个句子的 k 种input prompt(k为intent种类数) instance
            instance = TrainingInstance(example, max_seq_len)
            instance.make_instance(tokenizer, intent_label_map, slot_label_map, prompt_intent_sent,prompt_sent_list, intent_idx, slot_label_map_ori)
            if idx < 0:
                print('Training Example %s :' %idx)
                print("Input sentence: %s" % instance.prompt_input)
                print('Input_ids: %s' %instance.input_ids)
                print("Input segment ids: %s" % instance.segment_id)
                print('Intent_label :%s' % instance.intent_id)
            data.append(instance)

    return data


glue_processor = {
    'nlu': NluProcessor()
}

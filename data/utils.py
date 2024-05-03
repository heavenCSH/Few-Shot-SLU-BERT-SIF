import os


def generate_fewshot_data(ori_train_path,target_path, num_each_cls):
    ori_train_in = readfile(os.path.join(ori_train_path, "train/intent_seq.in"))
    ori_train_out = readfile(os.path.join(ori_train_path, "train/intent_seq.out"))
    intent_label_list = [x.split('=-=')[0] for x in readfile(os.path.join(ori_train_path, "vocab/intent_vocab"))]
    dic = {x: [] for x in intent_label_list}  # slot: [[i,o]]

    for i, o in zip(ori_train_in, ori_train_out):
        intent = o.split()[0]
        dic[intent].append([i, o])


    fewshot_train_in = []
    fewshot_train_out = []

    for k, v in dic.items():
        if "#" in k: # exclude multi-intent
            continue
        in_list = [x[0] for x in v]
        out_list = [x[1] for x in v]

        fewshot_train_in.extend(in_list[:num_each_cls])
        fewshot_train_out.extend(out_list[:num_each_cls])

    target_path = os.path.join(target_path, "fewshot-" + str(num_each_cls))
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    writefile(fewshot_train_in, os.path.join(target_path, "intent_seq.in"))
    writefile(fewshot_train_out, os.path.join(target_path, "intent_seq.out"))

def generate_fewshot_data_hwb(ori_train_path,target_path, num_each_cls,label_to_num):
    ori_train_in = readfile(os.path.join(ori_train_path, "train/seq.in")) # read sentences
    ori_train_out = readfile(os.path.join(ori_train_path, "train/seq.out")) # read intents and slots
    # ori_label = readfile(os.path.join(ori_train_path, "train/label")) # read intents
    intent_label_list = [x.split('=-=')[0] for x in readfile(os.path.join(ori_train_path, "vocab/intent_vocab"))]
    dic = {x: [] for x in intent_label_list}  # slot: [[i,o]]

    for sentence, outputs in zip(ori_train_in, ori_train_out):
        intents = outputs.split()[0]
        for intent in intents.split("#"):
            dic[intent].append([sentence, outputs])

    fewshot_train_in = []
    fewshot_train_out = []

    for k, v in dic.items():
        if "#" in k: # exclude multi-intent
            continue
        in_list = [x[0] for x in v]
        out_list = [x[1] for x in v]

        fewshot_train_in.extend(in_list[:label_to_num[k]])
        fewshot_train_out.extend(out_list[:label_to_num[k]])

    target_path = os.path.join(target_path, "fewshot-" + str(num_each_cls))
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    writefile(fewshot_train_in, os.path.join(target_path, "seq.in"))
    writefile(fewshot_train_out, os.path.join(target_path, "seq.out"))
    return target_path

def readfile(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        res = [x.strip() for x in f.readlines()]
    return res


def writefile(str_list, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines([x + '\n' for x in str_list])



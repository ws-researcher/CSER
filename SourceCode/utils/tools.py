import copy
import datetime
import os
from typing import List
import numpy as np
from random import sample
from SourceCode.utils.constant import pos_dict, tags_dict

np.random.seed(1741)
import torch
torch.manual_seed(1741)
import random
random.seed(1741)
import spacy
from transformers import RobertaTokenizer
# from utils.constant import *


tokenizer = RobertaTokenizer.from_pretrained('../pretrained_models/roberta-base', unk_token='<unk>')
nlp = spacy.load("en_core_web_sm")


# Padding function
def padding(sent, pos = False, max_sent_len = 194):
    if pos == False:
        one_list = [1] * max_sent_len # pad token id
        mask = [0] * max_sent_len
        one_list[0:len(sent)] = sent
        mask[0:len(sent)] = [1] * len(sent)
        return one_list, mask
    else:
        one_list = [0] * max_sent_len # none id 
        one_list[0:len(sent)] = sent
        return one_list
      
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def RoBERTa_list(content, token_list = None, token_span_SENT = None):
    encoded = tokenizer.encode(content)
    roberta_subword_to_ID = copy.deepcopy(encoded)
    roberta_subwords = []
    roberta_subwords_no_space = []
    removes = []
    for index, i in enumerate(encoded):
        r_token = tokenizer.decode([i])
        if r_token != " ":
            roberta_subwords.append(r_token)
            if r_token[0] == " ":
                roberta_subwords_no_space.append(r_token[1:])
            else:
                roberta_subwords_no_space.append(r_token)
        else:
            removes.append(index)

    roberta_subword_to_ID = [roberta_subword_to_ID[i] for i in range(len(roberta_subword_to_ID)) if i not in removes]

    roberta_subword_span = tokenized_to_origin_span(content, roberta_subwords_no_space[1:-1]) # w/o <s> and </s>
    roberta_subword_map = []
    if token_span_SENT is not None:
        roberta_subword_map.append(-1) # "<s>"
        for subword in roberta_subword_span:
            roberta_subword_map.append(token_id_lookup(token_span_SENT, subword[0], subword[1]))
        roberta_subword_map.append(-1) # "</s>" 
        return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, roberta_subword_map
    else:
        return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, -1

def tokenized_to_origin_span(text: str, token_list: List[str]):
    token_span = []
    pointer = 0
    for token in token_list:
        start = text.find(token, pointer)
        if start != -1:
            end = start + len(token) - 1
            pointer = end + 1
            token_span.append([start, end])
            assert text[start: end+1] == token, f"token: {token} - text:{text}"
        else:
            token_span.append([-100, -100])
    return token_span

def sent_id_lookup(my_dict, start_char, end_char = None):
    for sent_dict in my_dict['sentences']:
        if end_char is None:
            if start_char >= sent_dict['sent_start_char'] and start_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']
        else:
            if start_char >= sent_dict['sent_start_char'] and end_char <= sent_dict['sent_end_char']:
                return sent_dict['sent_id']

def token_id_lookup(token_span_SENT, start_char, end_char):
    for index, token_span in enumerate(token_span_SENT):
        if start_char >= token_span[0] and end_char <= token_span[1]:
            return index

def span_SENT_to_DOC(token_span_SENT, sent_start):
    token_span_DOC = []
    #token_count = 0
    for token_span in token_span_SENT:
        start_char = token_span[0] + sent_start
        end_char = token_span[1] + sent_start
        #assert my_dict["doc_content"][start_char] == sent_dict["tokens"][token_count][0]
        token_span_DOC.append([start_char, end_char])
        #token_count += 1
    return token_span_DOC

def id_lookup(span_SENT, start_char):
    # this function is applicable to RoBERTa subword or token from ltf/spaCy
    # id: start from 0
    token_id = -1
    for token_span in span_SENT:
        token_id += 1
        if token_span[0] <= start_char and token_span[1] >= start_char:
            return token_id
    raise ValueError("Nothing is found. \n span sentence: {} \n start_char: {}".format(span_SENT, start_char))

def pos_to_id(sent_pos):
    id_pos_sent =  [pos_dict.get(pos) if pos_dict.get(pos) != None else 0 
                    for pos in sent_pos]
    return id_pos_sent

def tag_to_id(sent_tag):
    id_tag_sent =  [tags_dict.get(tag) if tags_dict.get(tag) != None else 0
                    for tag in sent_tag]
    return id_tag_sent
            
def pad_to_max_ns(ctx_augm_emb):
    max_ns = 0
    ctx_augm_emb_paded = []
    for ctx in ctx_augm_emb:
        # print(ctx.size())
        max_ns  = max(max_ns, ctx.size(0))
    
    for ctx in ctx_augm_emb:
        pad = torch.zeros((max_ns, 768))
        if ctx.size(0) < max_ns:
            pad[:ctx.size(0), :] = ctx
            ctx_augm_emb_paded.append(pad)
        else:
            ctx_augm_emb_paded.append(ctx)
    return ctx_augm_emb_paded

def word_dropout(seq_id, position, is_word=True, dropout_rate=0.05):
    if is_word==True:
        drop_sent = [3 if np.random.rand() < dropout_rate and i not in position else seq_id[i] for i in range(len(seq_id))]
    if is_word==False:
        drop_sent = [20 if np.random.rand() < dropout_rate and i not in position else seq_id[i] for i in range(len(seq_id))]
    # print(drop_sent)
    return drop_sent

def word_dropout_c(seq_id, pos, tag, position, is_word=True, dropout_rate=0.05):
    IN_index = get_reserve_index(tag)
    position.extend(IN_index)

    drop_sent = [3 if np.random.rand() < dropout_rate and i not in position else seq_id[i] for i in range(len(seq_id))]

    drop_pos = [20 if np.random.rand() < dropout_rate and i not in position else pos[i] for i in range(len(pos))]

    return drop_sent, drop_pos

def create_target(x_sent, y_sent, x_sent_id, y_sent_id, x_position, y_position):
    # 按前后顺序拼接anchor句子，并得到新的事件位置
    if x_sent_id < y_sent_id:
        sent = x_sent + y_sent[1:] # <s> x_sent </s> y_sent </s>
        y_position_new = y_position + len(x_sent) - 1
        x_position_new = x_position
        assert y_sent[y_position] == sent[y_position_new]
        assert x_sent[x_position] == sent[x_position_new]
    elif x_sent_id == y_sent_id:
        assert x_sent == y_sent
        sent = x_sent
        x_position_new = x_position
        y_position_new = y_position
        assert y_sent[y_position] == sent[y_position_new]
        assert x_sent[x_position] == sent[x_position_new]
    else:
        sent = y_sent + x_sent[1:]
        y_position_new = y_position
        x_position_new = x_position + len(y_sent) - 1
        assert y_sent[y_position] == sent[y_position_new]
        assert x_sent[x_position] == sent[x_position_new]
    return sent, x_position_new, y_position_new

def make_predictor_input(sents, sents_pos, sents_tag, x_position, y_position, flag, xy, dropout_rate=0.05, is_test=False):
    bs = len(sents)
    assert len(x_position) == bs, 'Each element must be same batch size'
    max_len = 0
    _augm_target = []
    _augm_pos_target = []
    _x_augm_position = []
    _y_augm_position = []
    flag_new = []
    xy_new = []

    augm_target = []
    augm_target_mask = []
    augm_pos_target = []
    x_augm_position = []
    y_augm_position = []


    for i in range(bs):
        sent = sents[i]
        pos = sents_pos[i]
        tag = sents_tag[i]
        x_positioni = x_position[i]
        y_positioni = y_position[i]
        if is_test == False:

            augment = word_dropout(sent, [x_positioni, y_positioni], dropout_rate=dropout_rate)
            pos_augment = word_dropout(pos, [x_positioni, y_positioni], is_word=False, dropout_rate=dropout_rate)

            max_len = max(len(augment), max_len)
            x_augm_position.append(x_positioni)
            y_augm_position.append(y_positioni)
            _augm_target.append(augment)
            _augm_pos_target.append(pos_augment)
            flag_new.append(flag[i])
            xy_new.append(xy[i])

        else:

            max_len = max(len(sent), max_len)
            x_augm_position.append(x_positioni)
            y_augm_position.append(y_positioni)
            _augm_target.append(sent)
            _augm_pos_target.append(pos)
            flag_new.append(flag[i])
            xy_new.append(xy[i])

    for i in range(len(_augm_target)):
        _augment = _augm_target[i]
        _pos_augment = _augm_pos_target[i]
        pad, mask = padding(_augment, max_sent_len=max_len)
        augm_target.append(pad)
        augm_target_mask.append(mask)
        augm_pos_target.append(padding(_pos_augment, pos=True, max_sent_len=max_len))

    augm_target = torch.tensor(augm_target)
    augm_target_mask = torch.tensor(augm_target_mask)
    augm_pos_target = torch.tensor(augm_pos_target)
    x_augm_position = torch.tensor(x_augm_position)
    y_augm_position = torch.tensor(y_augm_position)
    flag = torch.tensor(flag_new)
    xy = torch.tensor(xy_new)
    return augm_target, augm_target_mask, augm_pos_target, x_augm_position, y_augm_position, flag, xy

def make_zl_input(sents, sents_pos, sents_tag, x_position, y_position, flag, xy, dropout_rate=0.05, is_test=False):
    bs = len(sents)
    assert len(x_position) == bs, 'Each element must be same batch size'
    max_len = 0
    _augm_target = []
    _augm_pos_target = []
    _x_augm_position = []
    _y_augm_position = []
    flag_new = []
    xy_new = []

    augm_target = []
    augm_target_mask = []
    augm_pos_target = []
    x_augm_position = []
    y_augm_position = []


    for i in range(bs):
        sent = sents[i]
        pos = sents_pos[i]
        tag = sents_tag[i]
        x_positioni = x_position[i]
        y_positioni = y_position[i]

        augment = word_dropout(sent, [x_positioni, y_positioni], dropout_rate=dropout_rate)
        pos_augment = word_dropout(pos, [x_positioni, y_positioni], is_word=False, dropout_rate=dropout_rate)

        max_len = max(len(augment), max_len)
        x_augm_position.append(x_positioni)
        y_augm_position.append(y_positioni)
        _augm_target.append(augment)
        _augm_pos_target.append(pos_augment)
        flag_new.append(flag[i])
        xy_new.append(xy[i])


    for i in range(len(_augm_target)):
        augment = _augm_target[i]
        pos_augment = _augm_pos_target[i]
        pad, mask = padding(augment, max_sent_len=max_len)
        augm_target.append(pad)
        augm_target_mask.append(mask)
        augm_pos_target.append(padding(pos_augment, pos=True, max_sent_len=max_len))

    target = torch.tensor(augm_target)
    target_mask = torch.tensor(augm_target_mask)
    pos_target = torch.tensor(augm_pos_target)
    x_position = torch.tensor(x_augm_position)
    y_position = torch.tensor(y_augm_position)
    flag = torch.tensor(flag_new)
    xy = torch.tensor(xy_new)
    return target, target_mask, pos_target, x_position, y_position, flag, xy


def make_sc_input(sents, sents_pos, sents_tag, x_position, y_position, flag, xy, dropout_rate=0.05, views=None, is_reverse=False):
    bs = len(sents)
    assert len(x_position) == bs, 'Each element must be same batch size'
    max_len = 0
    _augm_target = []
    _augm_pos_target = []
    _x_augm_position = []
    _y_augm_position = []
    flag_new = []
    xy_new = []

    augm_target = []
    augm_target_mask = []
    augm_pos_target = []
    x_augm_position = []
    y_augm_position = []

    for i in range(bs):
        sent = sents[i]
        pos = sents_pos[i]
        tag = sents_tag[i]
        x_positioni = x_position[i]
        y_positioni = y_position[i]

        augs = [sent]
        pos_augs = [pos]
        for j in range(views):
            augment, pos_augment = word_dropout_c(sent, pos, tag, [x_positioni, y_positioni],
                                                dropout_rate=dropout_rate)
            augs.append(augment)
            pos_augs.append(pos_augment)

        max_len = max(len(sent), max_len)
        _augm_target.append(augs)
        _augm_pos_target.append(pos_augs)
        flag_new.append(flag[i])
        xy_new.append(xy[i])

        if is_reverse:
            _x_augm_position.append(y_positioni)
            _y_augm_position.append(x_positioni)
        else:
            _x_augm_position.append(x_positioni)
            _y_augm_position.append(y_positioni)

    for i in range(bs):
        _augment = _augm_target[i]
        _pos_augment = _augm_pos_target[i]

        pads, masks, pos_pads, x_augm_positions, y_augm_positions = [], [], [], [], []
        for j in range(len(_augment)):
            pad, mask = padding(_augment[j], max_sent_len=max_len)
            pads.append(pad)
            masks.append(mask)
            pos_pad = padding(_pos_augment[j], pos=True, max_sent_len=max_len)
            pos_pads.append(pos_pad)
            x_augm_positions.append(_x_augm_position[i])
            y_augm_positions.append(_y_augm_position[i])

        augm_target.append(pads)
        augm_target_mask.append(masks)
        augm_pos_target.append(pos_pads)
        x_augm_position.append(x_augm_positions)
        y_augm_position.append(y_augm_positions)

    augm_target = torch.tensor(augm_target)
    augm_target_mask = torch.tensor(augm_target_mask)
    augm_pos_target = torch.tensor(augm_pos_target)
    x_augm_position = torch.tensor(x_augm_position)
    y_augm_position = torch.tensor(y_augm_position)

    augm_target = torch.reshape(augm_target, (-1, augm_target.shape[-1]))
    augm_target_mask = torch.reshape(augm_target_mask, (-1, augm_target_mask.shape[-1]))
    augm_pos_target = torch.reshape(augm_pos_target, (-1, augm_pos_target.shape[-1]))
    x_augm_position = torch.reshape(x_augm_position, (-1,))
    y_augm_position = torch.reshape(y_augm_position, (-1,))

    return augm_target, augm_target_mask, augm_pos_target, x_augm_position, y_augm_position, flag_new, xy_new

def augment_target(x_sent, y_sent, x_sent_id, y_sent_id, x_possition, y_possition, is_pos=False, is_reverse=False):

    sent = []
    if x_sent_id < y_sent_id and not is_reverse:
        if is_pos:
            sent = [0] + x_sent[1:] + [0] + y_sent[1:] + [0]
        else:
            sent = [0] + x_sent[1:] + [2] + y_sent[1:] + [2]

        x_possition_new = x_possition
        y_possition_new = len(x_sent) + y_possition
    elif x_sent_id < y_sent_id and is_reverse:
        if is_pos:
            sent = [0] + y_sent[1:] + [0] + x_sent[1:] + [0]
        else:
            sent = [0] + y_sent[1:] + [2] + x_sent[1:] + [2]

        x_possition_new = len(y_sent) + x_possition
        y_possition_new = y_possition

    elif x_sent_id == y_sent_id:
        if is_pos:
            sent = [0] + x_sent[1:] + [0]
        else:
            sent = [0] + x_sent[1:] + [2]

        x_possition_new = x_possition
        y_possition_new = y_possition

    # elif x_sent_id == y_sent_id and not is_reverse:
    #     if is_pos:
    #         sent = [0] + x_sent[1:] + [0]
    #     else:
    #         sent = [0] + x_sent[1:] + [2]
    #
    #     x_possition_new = x_possition
    #     y_possition_new = y_possition

    # elif x_sent_id == y_sent_id and is_reverse:
    #     x_sent_r = list(reversed(x_sent))
    #     if is_pos:
    #         sent = [0] + x_sent_r[:-1] + [0]
    #     else:
    #         sent = [0] + x_sent_r[:-1] + [2]
    #
    #     x_possition_new = len(sent) - x_possition - 1
    #     y_possition_new = len(sent) - y_possition - 1

    elif x_sent_id > y_sent_id and not is_reverse:
        if is_pos:
            sent = [0] + y_sent[1:] + [0] + x_sent[1:] + [0]

        else:
            sent = [0] + y_sent[1:] + [2] + x_sent[1:] + [2]
        y_possition_new = y_possition
        x_possition_new = len(y_sent) + x_possition

    elif x_sent_id > y_sent_id and is_reverse:
        if is_pos:
            sent = [0] + x_sent[1:] + [0] + y_sent[1:] + [0]

        else:
            sent = [0] + x_sent[1:] + [2] + y_sent[1:] + [2]
        y_possition_new = len(x_sent) + y_possition
        x_possition_new = x_possition

    assert sent[x_possition_new] == x_sent[x_possition]
    assert sent[y_possition_new] == y_sent[y_possition]

    return sent, x_possition_new, y_possition_new

def processing_vague(logits, threshold, vague_id):
    bs = logits.size(0)
    predicts = []
    for i in range(bs):
        logit = logits[i].detach()
        logit = torch.softmax(logit, dim=0)
        entropy = - torch.sum(logit * torch.log(logit)).cpu().item()
        if entropy > threshold:
            predict = vague_id
        else:
            predict = torch.max(logit.unsqueeze(0), 1).indices.cpu().item()
        predicts.append(predict)
    # print(predicts)
    return predicts


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collate_fn(batch):
    return tuple(zip(*batch))


def spilt(all_data, r):
    n_data = len(all_data)
    n_train = int(n_data*r[0])
    n_validate = int(n_data * r[1])
    train = all_data[:n_train]
    validate = all_data[n_train: n_train + n_validate]
    test = all_data[n_train + n_validate:]
    return train, validate, test

def get_reserve_index(poses):
    punctuations = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    reserve_index = set()

    id = tags_dict.get("IN")

    num_pos = len(poses)
    for i in range(num_pos):
        if poses[i] == id:
            reserve_index.add(i)
            if i != num_pos - 1 and poses[i + 1] not in punctuations:
                reserve_index.add(i + 1)

    return reserve_index


def read_evaluation_file(fn):
    res = []
    if not os.path.exists(fn):
        return res
    for line in open(fn):
        fileds = line.strip().split('\t')
        res.append(fileds)
    return res


def patternArgm(x_sent, y_sent, x_sent_pos, y_sent_pos, x_sent_tag, y_sent_tag, x_sent_id, y_sent_id, x_position,
                                             y_position, flag, xy):
    BeforePatternsWords = ["before", "then", "till", "until", "so", "thus", "therefore", "so that",
                           " before", " then", " till", " until", " so", " thus", " therefore", " so that",
                           "before ", "then ", "till ", "until ", "so ", "thus ", "therefore ", "so that ",
                           " before ", " then ", " till ", " until ", " so ", " thus ", " therefore ", " so that ",
                           "Before", "Then", "Till", "Until", "So", "Thus", "Therefore", "So that",
                           " Before", " Then", " Till", " Until", " So", " Thus", " Therefore", " So that",
                           "Before ", "Then ", "Till ", "Until ", "So ", "Thus ", "Therefore ", "So that ",
                           " Before ", " Then ", " Till ", " Until ", " So ", " Thus ", " Therefore ", " So that ",
                           "BEFORE", "THEN", "TILL", "UNTIL", "SO", "THUS", "THEREFORE", "SO THAT",
                           " BEFORE", " THEN", " TILL", " UNTIL", " SO", " THUS", " THEREFORE", " SO THAT",
                           "BEFORE ", "THEN ", "TILL ", "UNTIL ", "SO ", "THUS ", "THEREFORE ", "SO THAT ",
                           " BEFORE ", " THEN ", " TILL ", " UNTIL ", " SO ", " THUS ", " THEREFORE ", " SO THAT "]
    AfterPatternsWords = ["after", "once", "because",
                          " after", " once", " because",
                          "after ", "once ", "because ",
                          " after ", " once ", " because ",
                          "After", "Once", "Because",
                          " After", " Once", " Because",
                          "After ", "Once ", "Because ",
                          " After ", " Once ", " Because ",
                          "AFTER", "ONCE", "BECAUSE",
                          " AFTER", " ONCE", " BECAUSE",
                          "AFTER ", "ONCE ", "BECAUSE ",
                          " AFTER ", " ONCE ", " BECAUSE "
                          ]
    EqualPatternsWords = ["meanwhile", "meantime", "at the same time",
                          " meanwhile", " meantime", " at the same time",
                          "meanwhile ", "meantime ", "at the same time ",
                          " meanwhile ", " meantime ", " at the same time ",
                          "Meanwhile", "Meantime", "At the same time",
                          " Meanwhile", " Meantime", " At the same time",
                          "Meanwhile ", "Meantime ", "At the same time ",
                          " Meanwhile ", " Meantime ", " At the same time ",
                          "MEANWHILE", "MEANTIME", "AT THE SAME TIME",
                          " MEANWHILE", " MEANTIME", " AT THE SAME TIME",
                          "MEANWHILE ", "MEANTIME ", "AT THE SAME TIME ",
                          " MEANWHILE ", " MEANTIME ", " AT THE SAME TIME "
                          ]

    BeforePatternsID = []
    for ws in BeforePatternsWords:
        encoded = tokenizer.encode(ws)
        BeforePatternsID.append(encoded[1:-1])

    AfterPatternsID = []
    for ws in AfterPatternsWords:
        encoded = tokenizer.encode(ws)
        AfterPatternsID.append(encoded[1:-1])

    EqualPatternsID = []
    for ws in EqualPatternsWords:
        encoded = tokenizer.encode(ws)
        EqualPatternsID.append(encoded[1:-1])

    bs = len(x_sent)
    labels = []
    for i in range(bs):
        sent, x_possition_new, y_possition_new = augment_target(x_sent[i], y_sent[i], x_sent_id[i], y_sent_id[i],
                                                                   x_position[i], y_position[i])


        if x_possition_new < y_possition_new:
            midwords = sent[x_possition_new: y_possition_new]
        elif x_possition_new > y_possition_new:
            midwords = sent[y_possition_new: x_possition_new]
        else:
            midwords = []

        label = 3
        for p in BeforePatternsID:
            for i in range(len(midwords) - len(p) + 1):
                if midwords[i: i + len(p)] == p:
                    label = 0

        for p in AfterPatternsWords:
            for i in range(len(midwords) - len(p) + 1):
                if midwords[i: i + len(p)] == p:
                    label = 1

        for p in EqualPatternsID:
            for i in range(len(midwords) - len(p) + 1):
                if midwords[i: i + len(p)] == p:
                    label = 2

        labels.append(label)

    return labels
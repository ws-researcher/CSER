import os
from itertools import combinations
from pathlib import Path

import torch
from lxml import etree

from SourceCode.utils.constant import hieve_label_dict, temp_label_map, MATRES_timebank, MATRES_aquaint, \
    MATRES_platinum, tdd_label_dict, tbd_label_dict
from SourceCode.utils.tools import tokenized_to_origin_span, span_SENT_to_DOC, sent_id_lookup, id_lookup, nlp, \
    RoBERTa_list, tokenizer, read_evaluation_file
from SourceCode.utils.constant import *
torch.manual_seed(1741)
import random
random.seed(1741)
import numpy as np
np.random.seed(1741)
import bs4
import xml.etree.ElementTree as ET
from collections import defaultdict
from spacy.util import raise_error
# from nltk import sent_tokenize
from bs4 import BeautifulSoup as Soup
import csv
from trankit import Pipeline
p = Pipeline('english', cache_dir='/data/ws/cache/trankit')



# =========================
#       HiEve Reader
# =========================
def tsvx_reader(dir_name, file_name):
    my_dict = {}
    my_dict["doc_id"] = file_name.replace(".tsvx", "") # e.g., article-10901.tsvx
    my_dict["event_dict"] = {}
    my_dict["sentences"] = []
    my_dict["relation_dict"] = {}
    
    # Read tsvx file
    for line in open(dir_name + file_name, encoding='UTF-8'):
        line = line.split('\t')
        if line[0] == 'Text':
            my_dict["doc_content"] = line[1]
        elif line[0] == 'Event':
            end_char = int(line[4]) + len(line[2]) - 1
            my_dict["event_dict"][int(line[1])] = {"mention": line[2], "start_char": int(line[4]), "end_char": end_char} 
            # keys to be added later: sent_id & subword_id
        elif line[0] == 'Relation':
            event_id1 = int(line[1])
            event_id2 = int(line[2])
            rel = hieve_label_dict[line[3]]
            my_dict["relation_dict"][(event_id1, event_id2)] = rel
        else:
            raise ValueError("Reading a file not in HiEve tsvx format...")
    
    # Split document into sentences
    sent_tokenized_text = [str(sent) for sent in nlp(my_dict["doc_content"]).sents]
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    count_sent = 0
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        count_sent += 1
        spacy_token = nlp(sent_dict["content"])
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        sent_dict["tag"] = []
        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token.text)
            sent_dict["pos"].append(token.pos_)
            sent_dict["tag"].append(token.tag_)
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])

        # RoBERTa tokenizer
        sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
        sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
        RoBERTa_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])
            
        sent_dict["roberta_subword_span_DOC"] = \
        span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])
        
        sent_dict["roberta_subword_pos"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_pos"].append("None")
            else:
                sent_dict["roberta_subword_pos"].append(sent_dict["pos"][token_id])

        sent_dict["roberta_subword_tag"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_tag"].append("None")
            else:
                sent_dict["roberta_subword_tag"].append(sent_dict["tag"][token_id])
        
        my_dict["sentences"].append(sent_dict)
    
    # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = sent_id_lookup(my_dict, event_dict["start_char"], event_dict["end_char"])
        my_dict["event_dict"][event_id]["token_id"] = id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
        my_dict["event_dict"][event_id]["roberta_subword_id"] = poss = \
        id_lookup(my_dict["sentences"][sent_id]["roberta_subword_span_DOC"], event_dict["start_char"]) + 1 # sentence is added <s> token
        # try:
        mention = event_dict["mention"]
        sub_word_id = my_dict["sentences"][sent_id]["roberta_subword_to_ID"][poss]
        sub_word = tokenizer.decode([sub_word_id])
        assert sub_word.strip() in mention

    return my_dict

# ========================================
#        MATRES: read relation file
# ========================================
# MATRES has separate text files and relation files
# We first read relation files
eiid_to_event_trigger = {}
eiid_pair_to_label = {} 

def matres_reader(matres_file):
    with open(matres_file, 'r', encoding='UTF-8') as f:
        content = f.read().split('\n')
        for i, rel in enumerate(content):
            rel = rel.split("\t")
            fname = "MATRES" + "_" + rel[0]
            trigger1 = rel[1]
            trigger2 = rel[2]
            eiid1 = int(rel[3])
            eiid2 = int(rel[4])
            tempRel = temp_label_map[rel[5]]

            if fname not in eiid_to_event_trigger:
                eiid_to_event_trigger[fname] = {}
                eiid_pair_to_label[fname] = {}
            eiid_pair_to_label[fname][(eiid1, eiid2)] = tempRel
            if eiid1 not in eiid_to_event_trigger[fname].keys():
                eiid_to_event_trigger[fname][eiid1] = trigger1
            if eiid2 not in eiid_to_event_trigger[fname].keys():
                eiid_to_event_trigger[fname][eiid2] = trigger2

matres_reader(MATRES_timebank)
matres_reader(MATRES_aquaint)
matres_reader(MATRES_platinum)

def tml_reader(dir_name, file_name):
    my_dict = {}
    my_dict["event_dict"] = {}
    my_dict["eID_dict"] = {}
    # my_dict["doc_id"] = file_name.replace(".tml", "")
    my_dict["doc_id"] = "MATRES" + "_" + file_name.replace(".tml", "")
    # e.g., file_name = "ABC19980108.1830.0711.tml"
    # dir_name = '/shared/why16gzl/logic_driven/EMNLP-2020/MATRES/TBAQ-cleaned/TimeBank/'
    tree = ET.parse(dir_name + file_name)
    root = tree.getroot()
    MY_STRING = str(ET.tostring(root))
    # ================================================
    # Load the lines involving event information first
    # ================================================
    for makeinstance in root.findall('MAKEINSTANCE'):
        instance_str = str(ET.tostring(makeinstance)).split(" ")
        try:
            assert instance_str[3].split("=")[0] == "eventID"
            assert instance_str[2].split("=")[0] == "eiid"
            eiid = int(instance_str[2].split("=")[1].replace("\"", "")[2:])
            eID = instance_str[3].split("=")[1].replace("\"", "")
        except:
            for i in instance_str:
                if i.split("=")[0] == "eventID":
                    eID = i.split("=")[1].replace("\"", "")
                if i.split("=")[0] == "eiid":
                    eiid = int(i.split("=")[1].replace("\"", "")[2:])
        # Not all document in the dataset contributes relation pairs in MATRES
        # Not all events in a document constitute relation pairs in MATRES
        if my_dict["doc_id"] in eiid_to_event_trigger.keys():
            if eiid in eiid_to_event_trigger[my_dict["doc_id"]].keys():
                my_dict["event_dict"][eiid] = {"eID": eID, "mention": eiid_to_event_trigger[my_dict["doc_id"]][eiid]}
                my_dict["eID_dict"][eID] = {"eiid": eiid}
        
    # ==================================
    #              Load Text
    # ==================================
    start = MY_STRING.find("<TEXT>") + 6
    end = MY_STRING.find("</TEXT>")
    MY_TEXT = MY_STRING[start:end]
    while MY_TEXT[0] == " ":
        MY_TEXT = MY_TEXT[1:]
    MY_TEXT = MY_TEXT.replace("\\n", " ")
    MY_TEXT = MY_TEXT.replace("\\'", "\'")
    MY_TEXT = MY_TEXT.replace("  ", " ")
    MY_TEXT = MY_TEXT.replace(" ...", "...")
    
    # ========================================================
    #    Load position of events, in the meantime replacing 
    #    "<EVENT eid="e1" class="OCCURRENCE">turning</EVENT>"
    #    with "turning"
    # ========================================================
    event_dict = dict()
    while MY_TEXT.find("<") != -1:
        start = MY_TEXT.find("<")
        end = MY_TEXT.find(">")
        if MY_TEXT[start + 1] == "E":
            event_description = MY_TEXT[start:end].split(" ")
            # print(event_description)
            # eID = (event_description[1].split("="))[1].replace("\"", "")
            # print(eID)
            for item in event_description:
                if item.startswith("eid"):
                    eID = (item.split("="))[1].replace("\"", "")
                if item.startswith("class"):
                    eClass = (item.split("="))[1].replace("\"", "")
            MY_TEXT = MY_TEXT[:start] + MY_TEXT[(end + 1):]
            if eID in my_dict["eID_dict"].keys():
                eiid = my_dict['eID_dict'][eID]['eiid']
                my_dict["event_dict"][eiid]["start_char"] = start # loading position of events
                end = start + len(my_dict["event_dict"][eiid]['mention']) - 1
                my_dict['event_dict'][eiid]['end_char'] = end
                my_dict['event_dict'][eiid]['Class'] = eClass
                event_dict[(start, end)] = eClass
        else:
            MY_TEXT = MY_TEXT[:start] + MY_TEXT[(end + 1):]
    
    # =====================================
    # Enter the routine for text processing
    # =====================================
    my_dict["doc_content"] = MY_TEXT
    my_dict["sentences"] = []
    my_dict["is_event_label"] = []
    my_dict["event_label"] = []
    my_dict["relation_dict"] = {}
    sent_tokenized_text = [str(sent) for sent in nlp(my_dict["doc_content"]).sents]
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    count_sent = 0
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        count_sent += 1
        spacy_token = nlp(sent_dict["content"])
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        sent_dict["tag"] = []
        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token.text)
            sent_dict["pos"].append(token.pos_)
            sent_dict["tag"].append(token.tag_)
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
        span_sent_to_DOC = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])
        sent_dict["token_span_DOC"] = span_sent_to_DOC
        event_label_list = [event_dict.get((s, e)) if (s, e) in event_dict.keys() else "None" for s, e in span_sent_to_DOC]
        is_event_label_list = [1 if (s, e) in event_dict.keys() else 0 for s, e in span_sent_to_DOC]
        my_dict["event_label"].append(event_label_list)
        my_dict["is_event_label"].append(is_event_label_list)

        # RoBERTa tokenizer
        sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
        sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
        RoBERTa_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])
            
        sent_dict["roberta_subword_span_DOC"] = \
        span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])
        
        sent_dict["roberta_subword_pos"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_pos"].append("None")
            else:
                sent_dict["roberta_subword_pos"].append(sent_dict["pos"][token_id])

        sent_dict["roberta_subword_tag"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_tag"].append("None")
            else:
                sent_dict["roberta_subword_tag"].append(sent_dict["tag"][token_id])
        
        my_dict["sentences"].append(sent_dict)
        
        # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():
        # print(event_id)
        # print(event_dict)
        # try:
        assert str(my_dict["doc_content"][event_dict["start_char"]:event_dict["end_char"] + 1]).lower() == str(event_dict["mention"]).lower()
        assert str(my_dict["doc_content"][event_dict["start_char"]:event_dict["end_char"] + 1]).strip() != ""
        # except:
        #     print("doc: ", my_dict["doc_content"][event_dict["start_char"]:event_dict["end_char"]])
        #     print("mention: ", event_dict["mention"])
        #     print(my_dict["doc_id"])
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = \
        sent_id_lookup(my_dict, event_dict["start_char"])
        my_dict["event_dict"][event_id]["token_id"] = \
        id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
        my_dict["event_dict"][event_id]["roberta_subword_id"] = poss = \
        id_lookup(my_dict["sentences"][sent_id]["roberta_subword_span_DOC"], event_dict["start_char"]) + 1 # sentence is added <s> token

        mention = event_dict["mention"]
        sub_word_id = my_dict["sentences"][sent_id]["roberta_subword_to_ID"][poss]
        sub_word = tokenizer.decode([sub_word_id])
        assert sub_word.strip() in mention

    if eiid_pair_to_label.get(my_dict['doc_id']) == None:
        return None
    relation_dict = eiid_pair_to_label[my_dict['doc_id']]
    my_dict['relation_dict'] = relation_dict

    return my_dict


def i2b2_xml_reader(dir_name, file_name):
    my_dict = {}
    my_dict["event_dict"] = {}
    my_dict["eID_dict"] = {}
    my_dict["doc_id"] = "I2B2" + "_" + file_name.replace(".xml", "")
    # e.g., file_name = "ABC19980108.1830.0711.tml"
    # dir_name = '/shared/why16gzl/logic_driven/EMNLP-2020/MATRES/TBAQ-cleaned/TimeBank/'

    text_path = Path(dir_name + file_name)
    with text_path.open(encoding='utf-8') as f:
        xml_tree = etree.parse(f)
        all_text = xml_tree.xpath('.//TEXT')[0].text
        events = xml_tree.xpath('.//EVENT')
        relations = xml_tree.xpath('.//TLINK')

    # ================================================
    # Load the lines involving event information first
    # ================================================
    event_dict = dict()
    for makeinstance in events:
        instance_str = makeinstance.attrib
        eid = instance_str["id"]
        start = int(instance_str["start"])
        end = int(instance_str["end"]) - 1
        eClass = instance_str["type"]
        mention = instance_str["text"]
        my_dict["event_dict"][eid] = {"eID": instance_str["id"], "mention": mention,
                                      "start_char": start, 'end_char': end,
                                      'Class': eClass}
        event_dict[(start, end)] = eClass

    # =====================================
    # Enter the routine for text processing
    # =====================================
    my_dict["doc_content"] = all_text
    my_dict["sentences"] = []
    my_dict["is_event_label"] = []
    my_dict["event_label"] = []
    my_dict["relation_dict"] = {}
    sent_tokenized_text = [str(sent) for sent in nlp(my_dict["doc_content"]).sents]
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    count_sent = 0
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        count_sent += 1
        spacy_token = nlp(sent_dict["content"])
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        sent_dict["tag"] = []
        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token.text)
            sent_dict["pos"].append(token.pos_)
            sent_dict["tag"].append(token.tag_)
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
        span_sent_to_DOC = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])
        sent_dict["token_span_DOC"] = span_sent_to_DOC

        event_dict_spilt = dict()
        for event_span in event_dict.keys():
            for i in range(len(span_sent_to_DOC)):
                s = span_sent_to_DOC[i][0]
                e = span_sent_to_DOC[i][1]
                if s >= event_span[0] and e <= event_span[1]:
                    spans_list = event_dict_spilt.get(event_span, [])
                    spans_list.append((s, e))
                    event_dict_spilt[event_span] = spans_list

        event_label_list = []
        for s, e in span_sent_to_DOC:
            flag = True
            for key in event_dict_spilt.keys():
                for i, v in enumerate(event_dict_spilt.get(key)):
                    if i == 0 and (s, e) == v:
                        label = "B-" + event_dict.get(key)
                        event_label_list.append(label)
                        flag = False
                    elif i > 0 and (s, e) == v:
                        label = "O-" + event_dict.get(key)
                        event_label_list.append(label)
                        flag = False
            if flag:
                event_label_list.append("None")

        is_event_label_list = [1 if label != "None" else 0 for label in event_label_list]
        my_dict["event_label"].append(event_label_list)
        my_dict["is_event_label"].append(is_event_label_list)

        # RoBERTa tokenizer
        sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
        sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
            RoBERTa_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])

        sent_dict["roberta_subword_span_DOC"] = \
            span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])

        sent_dict["roberta_subword_pos"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_pos"].append("None")
            else:
                sent_dict["roberta_subword_pos"].append(sent_dict["pos"][token_id])

        sent_dict["roberta_subword_tag"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_tag"].append("None")
            else:
                sent_dict["roberta_subword_tag"].append(sent_dict["tag"][token_id])

        my_dict["sentences"].append(sent_dict)

        # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():

        # a = str(my_dict["doc_content"][event_dict["start_char"]:event_dict["end_char"] + 1]).lower()
        # b = str( event_dict["mention"]).lower()
        # assert str(my_dict["doc_content"][event_dict["start_char"]:event_dict["end_char"] + 1]).lower() == str(
        #     event_dict["mention"]).lower()
        # assert str(my_dict["doc_content"][event_dict["start_char"]:event_dict["end_char"] + 1]).strip() != ""

        my_dict["event_dict"][event_id]["sent_id"] = sent_id = \
            sent_id_lookup(my_dict, event_dict["start_char"])
        my_dict["event_dict"][event_id]["token_id"] = \
            id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
        my_dict["event_dict"][event_id]["roberta_subword_id"] = poss = \
            id_lookup(my_dict["sentences"][sent_id]["roberta_subword_span_DOC"],
                      event_dict["start_char"]) + 1  # sentence is added <s> token

        mention = event_dict["mention"]
        sub_word_id = my_dict["sentences"][sent_id]["roberta_subword_to_ID"][poss]
        sub_word = tokenizer.decode([sub_word_id])
        # assert sub_word.strip() in mention

    my_dict['relation_dict'] = {}
    for item in relations:
        attr:dict = item.attrib
        e1_id = attr.get('fromID')
        e2_id = attr.get('toID')
        rel = attr.get('type')
        if e1_id[0] == "E" and  e2_id[0] == "E" and rel != "":
            rel_id = i2b2_label_dict.get(rel)
            my_dict['relation_dict'][(e1_id, e2_id)] = rel_id
            if rel_id == None:
                print(item)

    return my_dict


# =========================
#  TimeBank-densen Reader
# =========================
def tbd_reader(tdd_file):
    mapping = []
    with open(tdd_file, 'r', encoding='UTF-8') as f:
        read_tsv = csv.reader(f, delimiter="\t")
        for item in read_tsv:
            if "" in item:
                item.remove("")
            if item[1][0] == "e" and item[2][0] == "e":
                mapping.append(item)
    return mapping

def doc_mapping(corpus):
    mapping = defaultdict(list)
    for rel in corpus:
        _rel = (rel[1], rel[2], rel[3])
        mapping["TBD" +"_"+ rel[0]].append(_rel)
    return dict(mapping)

TBD = tbd_reader(TBD)
TBD = doc_mapping(TBD)

def tbd_tml_reader(dir_name, file_name):
    my_dict = {}
    my_dict["event_dict"] = {}
    eid_to_eiid = {}
    my_dict["doc_id"] = "TBD" +"_"+ file_name.replace(".tml", "")
    try:
        xml_dom = Soup(open(dir_name + file_name, 'r', encoding='UTF-8'), 'lxml')
    except:
        print("Can't load this file: {} T_T". format(dir_name + file_name))
        return None

    for item in xml_dom.find_all('makeinstance'):
        eiid = item.attrs['eiid']
        eid = item.attrs['eventid']
        my_dict['event_dict'][eid] = {}
        my_dict['event_dict'][eid]['tense'] = item.attrs['tense']
        my_dict['event_dict'][eid]['aspect'] = item.attrs['aspect']
        my_dict['event_dict'][eid]['polarity'] = item.attrs.get('polarity')
        eid_to_eiid[eid] = eiid

    raw_content = xml_dom.find('text')
    content = ''
    pointer = 0
    for item in raw_content.contents:

        if type(item) == bs4.element.Tag and item.name == 'event':
            eid = item.attrs['eid']
            eiid = eid_to_eiid[eid]
            my_dict['event_dict'][eid]['mention'] = item.text
            my_dict['event_dict'][eid]['class'] = item.attrs['class']
            my_dict['event_dict'][eid]['start_char'] = pointer
            end_char = pointer + len(my_dict["event_dict"][eid]['mention']) - 1
            my_dict['event_dict'][eid]['end_char'] = end_char

        if type(item) == bs4.element.NavigableString:
            content += str(item)
            pointer += len(str(item))
        else:
            content += item.text
            pointer += len(item.text)

    my_dict["doc_content"] = content
    my_dict["sentences"] = []
    sent_tokenized_text = [str(sent) for sent in nlp(my_dict["doc_content"]).sents]
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    count_sent = 0
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        count_sent += 1
        spacy_token = nlp(sent_dict["content"])
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        sent_dict["tag"] = []
        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token.text)
            sent_dict["pos"].append(token.pos_)
            sent_dict["tag"].append(token.tag_)
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])

        # RoBERTa tokenizer
        sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
        sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
        RoBERTa_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])
            
        sent_dict["roberta_subword_span_DOC"] = \
        span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])
        
        sent_dict["roberta_subword_pos"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_pos"].append("None")
            else:
                sent_dict["roberta_subword_pos"].append(sent_dict["pos"][token_id])

        sent_dict["roberta_subword_tag"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_tag"].append("None")
            else:
                sent_dict["roberta_subword_tag"].append(sent_dict["tag"][token_id])
        
        my_dict["sentences"].append(sent_dict)
        
        # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():
        # print(event_id)
        # print(event_dict)
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = \
        sent_id_lookup(my_dict, event_dict["start_char"])
        my_dict["event_dict"][event_id]["token_id"] = \
        id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
        my_dict["event_dict"][event_id]["roberta_subword_id"] = poss = \
        id_lookup(my_dict["sentences"][sent_id]["roberta_subword_span_DOC"], event_dict["start_char"]) + 1 # sentence is added <s> token
        # try:
        mention = event_dict["mention"]
        sub_word_id = my_dict["sentences"][sent_id]["roberta_subword_to_ID"][poss]
        sub_word = tokenizer.decode([sub_word_id])
        assert sub_word.strip() in mention

    # my_dict['relation_dict'] = {}
    # for item in xml_dom.find_all('tlink'):
    #     attr:dict = item.attrs
    #     e1_id = attr.get('eventinstanceid')
    #     e2_id = attr.get('relatedtoeventinstance')
    #     if  e1_id != None and  e2_id != None:
    #         rel = attr.get('reltype')
    #         rel_id = tbd_label_dict.get(rel)
    #         my_dict['relation_dict'][(e1_id, e2_id)] = rel_id
    #         if rel_id == None:
    #             print(item)

    my_dict['relation_dict'] = {}
    rel_list = TBD.get(my_dict["doc_id"])
    if rel_list == None:
        return None
    eids = my_dict['event_dict'].keys()

    # print(rel_list)
    for item in rel_list:
        eid1, eid2, rel = item
        # print(item)
        if eid1 in eids and eid2 in eids:
            # print(tdd_label_dict[rel])
            my_dict['relation_dict'][(eid1, eid2)] = tbd_label_dict[rel]

    return my_dict


# =========================
#  Causal-TimeBank Reader
# =========================
def CTB_tml_reader(dir_name, file_name):
    my_dict = {}
    my_dict["event_dict"] = {}
    eid_to_eiid = {}
    my_dict["doc_id"] = "CTB" + "_" + file_name.replace(".tml", "")
    try:
        xml_dom = Soup(open(dir_name + file_name, 'r', encoding='UTF-8'), 'lxml')
    except:
        print("Can't load this file: {} T_T".format(dir_name + file_name))
        return None

    for item in xml_dom.find_all('makeinstance'):
        eiid = item.attrs['eiid']
        eid = item.attrs['eventid']
        my_dict['event_dict'][eiid] = {}
        my_dict['event_dict'][eiid]['tense'] = item.attrs['tense']
        my_dict['event_dict'][eiid]['aspect'] = item.attrs['aspect']
        my_dict['event_dict'][eiid]['polarity'] = item.attrs.get('polarity')
        eid_to_eiid[eid] = eiid

    raw_content = xml_dom.find('text')
    content = ''
    pointer = 0
    for item in raw_content.contents:

        if type(item) == bs4.element.Tag and item.name == 'event':
            eid = item.attrs['eid']
            if eid not in eid_to_eiid.keys():
                continue
            eiid = eid_to_eiid[eid]
            my_dict['event_dict'][eiid]['mention'] = item.text
            my_dict['event_dict'][eiid]['class'] = item.attrs['class']
            my_dict['event_dict'][eiid]['start_char'] = pointer
            end_char = pointer + len(my_dict["event_dict"][eiid]['mention']) - 1
            my_dict['event_dict'][eiid]['end_char'] = end_char

        if type(item) == bs4.element.Tag and item.name == 'c-signal':
            for it in item.contents:
                if type(it) == bs4.element.Tag and it.name == 'event':
                    eid = it.attrs['eid']
                    if eid not in eid_to_eiid.keys():
                        continue
                    eiid = eid_to_eiid[eid]
                    my_dict['event_dict'][eiid]['mention'] = it.text
                    my_dict['event_dict'][eiid]['class'] = it.attrs['class']
                    my_dict['event_dict'][eiid]['start_char'] = pointer
                    end_char = pointer + len(my_dict["event_dict"][eiid]['mention']) - 1
                    my_dict['event_dict'][eiid]['end_char'] = end_char

                if type(it) == bs4.element.NavigableString:
                    content += str(it)
                    pointer += len(str(it))
                else:
                    content += it.text
                    pointer += len(it.text)

        if type(item) == bs4.element.NavigableString:
            content += str(item)
            pointer += len(str(item))
        else:
            content += item.text
            pointer += len(item.text)

    my_dict["doc_content"] = content
    my_dict["sentences"] = []
    sent_tokenized_text = [str(sent) for sent in nlp(my_dict["doc_content"]).sents]
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    count_sent = 0
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        count_sent += 1
        spacy_token = nlp(sent_dict["content"])
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        sent_dict["tag"] = []
        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token.text)
            sent_dict["pos"].append(token.pos_)
            sent_dict["tag"].append(token.tag_)
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])

        # RoBERTa tokenizer
        sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
        sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
            RoBERTa_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])

        sent_dict["roberta_subword_span_DOC"] = \
            span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])

        sent_dict["roberta_subword_pos"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_pos"].append("None")
            else:
                sent_dict["roberta_subword_pos"].append(sent_dict["pos"][token_id])

        sent_dict["roberta_subword_tag"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_tag"].append("None")
            else:
                sent_dict["roberta_subword_tag"].append(sent_dict["tag"][token_id])

        my_dict["sentences"].append(sent_dict)

        # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():
        # print(event_id)
        # print(event_dict)
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = \
            sent_id_lookup(my_dict, event_dict["start_char"])
        my_dict["event_dict"][event_id]["token_id"] = \
            id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
        my_dict["event_dict"][event_id]["roberta_subword_id"] = poss = \
            id_lookup(my_dict["sentences"][sent_id]["roberta_subword_span_DOC"],
                      event_dict["start_char"]) + 1  # sentence is added <s> token
        # try:
        mention = event_dict["mention"]
        sub_word_id = my_dict["sentences"][sent_id]["roberta_subword_to_ID"][poss]
        sub_word = tokenizer.decode([sub_word_id])
        assert sub_word.strip() in mention

    my_dict['relation_dict'] = {}
    for item in xml_dom.find_all('tlink'):
        attr: dict = item.attrs
        e1_id = attr.get('eventinstanceid')
        e2_id = attr.get('relatedtoeventinstance')
        if e1_id != None and e2_id != None:
            rel = attr.get('reltype')
            rel_id = CTB_label_dict.get(rel)
            if rel == "SIMULTANEOUS" or rel == "IS_INCLUDED":
                my_dict['relation_dict'][(e1_id, e2_id)] = rel_id

    for item in xml_dom.find_all('clink'):
        attr: dict = item.attrs
        e1_id = attr.get('eventinstanceid')
        e2_id = attr.get('relatedtoeventinstance')
        if e1_id != None and e2_id != None:
            rel = "CAUSAL"
            rel_id = CTB_label_dict.get(rel)
            if rel_id == None:
                print(item)
                continue
            my_dict['relation_dict'][(e1_id, e2_id)] = rel_id

    return my_dict

# =========================
#  EventStoryLine Reader
# =========================
def esl_reader(ecbtopic, ecbstartopic, evaluationtopic, evaluationcoreftopic):
    if os.path.isdir(ecbtopic) and os.path.isdir(ecbstartopic) and os.path.isdir(evaluationtopic):
        if ecbtopic[-1] != '/':
            ecbtopic += '/'
        if ecbstartopic[-1] != '/':
            ecbstartopic += '/'
        if evaluationtopic[-1] != '/':
            evaluationtopic += '/'
        if evaluationcoreftopic[-1] != '/':
            evaluationcoreftopic += '/'


    topic_corpus = []
    for f in os.listdir(ecbstartopic):

        if f.endswith('xml'):
            ecb_file = f
            star_file = ecbstartopic + f
            evaluate_file = evaluationtopic + ecb_file.replace(".xml", "") + ".xml"
            evaluate_coref_file = evaluationcoreftopic + f

            my_dict = {}
            my_dict["event_dict"] = {}
            mid_to_tid = {}
            tid_to_mid = {}
            my_dict["doc_id"] = "ESL" + "_" + ecb_file.replace(".xml", "")

            evaluation_data = read_evaluation_file(evaluate_file)

            # try:
            #     ecb = etree.parse(ecbtopic + ecb_file, etree.XMLParser(remove_blank_text=True))
            #     ecb_root = ecb.getroot()
            #     ecb_root.getchildren()
            # except:
            #     print("Can't load this file: {} T_T".format(ecbtopic + ecb_file))
            #     return None

            try:
                ecbstar = etree.parse(star_file, etree.XMLParser(remove_blank_text=True))
                ecbstar_root = ecbstar.getroot()
                ecbstar_root.getchildren()
            except:
                print("Can't load this file: {} T_T".format(ecbtopic + ecb_file))
                return None


            content = ''
            pointer = 0
            token_dict = {}
            for elem in ecbstar_root.findall('token'):
                tid = elem.get("t_id")
                sid = elem.get("sentence")
                token_dict[tid] = {}
                token_dict[tid]["text"] = elem.text
                token_dict[tid]["sid"] = sid
                token_dict[tid]["start_char"] = pointer
                end_char = pointer + len(elem.text)
                token_dict[tid]["end_char"] = end_char

                content += str(elem.text) + " "
                pointer += len(str(elem.text)) + 1

            for elem in ecbstar_root.findall('Markables/'):
                if elem.tag.startswith("ACTION") or elem.tag.startswith("NEG_ACTION"):
                    event_mention_id = elem.get('m_id', 'nothing')
                    token_mention_ids = []
                    for token_id in elem.findall('token_anchor'):
                        token_mention_id = token_id.get('t_id', 'nothing')
                        token_mention_ids.append(token_mention_id)

                    if len(token_mention_ids) == 0:
                        continue

                    my_dict['event_dict'][event_mention_id] = {}
                    token_mention_ids.sort()
                    mid_to_tid[event_mention_id] = "_".join(token_mention_ids)
                    tid_to_mid["_".join(token_mention_ids)] = event_mention_id

                    mentions = ""
                    for tid in token_mention_ids:
                        if tid not in token_dict.keys():
                            continue
                        mentions += token_dict[tid].get("text")

                    start_char = token_dict.get(token_mention_ids[0]).get("start_char")
                    end_char = token_dict.get(token_mention_ids[-1]).get("end_char")
                    eventClass = str(elem.tag)[7:]

                    my_dict['event_dict'][event_mention_id]['mention'] = mentions
                    my_dict['event_dict'][event_mention_id]['class'] = eventClass
                    my_dict['event_dict'][event_mention_id]['start_char'] = start_char
                    my_dict['event_dict'][event_mention_id]['end_char'] = end_char

            my_dict['relation_dict'] = {}
            for e1 in my_dict['event_dict'].keys():
                for e2 in my_dict['event_dict'].keys():
                    t1 = mid_to_tid.get(e1).split("_")[0]
                    t2 = mid_to_tid.get(e2).split("_")[0]
                    s1 = token_dict.get(t1).get("sid")
                    s2 = token_dict.get(t2).get("sid")
                    if s1 == s2 and e1 != e2:
                        rel_id = ESL_label_dict.get("NONE")
                        my_dict['relation_dict'][(e1, e2)] = rel_id

            # for elem in ecbstar_root.findall('Relations/'):
            #     if elem.tag == "PLOT_LINK":
            #         source = elem.find('source').get('m_id', 'null')
            #         target = elem.find('target').get('m_id', 'null')
            #         rel = elem.get('relType', 'null')
            #         rel_id = ESL_label_dict.get(rel)
            #         my_dict['relation_dict'][(source, target)] = rel_id

            for event_pair in evaluation_data:
                source = tid_to_mid.get(event_pair[0], 'null')
                target = tid_to_mid.get(event_pair[1], 'null')
                relType = event_pair[2]
                rel_id = ESL_label_dict.get(relType)
                assert source != 'null' or target != 'null'
                my_dict['relation_dict'][(source, target)] = rel_id

            my_dict["doc_content"] = content
            my_dict["sentences"] = []
            sent_tokenized_text = [str(sent) for sent in nlp(my_dict["doc_content"]).sents]
            sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
            count_sent = 0
            for sent in sent_tokenized_text:
                sent_dict = {}
                sent_dict["sent_id"] = count_sent
                sent_dict["content"] = sent
                sent_dict["sent_start_char"] = sent_span[count_sent][0]
                sent_dict["sent_end_char"] = sent_span[count_sent][1]
                count_sent += 1
                spacy_token = nlp(sent_dict["content"])
                sent_dict["tokens"] = []
                sent_dict["pos"] = []
                sent_dict["tag"] = []
                # spaCy-tokenized tokens & Part-Of-Speech Tagging
                for token in spacy_token:
                    sent_dict["tokens"].append(token.text)
                    sent_dict["pos"].append(token.pos_)
                    sent_dict["tag"].append(token.tag_)
                sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
                sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])

                # RoBERTa tokenizer
                sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
                sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
                    RoBERTa_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])

                sent_dict["roberta_subword_span_DOC"] = \
                    span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])

                sent_dict["roberta_subword_pos"] = []
                for token_id in sent_dict["roberta_subword_map"]:
                    if token_id == -1 or token_id is None:
                        sent_dict["roberta_subword_pos"].append("None")
                    else:
                        sent_dict["roberta_subword_pos"].append(sent_dict["pos"][token_id])

                sent_dict["roberta_subword_tag"] = []
                for token_id in sent_dict["roberta_subword_map"]:
                    if token_id == -1 or token_id is None:
                        sent_dict["roberta_subword_tag"].append("None")
                    else:
                        sent_dict["roberta_subword_tag"].append(sent_dict["tag"][token_id])

                my_dict["sentences"].append(sent_dict)

                # Add sent_id as an attribute of event
            for event_id, event_dict in my_dict["event_dict"].items():
                # print(event_id)
                # print(event_dict)
                my_dict["event_dict"][event_id]["sent_id"] = sent_id = \
                    sent_id_lookup(my_dict, event_dict["start_char"])
                my_dict["event_dict"][event_id]["token_id"] = \
                    id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
                my_dict["event_dict"][event_id]["roberta_subword_id"] = poss = \
                    id_lookup(my_dict["sentences"][sent_id]["roberta_subword_span_DOC"],
                              event_dict["start_char"]) + 1  # sentence is added <s> token
                # try:
                mention = event_dict["mention"]
                sub_word_id = my_dict["sentences"][sent_id]["roberta_subword_to_ID"][poss]
                sub_word = tokenizer.decode([sub_word_id])
                assert sub_word.strip() in mention

            topic_corpus.append(my_dict)

    return topic_corpus

# =========================
#   TDDiscourse - Reader
# =========================
def tdd_reader(tdd_file):
    mapping = []
    with open(tdd_file, 'r', encoding='UTF-8') as f:
        read_tsv = csv.reader(f, delimiter="\t")
        for item in read_tsv:
            mapping.append(item)
    return mapping

def doc_mapping(corpus, type):
    mapping = defaultdict(list)
    for rel in corpus:
        _rel = (rel[1], rel[2], rel[3])
        mapping["TDD" +"_"+ type +"_"+ rel[0]].append(_rel)
    return dict(mapping)

TDD_man_train = tdd_reader(TDD_man_train)
TDD_man_test = tdd_reader(TDD_man_test)
TDD_man_val = tdd_reader(TDD_man_val)
TDD_man = TDD_man_train + TDD_man_val + TDD_man_test
TDD_man = doc_mapping(TDD_man, "man")

TDD_auto_train = tdd_reader(TDD_auto_train)
TDD_auto_test = tdd_reader(TDD_auto_test)
TDD_auto_val = tdd_reader(TDD_auto_val)
TDD_auto = TDD_auto_train + TDD_auto_val + TDD_auto_test
TDD_auto = doc_mapping(TDD_auto, "auto")


def tdd_tml_reader(dir_name, file_name, type_doc):
    my_dict = {}
    my_dict["event_dict"] = {}
    my_dict["doc_id"] = "TDD" +"_"+ type_doc +"_"+ file_name.replace(".tml", "")
    try:
        xml_dom = Soup(open(dir_name + file_name, 'r', encoding='UTF-8'), 'lxml')
    except:
        print("Can't load this file: {} T_T". format(dir_name + file_name))
        return None
    
    for item in xml_dom.find_all('makeinstance'):
        eid = item.attrs['eventid']
        my_dict['event_dict'][eid] = {}
        my_dict['event_dict'][eid]['tense'] = item.attrs['tense']
        my_dict['event_dict'][eid]['aspect'] = item.attrs['aspect']
        my_dict['event_dict'][eid]['polarity'] = item.attrs.get('polarity')

    raw_content = xml_dom.find('text')
    content = ''
    pointer = 0
    for item in raw_content.contents:
        if type(item) == bs4.element.Tag and item.name == 'event':
            eid = item.attrs['eid']
            my_dict['event_dict'][eid]['mention'] = item.text
            my_dict['event_dict'][eid]['class'] = item.attrs['class']
            my_dict['event_dict'][eid]['start_char'] = pointer
            end_char = pointer + len(my_dict["event_dict"][eid]['mention']) - 1
            my_dict['event_dict'][eid]['end_char'] = end_char
        if type(item) == bs4.element.NavigableString:
            content += str(item)
            pointer += len(str(item))
        else:
            content += item.text
            pointer += len(item.text)
    my_dict["doc_content"] = content

    # print(my_dict["doc_content"])

    my_dict["sentences"] = []
    sent_tokenized_text = [str(sent) for sent in nlp(my_dict["doc_content"]).sents]
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    count_sent = 0
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        count_sent += 1
        spacy_token = nlp(sent_dict["content"])
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        sent_dict["tag"] = []
        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token.text)
            sent_dict["pos"].append(token.pos_)
            sent_dict["tag"].append(token.tag_)
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])

        # RoBERTa tokenizer
        sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
        sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
        RoBERTa_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])
            
        sent_dict["roberta_subword_span_DOC"] = \
        span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])
        
        sent_dict["roberta_subword_pos"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_pos"].append("None")
            else:
                sent_dict["roberta_subword_pos"].append(sent_dict["pos"][token_id])

        sent_dict["roberta_subword_tag"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_tag"].append("None")
            else:
                sent_dict["roberta_subword_tag"].append(sent_dict["tag"][token_id])
        
        my_dict["sentences"].append(sent_dict)
        
        # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():
        # print(event_id)
        # print(event_dict)
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = \
        sent_id_lookup(my_dict, event_dict["start_char"])
        my_dict["event_dict"][event_id]["token_id"] = \
        id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
        my_dict["event_dict"][event_id]["roberta_subword_id"] = poss = \
        id_lookup(my_dict["sentences"][sent_id]["roberta_subword_span_DOC"], event_dict["start_char"]) + 1 # sentence is added <s> token
        # try:
        mention = event_dict["mention"]
        sub_word_id = my_dict["sentences"][sent_id]["roberta_subword_to_ID"][poss]
        sub_word = tokenizer.decode([sub_word_id])
        assert sub_word.strip() in mention
    
    my_dict['relation_dict'] = {}
    if type_doc == 'man':
        rel_list = TDD_man.get(my_dict["doc_id"])
        if rel_list == None:
            return None
    elif type_doc == 'auto':
        rel_list = TDD_auto.get(my_dict["doc_id"])
        if rel_list == None:
            return None
    else:
        raise_error("Don't have this corpus!")
    eids = my_dict['event_dict'].keys()

    # print(rel_list)
    for item in rel_list:
        eid1, eid2, rel = item
        # print(item)
        if eid1 in eids and eid2 in eids:
            # print(tdd_label_dict[rel])
            my_dict['relation_dict'][(eid1, eid2)] = tdd_label_dict[rel]
    
    return my_dict


def mulerx_tsvx_reader(dir_name:str, file_name: str, model='RoBERTa'):
    my_dict = {}
    my_dict["doc_id"] = file_name.replace(".tsvx", "") # e.g., article-10901.tsvx
    my_dict["event_dict"] = {}
    my_dict["sentences"] = []
    my_dict["relation_dict"] = {}

    # Read tsvx file
    for line in open(dir_name + file_name, encoding='UTF-8'):
        line = line.split('\t')
        if line[0] == 'Text':
            my_dict["doc_content"] = '\t'.join(line[1:])
        elif line[0] == 'Event':
            end_char = int(line[4]) + len(line[2]) - 1
            my_dict["event_dict"][line[1]] = {"mention": line[2], "start_char": int(line[4]), "end_char": end_char} 
            # keys to be added later: sent_id & subword_id
        elif line[0] == 'Relation':
            event_id1 = line[1]
            event_id2 = line[2]
            rel = mulerx_label_dict[line[3]]
            # my_dict["relation_dict"][(event_id1, event_id2)] = rel
        else:
            raise ValueError("Reading a file not in HiEve tsvx format...")
    
    for eid1, eid2 in combinations(my_dict["event_dict"].keys(), 2):
        if my_dict["relation_dict"].get((eid1, eid2)) == None:
            my_dict["relation_dict"][(eid1, eid2)] = 3
    
    # Split document into sentences
    sent_tokenized_text = [sent['text'] for sent in p.ssplit(my_dict["doc_content"])['sentences']]
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    count_sent = 0
    for sent in sent_tokenized_text:
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        count_sent += 1
        _spacy_token = p.posdep(sent_dict["content"], is_sent=True)['tokens']
        spacy_token = []
        for token in _spacy_token:
            if token.get('expanded') == None:
                spacy_token.append(token)
            else:
                spacy_token = spacy_token + token['expanded']
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        # Trankit-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token['text'])
            sent_dict["pos"].append(token['upos'])
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])

        # RoBERTa tokenizer
        if model == 'mBERT':
            sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
            sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
            mBERT_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])
        
        if model == 'XML-R':
            sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
            sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
            XLMR_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])
        
        if model == 'RoBERTa':
            sent_dict["roberta_subword_to_ID"], sent_dict["roberta_subwords"], \
            sent_dict["roberta_subword_span_SENT"], sent_dict["roberta_subword_map"] = \
            RoBERTa_list(sent_dict["content"], sent_dict["tokens"], sent_dict["token_span_SENT"])
            
        sent_dict["roberta_subword_span_DOC"] = \
        span_SENT_to_DOC(sent_dict["roberta_subword_span_SENT"], sent_dict["sent_start_char"])
        
        sent_dict["roberta_subword_pos"] = []
        for token_id in sent_dict["roberta_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["roberta_subword_pos"].append("None")
            else:
                sent_dict["roberta_subword_pos"].append(sent_dict["pos"][token_id])
        
        my_dict["sentences"].append(sent_dict)
    
    # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = sent_id_lookup(my_dict, event_dict["start_char"], event_dict["end_char"])
        my_dict["event_dict"][event_id]["token_id"] = id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
        my_dict["event_dict"][event_id]["roberta_subword_id"] = poss = id_lookup(my_dict["sentences"][sent_id]["roberta_subword_span_DOC"], event_dict["start_char"]) + 1 # sentence is added <s> token
    
        mention: str = event_dict["mention"]
        sub_word_id = my_dict["sentences"][sent_id]["roberta_subword_to_ID"][poss]
        if model == 'mBERT':
            sub_word: str = mBERT_tokenizer.decode([sub_word_id])
            if sub_word.replace("#", '').strip() not in mention:
                try:
                    print(f'{sub_word} - {mention} - {poss} - {my_dict["sentences"][sent_id]["roberta_subwords"][poss]}')
                except:
                    print("We have wrong on picking words")
        elif model == 'XML-R':
            sub_word = xlmr_tokenizer.decode([sub_word_id])
            if sub_word.strip() not in mention:
                try:
                    print(f'{sub_word} - {mention} - {poss} - {my_dict["sentences"][sent_id]["roberta_subwords"][poss]}')
                except:
                    print("We have wrong on picking words")
        
        elif model == 'RoBERTa':
            sub_word = tokenizer.decode([sub_word_id])
            if sub_word.strip() not in mention:
                try:
                    print(f'{sub_word} - {mention} - {poss} - {my_dict["sentences"][sent_id]["roberta_subwords"][poss]}')
                except:
                    print("We have wrong on picking words")

    return my_dict

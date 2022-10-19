import torch
from sklearn.model_selection import train_test_split

from SourceCode.data_loader.reader import mulerx_tsvx_reader, tbd_tml_reader, tdd_tml_reader, tml_reader, tsvx_reader, \
    i2b2_xml_reader, CTB_tml_reader, esl_reader
from SourceCode.utils.SentenceEncoder import SentenceEncoder
from SourceCode.utils.constant import temp_label_map
from SourceCode.utils.tools import create_target, padding, pos_to_id, tag_to_id

torch.manual_seed(1741)
import random
from random import sample
random.seed(1741)
import numpy as np
np.random.seed(1741)
from collections import defaultdict
import pickle
import os
import tqdm
from itertools import combinations
import gc


class Reader(object):
    def __init__(self, type) -> None:
        super().__init__()
        self.type = type
    
    def read(self, dir_name, file_name):
        if self.type == 'tsvx':
            return tsvx_reader(dir_name, file_name)
        elif self.type == 'tml':
            return tml_reader(dir_name, file_name)
        elif self.type == 'mulerx':
            return mulerx_tsvx_reader(dir_name, file_name)
        elif self.type == 'tbd_tml':
            return tbd_tml_reader(dir_name, file_name)
        elif self.type == 'tdd_man':
            return tdd_tml_reader(dir_name, file_name, type_doc='man')
        elif self.type == 'tdd_auto':
            return tdd_tml_reader(dir_name, file_name, type_doc='auto')
        elif self.type == 'i2b2_xml':
            return i2b2_xml_reader(dir_name, file_name)
        elif self.type == 'CTB_tml':
            return CTB_tml_reader(dir_name, file_name)
        else:
            raise ValueError("We have not supported {} type yet!".format(self.type))

def load_dataset(dir_name, type):
    reader = Reader(type)
    onlyfiles = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
    corpus = []
    for file_name in tqdm.tqdm(onlyfiles):
        # if file_name == "ABC19980120.1830.0957.tml":
        #     print("s")
        # else:
        #     continue
        if type == 'i2b2_xml':
            if file_name.endswith('.xml'):
                my_dict = reader.read(dir_name, file_name)
                if my_dict != None:
                    corpus.append(my_dict)
        else:
            my_dict = reader.read(dir_name, file_name)
            if my_dict != None:
                corpus.append(my_dict)
    return corpus

def load_ESL_dataset(topics):
    version = 'v1.0'
    ECBplusTopic = '../datasets/EventStoryLine/ECB+_LREC2014/ECB+/'
    ECBstarTopic = '../datasets/EventStoryLine/annotated_data/' + version + '/'
    EvaluationTopic = '../datasets/EventStoryLine/evaluation_format/full_corpus/' + version + '/event_mentions_extended/'
    EvaluationCrofTopic = '../datasets/EventStoryLine/evaluation_format/full_corpus/' + version + '/coref_chain/'

    # topics = [f for f in os.listdir(ECBstarTopic) if os.path.isdir(os.path.join(ECBstarTopic, f))]
    corpus = []
    for topic in tqdm.tqdm(topics):
        topic = str(topic)
        dir1, dir2, dir3, dir4 = ECBplusTopic + topic, ECBstarTopic + topic, EvaluationTopic + topic, EvaluationCrofTopic + topic
        topic_corpus = esl_reader(dir1, dir2, dir3, dir4)
        if topic_corpus != None:
            corpus.extend(topic_corpus)
    return corpus


global_sent_encoder = SentenceEncoder('roberta-base')

def loader(dataset, min_ns = None, file_type=None, file_path=None, label_type=None):
    sent_encoder = global_sent_encoder

    def get_data_point(my_dict, flag):
        data = []
        eids = my_dict['event_dict'].keys()
        pair_events = list(combinations(eids, 2))
        doc_id = my_dict["doc_id"]

        for pair in pair_events:
            x, y = pair

            x_sent_id = my_dict['event_dict'][x]['sent_id']
            y_sent_id = my_dict['event_dict'][y]['sent_id']

            x_position = my_dict["event_dict"][x]["roberta_subword_id"]
            y_position = my_dict["event_dict"][y]["roberta_subword_id"]

            sents = [0]
            sents_pos = [0]
            sents_tag = [0]
            x_position_new = 0
            y_position_new = 0
            if x_sent_id < y_sent_id:
                x_position_new += x_position
                for i in range(x_sent_id, y_sent_id):
                    sent = my_dict["sentences"][i]["roberta_subword_to_ID"]
                    sent_pos = pos_to_id(my_dict["sentences"][i]["roberta_subword_pos"])
                    sent_tag = tag_to_id(my_dict["sentences"][i]["roberta_subword_tag"])
                    sents += sent[1:] + [2]
                    sents_pos += sent_pos[1:] + [0]
                    sents_tag += sent_tag[1:] + [0]
                    y_position_new += len(sent)

                sent = my_dict["sentences"][y_sent_id]["roberta_subword_to_ID"]
                sent_pos = pos_to_id(my_dict["sentences"][y_sent_id]["roberta_subword_pos"])
                sent_tag = tag_to_id(my_dict["sentences"][y_sent_id]["roberta_subword_tag"])
                sents += sent[1:] + [2]
                sents_pos += sent_pos[1:] + [0]
                sents_tag += sent_tag[1:] + [0]
                y_position_new += y_position

            elif x_sent_id >= y_sent_id:
                y_position_new += y_position
                for i in range(y_sent_id, x_sent_id):
                    sent = my_dict["sentences"][i]["roberta_subword_to_ID"]
                    sent_pos = pos_to_id(my_dict["sentences"][i]["roberta_subword_pos"])
                    sent_tag = tag_to_id(my_dict["sentences"][i]["roberta_subword_tag"])
                    sents += sent[1:] + [2]
                    sents_pos += sent_pos[1:] + [0]
                    sents_tag += sent_tag[1:] + [0]
                    x_position_new += len(sent)

                sent = my_dict["sentences"][x_sent_id]["roberta_subword_to_ID"]
                sent_pos = pos_to_id(my_dict["sentences"][x_sent_id]["roberta_subword_pos"])
                sent_tag = tag_to_id(my_dict["sentences"][x_sent_id]["roberta_subword_tag"])
                sents += sent[1:] + [2]
                sents_pos += sent_pos[1:] + [0]
                sents_tag += sent_tag[1:] + [0]
                x_position_new += x_position

            if len(sents) > 512:
                x_position_new = 0
                y_position_new = 0
                x_sent = my_dict["sentences"][x_sent_id]["roberta_subword_to_ID"]
                y_sent = my_dict["sentences"][y_sent_id]["roberta_subword_to_ID"]
                x_sent_pos = pos_to_id(my_dict["sentences"][x_sent_id]["roberta_subword_pos"])
                y_sent_pos = pos_to_id(my_dict["sentences"][y_sent_id]["roberta_subword_pos"])
                x_sent_tag = tag_to_id(my_dict["sentences"][x_sent_id]["roberta_subword_tag"])
                y_sent_tag = tag_to_id(my_dict["sentences"][y_sent_id]["roberta_subword_tag"])
                if x_sent_id < y_sent_id:
                    x_position_new = x_position
                    y_position_new = y_position + len(x_sent)
                    sents = [0] + x_sent[1:] + [2] + y_sent[1:] + [2]
                    sents_pos = [0] + x_sent_pos[1:] + [0] + y_sent_pos[1:] + [0]
                    sents_tag = [0] + x_sent_tag[1:] + [0] + y_sent_tag[1:] + [0]
                elif x_sent_id >= y_sent_id:
                    y_position_new = y_position
                    x_position_new = x_position + len(y_sent)
                    sents = [0] + y_sent[1:] + [2] + x_sent[1:] + [2]
                    sents_pos = [0] + y_sent_pos[1:] + [0] + x_sent_pos[1:] + [0]
                    sents_tag = [0] + y_sent_tag[1:] + [0] + x_sent_tag[1:] + [0]

            xy = my_dict["relation_dict"].get((x, y))
            yx = my_dict["relation_dict"].get((y, x))

            candidates = [
                [doc_id, str(x), str(y), sents, sents_pos, sents_tag, x_position_new, y_position_new, flag, xy],
                [doc_id, str(y), str(x), sents, sents_pos, sents_tag, y_position_new, x_position_new, flag, yx],
            ]

            for item in candidates:
                if item[-1] != None:
                    data.append(item)
        return data

    # print("MATRES Loading .......")
    # aquaint_dir_name = "../datasets/MATRES/TBAQ-cleaned/AQUAINT/"
    # timebank_dir_name = "../datasets/MATRES/TBAQ-cleaned/TimeBank/"
    # platinum_dir_name = "../datasets/MATRES/te3-platinum/"
    # validate_MATRES = load_dataset(aquaint_dir_name, 'tml')
    # train_MATRES = load_dataset(timebank_dir_name, 'tml')
    # test_MATRES = load_dataset(platinum_dir_name, 'tml')

    print("TBD Loading .......")
    train_dir = "../datasets/TimeBank-dense/train/"
    test_dir = "../datasets/TimeBank-dense/test/"
    validate_dir = "../datasets/TimeBank-dense/dev/"
    train_TBD = load_dataset(train_dir, 'tbd_tml')
    test_TBD = load_dataset(test_dir, 'tbd_tml')
    validate_TBD = load_dataset(validate_dir, 'tbd_tml')
    #
    # print("TDD_man Loading .......")
    # train_TDD_man = load_dataset(train_dir, 'tdd_man')
    # test_TDD_man = load_dataset(test_dir, 'tdd_man')
    # validate_TDD_man = load_dataset(validate_dir, 'tdd_man')
    #
    # print("TDD_auto Loading .......")
    # train_TDD_auto = load_dataset(train_dir, 'tdd_auto')
    # test_TDD_auto = load_dataset(test_dir, 'tdd_auto')
    # validate_TDD_auto = load_dataset(validate_dir, 'tdd_auto')
    #
    # print("I2B2 Loading .......")
    # train_dir_I2B2 = "../datasets/I2B2/train/"
    # test_dir_I2B2 = "../datasets/I2B2/test/"
    # train_I2B2 = load_dataset(train_dir_I2B2, 'i2b2_xml')
    # test_I2B2 = load_dataset(test_dir_I2B2, 'i2b2_xml')
    #
    # print("Causal-TimeBank Loading .......")
    # data_dir_CTB = "../datasets/Causal-TimeBank/"
    # data_CTB = load_dataset(data_dir_CTB, 'CTB_tml')
    #
    # print("EventStoryLine Loading .......")
    # train_topic = [5, 7, 8, 32, 33, 35]
    # test_topic = [1, 3, 4, 12, 13, 14, 16, 18, 19, 20, 22, 23, 24, 30, 37, 41]
    # train_ESL = load_ESL_dataset(train_topic)
    # test_ESL = load_ESL_dataset(test_topic)


    train_set = []
    test_set = []
    validate_set = []
    train_rebalance = []

    if dataset == "MATRES":


        train = train_MATRES
        validate = validate_MATRES
        test = test_MATRES

        _tt = train + validate
        _tt = list(sorted(_tt, key=lambda x: x["doc_id"]))
        train, validate = train_test_split(_tt, test_size=0.7, train_size=0.3)

        # validate, train = train_test_split(_tt, test_size=30)

        # validate = validate + train_I2B2 + test_I2B2

        processed_dir = "../datasets/docEvR_processed1/"
        if not os.path.exists(processed_dir):
            os.mkdir(processed_dir)

        for my_dict in tqdm.tqdm(train):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 2)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                train_set.append(item)

        for my_dict in tqdm.tqdm(validate):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 2)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                validate_set.append(item)

        for my_dict in tqdm.tqdm(test):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 2)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                test_set.append(item)

    if dataset == "HiEve":
        print("HiEve Loading .....")
        dir_name = "../datasets/hievents_v2/processed/"
        corpus = load_dataset(dir_name, 'tsvx')
        corpus = list(sorted(corpus, key=lambda x: x["doc_id"]))
        train, test = train_test_split(corpus, train_size=0.8, test_size=0.2)
        train, validate = train_test_split(train, train_size=0.75, test_size=0.25)
        sp = 0.4

        processed_dir = "../datasets/hievents_v2/processed/docEvR_processed/"
        if not os.path.exists(processed_dir):
            os.mkdir(processed_dir)

        for my_dict in tqdm.tqdm(train):
            file_name = my_dict["doc_id"] + ".pkl"
            if os.path.exists(processed_dir+file_name):
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            else:
                data = get_data_point(my_dict, 1)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            for item in data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < sp:
                        train_set.append(item)
                else:
                    train_set.append(item)

        for my_dict in tqdm.tqdm(test):
            file_name = my_dict["doc_id"] + ".pkl"
            if os.path.exists(processed_dir+file_name):
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            else:
                data = get_data_point(my_dict, 1)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            for item in data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < sp:
                        test_set.append(item)
                else:
                    test_set.append(item)

        for my_dict in tqdm.tqdm(validate):
            file_name = my_dict["doc_id"] + ".pkl"
            if os.path.exists(processed_dir+file_name):
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            else:
                data = get_data_point(my_dict, 1)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            for item in data:
                if item[-1] == 3:
                    if random.uniform(0, 1) < sp:
                        validate_set.append(item)
                else:
                    validate_set.append(item)

    if dataset == 'TBD':

        train= train_TBD
        test = test_TBD
        validate = validate_TBD

        train = train + validate
        train = list(sorted(train, key=lambda x: x["doc_id"]))
        train, validate = train_test_split(train, test_size=0.9, train_size=0.1)

        # validate, train = train_test_split(_tt, test_size=0.3)

        # validate = validate + train_MATRES + test_MATRES + validate_MATRES

        processed_dir = "../datasets/docEvR_processed1/"
        if not os.path.exists(processed_dir):
            os.mkdir(processed_dir)

        for my_dict in tqdm.tqdm(train):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 4)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                train_set.append(item)

        for my_dict in tqdm.tqdm(test):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 4)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                test_set.append(item)

        for my_dict in tqdm.tqdm(validate):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 4)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                validate_set.append(item)

    if dataset == 'TDD_man':
        train= train_TDD_man
        test = test_TDD_man
        validate = validate_TDD_man

        _tt = train + validate
        _tt = list(sorted(_tt, key=lambda x: x["doc_id"]))
        train, validate = train_test_split(_tt, test_size=0.7, train_size=0.3)

        # _tt = train + validate
        # _tt = list(sorted(_tt, key=lambda x: x["doc_id"]))
        # train, validate = train_test_split(_tt, test_size=0.6, train_size=0.4)
        # validate = validate + train_TBD + test_TBD + validate_TBD + train_MATRES + test_MATRES + validate_MATRES \
        #                 + train_TDD_auto + test_TDD_auto + validate_TDD_auto + train_I2B2 + test_I2B2 + data_CTB + \
        #                 train_ESL + test_ESL

        processed_dir = "../datasets/TDDiscourse/TDDMan/docEvR_processed/"
        if not os.path.exists(processed_dir):
            os.mkdir(processed_dir)

        for my_dict in tqdm.tqdm(train):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 5)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                train_set.append(item)

        for my_dict in tqdm.tqdm(test):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 5)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                test_set.append(item)

        for my_dict in tqdm.tqdm(validate):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 5)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                validate_set.append(item)

    if dataset == 'TDD_auto':
        train= train_TDD_auto
        test = test_TDD_auto
        validate = validate_TDD_auto

        _tt = train + validate
        _tt = list(sorted(_tt, key=lambda x: x["doc_id"]))
        train, validate = train_test_split(_tt, test_size=0.7, train_size=0.3)

        # _tt = train + validate
        # _tt = list(sorted(_tt, key=lambda x: x["doc_id"]))
        # train, validate = train_test_split(_tt, test_size=0.6, train_size=0.4)
        # validate = validate + train_TBD + test_TBD + validate_TBD + train_TDD_man + test_TDD_man + validate_TDD_man \
        #                 + train_MATRES + test_MATRES + validate_MATRES + train_I2B2 + test_I2B2 + data_CTB + \
        #                 train_ESL + test_ESL

        processed_dir = "../datasets/TDDiscourse/TDDAuto/docEvR_processed/"
        if not os.path.exists(processed_dir):
            os.mkdir(processed_dir)

        for my_dict in tqdm.tqdm(train):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 5)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                train_set.append(item)

        for my_dict in tqdm.tqdm(test):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 5)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                test_set.append(item)

        for my_dict in tqdm.tqdm(validate):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir+file_name):
                data = get_data_point(my_dict, 5)
                with open(processed_dir+file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir+file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                validate_set.append(item)

    if dataset == "I2B2":
        train= train_I2B2
        test = test_I2B2

        _tt = train
        _tt = list(sorted(_tt, key=lambda x: x["doc_id"]))
        train, validate = train_test_split(_tt, test_size=0.6, train_size=0.4)
        validate = validate + train_TBD + test_TBD + validate_TBD + train_TDD_man + test_TDD_man + validate_TDD_man \
                        + train_TDD_auto + test_TDD_auto + validate_TDD_auto + train_MATRES + test_MATRES + validate_MATRES + data_CTB + \
                        train_ESL + test_ESL

        # processed_dir = "./datasets/MATRES/docEvR_processed_kg/"
        processed_dir = "../datasets/I2B2/docEvR_processed/"
        if not os.path.exists(processed_dir):
            os.mkdir(processed_dir)

        for my_dict in tqdm.tqdm(train):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir + file_name):
                data = get_data_point(my_dict, 3)
                with open(processed_dir + file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir + file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                train_set.append(item)

        for my_dict in tqdm.tqdm(validate):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir + file_name):
                data = get_data_point(my_dict, 3)
                with open(processed_dir + file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir + file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                validate_set.append(item)

        for my_dict in tqdm.tqdm(test):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir + file_name):
                data = get_data_point(my_dict, 3)
                with open(processed_dir + file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir + file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                test_set.append(item)

    if dataset == 'CTB': # Causal-TimeBank

        _tt = data_CTB
        _tt = list(sorted(_tt, key=lambda x: x["doc_id"]))
        train, test = train_test_split(_tt, test_size=0.6, train_size=0.4)
        validate = train_TBD + test_TBD + validate_TBD + train_TDD_man + test_TDD_man + validate_TDD_man \
                        + train_TDD_auto + test_TDD_auto + validate_TDD_auto + train_I2B2 + test_I2B2 + train_MATRES + test_MATRES + validate_MATRES + \
                        train_ESL + test_ESL

        processed_dir = "../datasets/Causal-TimeBank/docEvR_processed/"
        if not os.path.exists(processed_dir):
            os.mkdir(processed_dir)

        for my_dict in tqdm.tqdm(train):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir + file_name):
                data = get_data_point(my_dict, 4)
                with open(processed_dir + file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir + file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                train_set.append(item)

        for my_dict in tqdm.tqdm(test):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir + file_name):
                data = get_data_point(my_dict, 4)
                with open(processed_dir + file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir + file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                test_set.append(item)

        for my_dict in tqdm.tqdm(validate):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir + file_name):
                data = get_data_point(my_dict, 4)
                with open(processed_dir + file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir + file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                validate_set.append(item)

    if dataset == "ESL": # EventStoryLine
        train = train_ESL
        test = test_ESL

        _tt = train
        _tt = list(sorted(_tt, key=lambda x: x["doc_id"]))
        train, validate = train_test_split(_tt, test_size=0.6, train_size=0.4)
        validate = validate + train_TBD + test_TBD + validate_TBD + train_TDD_man + test_TDD_man + validate_TDD_man \
                        + train_TDD_auto + test_TDD_auto + validate_TDD_auto + train_I2B2 + test_I2B2 + data_CTB + \
                        train_MATRES + test_MATRES + validate_MATRES

        processed_dir = "../datasets/EventStoryLine/docEvR_processed/"
        if not os.path.exists(processed_dir):
            os.mkdir(processed_dir)

        for my_dict in tqdm.tqdm(train):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir + file_name):
                data = get_data_point(my_dict, 6)
                with open(processed_dir + file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir + file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                train_set.append(item)

        for my_dict in tqdm.tqdm(test):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir + file_name):
                data = get_data_point(my_dict, 6)
                with open(processed_dir + file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir + file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                test_set.append(item)

        for my_dict in tqdm.tqdm(validate):
            file_name = my_dict["doc_id"] + ".pkl"
            if not os.path.exists(processed_dir + file_name):
                data = get_data_point(my_dict, 6)
                with open(processed_dir + file_name, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(processed_dir + file_name, 'rb') as f:
                    data = pickle.load(f)
            for item in data:
                validate_set.append(item)

    # if dataset=='infer':
    #     reader = Reader(file_type)
    #     print(f'Reading file {file_path} ....')
    #     my_dict = reader.read('', file_path)
    #     data = get_data_point(my_dict, label_type)
    #     for item in data:
    #         if len(item[-4]) >= min_ns:
    #             test_set.append(item)
    #         else:
    #             test_short.append(item)
    #     print("Train_size: {}".format(len(train_set)))
    #     print("Test_size: {}".format(len(test_set)))
    #     print("Validate_size: {}".format(len(validate_set)))
    #     print("Train_size: {}".format(len(train_short)))
    #     print("Test_size: {}".format(len(test_short)))
    #     print("Validate_size: {}".format(len(validate_short)))

    del sent_encoder
    gc.collect()

    return train_set, test_set, validate_set

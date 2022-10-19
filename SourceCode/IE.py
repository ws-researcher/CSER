import copy
import os
import time

import torch
import tqdm
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import DataLoader

from SourceCode.data_loader.EventDataset import EventDataset
from SourceCode.data_loader.loader import loader
from SourceCode.models.EEREModel import EERRoberta, SupConLoss, ConLoss_S, ConLoss_SR, ConLoss_R
from SourceCode.utils.constant import CUDA
from SourceCode.utils.evaluation import ClassificationReport
from SourceCode.utils.tools import format_time, make_predictor_input, make_sc_input, augment_target, spilt, \
    make_zl_input, patternArgm
import torch.nn.functional as F

class Objective(object):
    def __init__(self, args):
        # Hold this implementation specific arguments as the fields of the class.
        self.args = args

    def __call__(self, trial):
        # self.args.sc_word_drop_rate = trial.suggest_float("sc_word_drop_rate", 0, 1)
        # self.args.drop_rate = trial.suggest_float("drop_rate", 0, 1)

        # self.args.ce_word_drop_rate = trial.suggest_float("ce_word_drop_rate", 0, 1)

        # self.args.roberta_type = trial.suggest_categorical('roberta_type', ['roberta-base', 'roberta-large'])
        self.checkpoint_path = os.path.join("../saved_models/", self.args.CS + str(self.args.sc_word_drop_rate) + self.args.datasets[0] + ".pt")
        self.train_dataloader, self.validate_dataloaders, self.test_dataloaders, self.all_dataloaders, self.trainData, self.vtData = self._build_dataloader()
        self.model = self._build_model()
        self.suc_loss, self.sc_loss, self.ce_loss = self._build_loss()
        self.optimizer, self.b_parameters, self.mlp_parameters, self.head_parameters = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        f1= self.train()
        return f1

    def _build_dataloader(self):
        params = {'batch_size': self.args.bs,
                  'shuffle': True,
                  'collate_fn': EventDataset.collate_fn}

        validate_dataloaders = {}
        test_dataloaders = {}
        all_dataloaders = {}
        trainData = []
        vtData = []
        for dataset in self.args.datasets:
            train, test, unlabeledData = loader(dataset, min_ns=5)

            all_data = train + unlabeledData
            # train, validate = train_test_split(all_data, test_size=0.7, train_size=0.3)
            vtData.extend(unlabeledData + test)
            trainData.extend(train)

            # validate_dataloader = DataLoader(EventDataset(validate), **params)
            test_dataloader = DataLoader(EventDataset(test), **params)
            # validate_dataloaders[dataset] = validate_dataloader
            test_dataloaders[dataset] = test_dataloader

            all_dataloader = DataLoader(EventDataset(all_data), **params)
            all_dataloaders[dataset] = all_dataloader

        train_dataloader = DataLoader(EventDataset(trainData), **params)

        return train_dataloader, validate_dataloaders, test_dataloaders, all_dataloaders, trainData, vtData

    def _build_model(self):
        EERRModel = EERRoberta(roberta_type=self.args.roberta_type, datasets=self.args.datasets, pos_dim=16,
                               fn_activate=self.args.fn_activate, drop_rate=self.args.drop_rate, task_weights=None, temperature=0.1)

        if CUDA:
            EERRModel = nn.DataParallel(EERRModel, device_ids=[0, 1])
            EERRModel = EERRModel.cuda()

        # if os.path.exists(self.checkpoint_path) and self.args.sc_epoche != 0:
        #     loaded_dict = torch.load(self.checkpoint_path)
        #     EERRModel.state_dict = loaded_dict
        return EERRModel

    def _build_loss(self):

        if self.args.CS == "S":
            sc_loss = ConLoss_S(self.args.temperature)
        elif self.args.CS == "SR":
            sc_loss = ConLoss_SR(self.args.temperature)
        else:
            sc_loss = ConLoss_R(self.args.temperature)

        suc_loss = SupConLoss(self.args.temperature)

        for dataset in self.args.datasets:
            if dataset == "MATRES":
                weights = [30.0 / 412, 30.0 / 263, 30.0 / 30, 30.0 / 113, ]
                weights = torch.tensor(weights)
                ce_loss = nn.CrossEntropyLoss(weight=weights)
            elif "TBD" in dataset:
                weights = [41.0/387, 41.0/287, 41.0/64, 41.0/74, 41.0/41, 41.0/642]
                weights = torch.tensor(weights)
                ce_loss = nn.CrossEntropyLoss(weight=weights)
            elif "TDD" in dataset:
                weights = [41.0/387, 41.0/287, 41.0/64, 41.0/74, 41.0/41]
                weights = torch.tensor(weights)
                ce_loss = nn.CrossEntropyLoss(weight=weights)
            elif "HiEve" in dataset:
                weights = [993.0/333, 993.0/349, 933.0/128, 933.0/453]
                weights = torch.tensor(weights)
                ce_loss = nn.CrossEntropyLoss(weight=weights)
            elif "I2B2" in dataset:
                weights = [213.0/368, 213.0/213, 213.0/1013]
                weights = torch.tensor(weights)
                ce_loss = nn.CrossEntropyLoss(weight=weights)
            elif "ESL" in dataset:
                # weights = [1591.0/1591, 1591.0/78950]
                # weights = torch.tensor(weights)
                # ce_loss = nn.CrossEntropyLoss(weight=weights)
                ce_loss = nn.CrossEntropyLoss()

        if CUDA:
            sc_loss = sc_loss.cuda()
            ce_loss = ce_loss.cuda()

        return suc_loss, sc_loss, ce_loss

    def _build_optimizer(self):
        model = self.model
        mlp = ['fc1', 'fc2', 'pos_emb']
        head = ["h1", "h2"]
        no_decay = ['bias', 'gamma', 'beta']
        group1 = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.']
        group2 = ['layer.6.', 'layer.7.', 'layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
        group3 = ['layer.12.', 'layer.13.', 'layer.14.', 'layer.15.', 'layer.16.', 'layer.17.']
        group4 = ['layer.18.', 'layer.19.', 'layer.20.', 'layer.21.', 'layer.22.', 'layer.23.']
        group_all = group1 + group2 + group3 + group4
        b_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in mlp) and not any(nd in n for nd in head) and not any(nd in n for nd in no_decay) and not any(
                            nd in n for nd in group_all)],
             'weight_decay_rate': 0.01,
             'lr': self.args.b_lr},
            # all params not include bert layers
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in mlp) and not any(nd in n for nd in head) and not any(nd in n for nd in no_decay) and any(
                            nd in n for nd in group1)],
             'weight_decay_rate': 0.01,
             'lr': self.args.b_lr * (self.args.b_lr_decay_rate ** 3)},  # param in group1
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in mlp) and not any(nd in n for nd in head) and not any(nd in n for nd in no_decay) and any(
                            nd in n for nd in group2)],
             'weight_decay_rate': 0.01,
             'lr': self.args.b_lr * (self.args.b_lr_decay_rate ** 2)},  # param in group2
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in mlp) and not any(nd in n for nd in head) and not any(nd in n for nd in no_decay) and any(
                            nd in n for nd in group3)],
             'weight_decay_rate': 0.01,
             'lr': self.args.b_lr * (self.args.b_lr_decay_rate ** 1)},  # param in group3
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in mlp) and not any(nd in n for nd in head) and not any(nd in n for nd in no_decay) and any(
                            nd in n for nd in group4)],
             'weight_decay_rate': 0.01,
             'lr': self.args.b_lr * (self.args.b_lr_decay_rate ** 0)},  # param in group4
            # no_decay
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in mlp) and not any(nd in n for nd in head) and any(nd in n for nd in no_decay) and not any(
                            nd in n for nd in group_all)],
             'weight_decay_rate': 0.00,
             'lr': self.args.b_lr},
            # all params not include bert layers
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in mlp) and not any(nd in n for nd in head) and any(nd in n for nd in no_decay) and any(
                            nd in n for nd in group1)],
             'weight_decay_rate': 0.00,
             'lr': self.args.b_lr * (self.args.b_lr_decay_rate ** 3)},  # param in group1
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in mlp) and not any(nd in n for nd in head) and any(nd in n for nd in no_decay) and any(
                            nd in n for nd in group2)],
             'weight_decay_rate': 0.00,
             'lr': self.args.b_lr * (self.args.b_lr_decay_rate ** 2)},  # param in group2
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in mlp) and not any(nd in n for nd in head) and any(nd in n for nd in no_decay) and any(
                            nd in n for nd in group3)],
             'weight_decay_rate': 0.00,
             'lr': self.args.b_lr * (self.args.b_lr_decay_rate ** 1)},  # param in group3
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in mlp) and not any(nd in n for nd in head) and any(nd in n for nd in no_decay) and any(
                            nd in n for nd in group4)],
             'weight_decay_rate': 0.00,
             'lr': self.args.b_lr * (self.args.b_lr_decay_rate ** 0)},  # param in group3
        ]
        mlp_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in mlp) and not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01,
             'lr': self.args.m_lr},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in mlp) and any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.00,
             'lr': self.args.m_lr},
        ]

        head_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in head) and not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01,
             'lr': self.args.h_lr},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in head) and any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.00,
             'lr': self.args.h_lr},
        ]

        optimizer_parameters = b_parameters + mlp_parameters + head_parameters

        optimizer = optim.AdamW(optimizer_parameters, weight_decay=self.args.weight_decay)

        # optimizer = nn.DataParallel(optimizer)

        return optimizer, b_parameters, mlp_parameters, head_parameters

    def _build_scheduler(self):
        num_alldata = 0
        for i in range(0, len(self.all_dataloaders)):
            dataset = self.args.datasets[i]
            num_alldata += len(self.all_dataloaders[dataset])

        num_training_steps = num_alldata * (self.args.sc_epoche)+len(self.train_dataloader) * (self.args.epoche)
        num_sc_steps = int((num_alldata * self.args.sc_epoche))

        def linear_lr_lambda(current_step: int):
            if current_step < num_sc_steps:
                # return float(current_step) / float(max(1, num_sc_steps))
                return 0.5 ** float(num_sc_steps - current_step) / float(max(1, num_sc_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(
                    max(1, num_training_steps - num_sc_steps))
            )

        # lambda scheduler for MLP
        def m_lr_lambda(current_step: int):
            if current_step >= num_sc_steps:
                return 0.5 ** int((current_step - num_sc_steps) / (2 * len(self.train_dataloader)))
            return 0

        def h_lr_lambda(current_step: int):
            if current_step < num_sc_steps:
                return 0.5 ** int(current_step / (2 * num_alldata))
            return 0

        lamd = [linear_lr_lambda] * 10
        mlp_lambda = [m_lr_lambda] * 2
        h_lambda = [h_lr_lambda] * 2
        lamd.extend(mlp_lambda)
        lamd.extend(h_lambda)

        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lamd)

        return scheduler

    def train(self):
        start_time = time.time()
        sum_f1 = 0

        # print("SUC training ........")
        # for i in range(self.args.sc_epoche):
        #     # if os.path.exists(self.checkpoint_path):
        #     #     break
        #     print("============== Epoch {} / {} ==============".format(i + 1, self.args.sc_epoche))
        #     t0 = time.time()
        #     self.model.train()
        #     self.model.zero_grad()
        #     predictor_loss = 0.0
        #     for step, batch in tqdm.tqdm(enumerate(self.train_rebalance_dataloader), desc="Training process for long doc",
        #                                  total=len(self.train_rebalance_dataloader)):
        #
        #         # doc_id, x, y, x_sent, y_sent, x_sent_id, y_sent_id, x_sent_pos, y_sent_pos, x_sent_tag, y_sent_tag, x_position, y_position, flag, xy = batch
        #         doc_id, x, y, sents, sents_pos, sents_tag, x_position_new, y_position_new, flag, xy = batch
        #
        #         if len(doc_id) % 2 != 0:
        #             continue
        #
        #         # xy_seedLabels = patternArgm(x_sent, y_sent, x_sent_pos, y_sent_pos, x_sent_tag, y_sent_tag, x_sent_id, y_sent_id, x_position,
        #         #                          y_position, flag, xy)
        #         # xy_seedLabels = torch.tensor(xy_seedLabels)
        #
        #         # self.optimizer.zero_grad()
        #         # self.model.zero_grad()
        #
        #
        #         augm_target, augm_target_mask, augm_pos_target, x_augm_position, y_augm_position, flag, xy = \
        #             make_sc_input(sents, sents_pos, sents_tag, x_position_new, y_position_new, flag, xy, dropout_rate=self.args.sc_word_drop_rate, views=0)
        #
        #         # augm_target_neg, augm_target_mask_neg, augm_pos_target_neg, x_augm_position_neg, y_augm_position_neg, flag, xy_neg = \
        #         #     make_sc_input(sents, sents_pos, sents_tag, x_position_new, y_position_new, flag, xy, dropout_rate=self.args.sc_word_drop_rate, views=self.args.views, is_reverse=True)
        #
        #
        #         xy = torch.tensor(xy)
        #         flag = torch.tensor(flag)
        #         if CUDA:
        #             augm_target = augm_target.cuda()
        #             augm_target_mask = augm_target_mask.cuda()
        #             augm_pos_target = augm_pos_target.cuda()
        #             x_augm_position = x_augm_position.cuda()
        #             y_augm_position = y_augm_position.cuda()
        #
        #         feat = self.model(augm_target, augm_target_mask, x_augm_position, y_augm_position, xy, flag,
        #                                sent_pos=augm_pos_target, views=self.args.views)
        #
        #         bs = len(doc_id)
        #         FeatViews = []
        #         for i in range(self.args.views + 1):
        #             FeatView = []
        #             for b in range(bs):
        #                 FeatView.append(feat[i + b * (self.args.views + 1)])
        #             FeatViews.append(torch.tensor(FeatView).unsqueeze(1))
        #
        #         feat1 = torch.cat(FeatViews, dim=1)
        #
        #         SCloss = self.suc_loss(feat1, labels=xy)
        #
        #         SCloss.backward(retain_graph=False)
        #         self.optimizer.step()
        #         self.scheduler.step()
        #         self.optimizer.zero_grad()
        #         predictor_loss += SCloss.cpu().item()
        #     print("Total training loss: {}".format(predictor_loss))

        print("SC training ........")
        for i in range(self.args.sc_epoche):
            # if os.path.exists(self.checkpoint_path):
            #     break
            print("============== Epoch {} / {} ==============".format(i + 1, self.args.sc_epoche))
            t0 = time.time()
            self.model.train()
            self.model.zero_grad()
            predictor_loss = 0.0
            for i in range(0, len(self.all_dataloaders)):
                dataset = self.args.datasets[i]
                all_dataloader = self.all_dataloaders.get(dataset)
                for step, batch in tqdm.tqdm(enumerate(all_dataloader), desc="Training process for long doc",
                                             total=len(all_dataloader)):

                    # doc_id, x, y, x_sent, y_sent, x_sent_id, y_sent_id, x_sent_pos, y_sent_pos, x_sent_tag, y_sent_tag, x_position, y_position, flag, xy = batch
                    doc_id, x, y, sents, sents_pos, sents_tag, x_position_new, y_position_new, flag, xy = batch

                    if len(doc_id) % 2 != 0:
                        continue

                    # xy_seedLabels = patternArgm(x_sent, y_sent, x_sent_pos, y_sent_pos, x_sent_tag, y_sent_tag, x_sent_id, y_sent_id, x_position,
                    #                          y_position, flag, xy)
                    # xy_seedLabels = torch.tensor(xy_seedLabels)

                    # self.optimizer.zero_grad()
                    # self.model.zero_grad()


                    augm_target, augm_target_mask, augm_pos_target, x_augm_position, y_augm_position, flag, _ = \
                        make_sc_input(sents, sents_pos, sents_tag, x_position_new, y_position_new, flag, xy, dropout_rate=self.args.sc_word_drop_rate, views=self.args.views)

                    augm_target_neg, augm_target_mask_neg, augm_pos_target_neg, x_augm_position_neg, y_augm_position_neg, flag, _ = \
                        make_sc_input(sents, sents_pos, sents_tag, x_position_new, y_position_new, flag, xy, dropout_rate=self.args.sc_word_drop_rate, views=self.args.views, is_reverse=True)


                    # xy = torch.tensor(xy)
                    flag = torch.tensor(flag)
                    # if CUDA:
                    #     augm_target = augm_target.cuda()
                    #     augm_target_mask = augm_target_mask.cuda()
                    #     augm_pos_target = augm_pos_target.cuda()
                    #     x_augm_position = x_augm_position.cuda()
                    #     y_augm_position = y_augm_position.cuda()
                    #
                    #     augm_target_neg = augm_target_neg.cuda()
                    #     augm_target_mask_neg = augm_target_mask_neg.cuda()
                    #     augm_pos_target_neg = augm_pos_target_neg.cuda()
                    #     x_augm_position_neg = x_augm_position_neg.cuda()
                    #     y_augm_position_neg = y_augm_position_neg.cuda()

                        # xy = xy.cuda()
                        # flag = flag.cuda()
                    if CUDA:
                        target = torch.cat((augm_target, augm_target_neg), 0).cuda()
                        target_mask = torch.cat((augm_target_mask, augm_target_mask_neg), 0).cuda()
                        pos_target = torch.cat((augm_pos_target, augm_pos_target_neg), 0).cuda()
                    else:
                        target = torch.cat((augm_target, augm_target_neg), 0)
                        target_mask = torch.cat((augm_target_mask, augm_target_mask_neg), 0)
                        pos_target = torch.cat((augm_pos_target, augm_pos_target_neg), 0)

                    with torch.no_grad():
                        x_position = torch.cat((x_augm_position, x_augm_position_neg), 0)
                        y_position = torch.cat((y_augm_position, y_augm_position_neg), 0)
                        flag = torch.cat((flag, flag), 0)
                    feat = self.model(target, target_mask, x_position, y_position, xy, flag,
                                           sent_pos=pos_target, views=2)

                    # feat = self.model(augm_target, augm_target_mask, x_augm_position, y_augm_position, xy, flag,
                    #                        sent_pos=augm_pos_target, views=2)

                    SCloss = self.sc_loss(feat, 3)
                    # SCloss = self.sc_loss(feat)
                    # SCloss = self.sc_loss(feat, labels=xy_seedLabels)

                    SCloss.backward(retain_graph=False)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    predictor_loss += SCloss.cpu().item()
                    # del feat, SCloss, augm_target, augm_target_mask, augm_pos_target, x_augm_position, y_augm_position, \
                    #     augm_target_neg, augm_target_mask_neg, augm_pos_target_neg, x_augm_position_neg, y_augm_position_neg, \
                    #     target, target_mask, pos_target, x_position, y_position, flag
                    # torch.cuda.empty_cache()

            epoch_training_time = format_time(time.time() - t0)
            print("Total training loss: {}".format(predictor_loss))

        # if self.args.sc_epoche != 0 and not os.path.exists(self.checkpoint_path):
        #     torch.save(self.model.state_dict(), self.checkpoint_path)

        print("Training models .....")
        for turn in range(1):
            for i in range(self.args.epoche + 5):
                if i >= self.args.epoche:
                    for group in self.b_parameters:
                        for param in group['params']:
                            param.requires_grad = False

                print("============== Epoch {} / {} ==============".format(i + 1, self.args.epoche + 5))
                t0 = time.time()

                self.model.train()
                self.model.zero_grad()

                for step, batch in tqdm.tqdm(enumerate(self.train_dataloader), desc="Training process for long doc",
                                             total=len(self.train_dataloader)):

                    # doc_id, x, y, x_sent, y_sent, x_sent_id, y_sent_id, x_sent_pos, y_sent_pos, x_sent_tag, y_sent_tag, x_position, y_position, flag, xy = batch
                    doc_id, x, y, sents, sents_pos, sents_tag, x_position_new, y_position_new, flag, xy = batch

                    self.optimizer.zero_grad()

                    augm_target, augm_target_mask, augm_pos_target, x_augm_position, y_augm_position, flag, xy = \
                        make_predictor_input(sents, sents_pos, sents_tag, x_position_new, y_position_new, flag, xy, dropout_rate=self.args.ce_word_drop_rate)

                    xy = torch.tensor(xy)
                    flag = torch.tensor(flag)
                    if CUDA:
                        augm_target = augm_target.cuda()
                        augm_target_mask = augm_target_mask.cuda()
                        augm_pos_target = augm_pos_target.cuda()
                        x_augm_position = x_augm_position.cuda()
                        y_augm_position = y_augm_position.cuda()
                        xy = xy.cuda()
                        flag = flag.cuda()
                    logits = self.model(augm_target, augm_target_mask, x_augm_position, y_augm_position, xy, flag,
                                           augm_pos_target)

                    CEloss = 0.0
                    for i in range(len(xy)):
                        logit  =  logits[i].unsqueeze(0)
                        target = xy[i].unsqueeze(0)
                        CEloss += self.ce_loss(logit, target)

                    CEloss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                # v_sum_f1, v_CMs, v_F1s = self.evaluate(self.model, self.validate_dataloaders, self.test_dataloaders)
                # t_sum_f1, t_CMs, t_F1s = self.evaluate(self.model, self.validate_dataloaders, self.test_dataloaders, is_test=True)
                # t_sum_f1 = self.evaluate(self.model, self.train_dataloader, self.test_dataloaders, is_test=False)
                t_sum_f1 = self.evaluate(self.model, self.validate_dataloaders, self.test_dataloaders, is_test=True)

                if t_sum_f1 > sum_f1:
                    sum_f1 = t_sum_f1
                    # torch.save(model, self.args.best_path + "EERRModel.pth")
                print("best F1:" + str(sum_f1))

            # self.getNewDataset()
        return sum_f1

    def evaluate(self, model, validate_dataloaders, test_dataloaders, is_test=False):
        F1s = []
        CMs = []
        sum_f1 = 0.0
        corpus_labels = {
            "MATRES": 4,
            "TBD": 6,
            "HiEve": 4
        }
        for i in range(0, len(test_dataloaders)):
            dataset = self.args.datasets[i]
            print("-------------------------------{}-------------------------------".format(dataset))
            if is_test:
                dataloader = test_dataloaders.get(dataset)
                print("Testset and best model was loaded!")
                print("Running on testset ..........")
            else:
                dataloader = validate_dataloaders
                # short_dataloader = self.validate_short_dataloaders[i]
                print("Running on validate set ..........")

            model.eval()
            pred = []
            gold = []

            E1 = []
            E2 = []
            x_id = []
            y_id = []
            d_id = []

            for step, batch in tqdm.tqdm(enumerate(dataloader), desc="Processing for long doc", total=len(dataloader)):
                # doc_id, x, y, x_sent, y_sent, x_sent_id, y_sent_id, x_sent_pos, y_sent_pos, x_sent_tag, y_sent_tag, x_position, y_position, flag, xy = batch
                doc_id, x, y, sents, sents_pos, sents_tag, x_position_new, y_position_new, flag, xy = batch

                augm_target, augm_target_mask, augm_pos_target, x_augm_position, y_augm_position, flag, xy = make_predictor_input(
                    sents, sents_pos, sents_tag, x_position_new, y_position_new, flag, xy,
                    dropout_rate=self.args.ce_word_drop_rate, is_test=True)
                xy = torch.tensor(xy)
                flag = torch.tensor(flag)
                if CUDA:
                    augm_target = augm_target.cuda()
                    augm_target_mask = augm_target_mask.cuda()
                    augm_pos_target = augm_pos_target.cuda()
                    x_augm_position = x_augm_position.cuda()
                    y_augm_position = y_augm_position.cuda()
                    xy = xy.cuda()
                    flag = flag.cuda()
                logits = model(augm_target, augm_target_mask, x_augm_position, y_augm_position, xy,
                                       flag, augm_pos_target)

                labels = xy.cpu().numpy()
                y_pred = torch.max(logits, 1).indices.cpu().numpy()
                gold.extend(labels)
                pred.extend(y_pred)
                E1.extend(x)
                E2.extend(y)
                # x_id.extend(x_sent_id)
                # y_id.extend(y_sent_id)
                d_id.extend(doc_id)

        #     F1 = 0
        #     if dataset in ["TBD"]:
        #         F1 = ClassificationReport("tbd", gold, pred, exclude_vague=False).get_result()
        #     elif dataset in ["MATRES"]:
        #         F1 = ClassificationReport("matres", gold, pred).get_result()
        #     sum_f1 += F1
        # return sum_f1

            w = []
            for i in range(len(gold)):
                if pred[i] != gold[i]:
                    w.append((i, pred[i], gold[i]))

            CM = confusion_matrix(gold, pred)
            if dataset in ["MATRES"]:
                # no eval in vague
                num_label = corpus_labels[dataset]
                true = sum([CM[i, i] for i in range(num_label - 1)])
                sum_pred = sum([CM[i, 0:(num_label - 1)].sum() for i in range(num_label)])
                sum_gold = sum([CM[i].sum() for i in range(num_label - 1)])
                P = true / sum_pred
                R = true / sum_gold
                F1 = 2 * P * R / (P + R)
                print("  P: {0:.3f}".format(P))
                print("  R: {0:.3f}".format(R))
                print("  F1: {0:.3f}".format(F1))
                print("  Confusion Matrix")
                print(CM)
                print("Classification report: \n {}".format(classification_report(gold, pred)))
            elif dataset == "HiEve":
                num_label = corpus_labels[dataset]
                true = sum([CM[i, i] for i in range(2)])
                sum_pred = sum([CM[i, 0:2].sum() for i in range(num_label)])
                sum_gold = sum([CM[i].sum() for i in range(2)])
                P = true / sum_pred
                R = true / sum_gold
                F1 = 2 * P * R / (P + R)
                print("  P: {0:.3f}".format(P))
                print("  R: {0:.3f}".format(R))
                print("  F1: {0:.3f}".format(F1))
                print("  Confusion Matrix")
                print(CM)
                print("Classification report HiEve: \n {}".format(classification_report(gold, pred)))
            else:
                P, R, F1 = precision_recall_fscore_support(gold, pred, average='micro')[0:3]
                print("  P: {0:.3f}".format(P))
                print("  R: {0:.3f}".format(R))
                print("  F1: {0:.3f}".format(F1))
                print("  Confusion Matrix")
                print(CM)
                print("Classification report: \n {}".format(classification_report(gold, pred)))

            sum_f1 += F1
            CMs.append(CM)
            F1s.append(F1)

        return sum_f1

        # return sum_f1, CMs, F1s

    def getNewDataset(self):
        new_sample = []
        # new_vtData = []
        rm_index = []
        bs = 10
        vt_dataloader = DataLoader(EventDataset(self.vtData), batch_size= bs, collate_fn = EventDataset.collate_fn)
        for step, batch in enumerate(vt_dataloader):
            # new_vtData = copy.deepcopy(self.vtData)
            # doc_id, x, y, x_sent, y_sent, x_sent_id, y_sent_id, x_sent_pos, y_sent_pos, x_sent_tag, y_sent_tag, x_position, y_position, flag, xy = batch
            doc_id, x, y, sents, sents_pos, sents_tag, x_position_new, y_position_new, flag, xy = batch

            target, target_mask, pos_target, x_position, y_position, flag, xy = make_zl_input(
                sents, sents_pos, sents_tag, x_position_new, y_position_new, flag, xy,
                dropout_rate=self.args.ce_word_drop_rate, is_test=True)

            if CUDA:
                target = target.cuda()
                target_mask = target_mask.cuda()
                pos_target = pos_target.cuda()
                x_position = x_position.cuda()
                y_position = y_position.cuda()
                xy = xy.cuda()
                flag = flag.cuda()

            logits = self.model(target, target_mask, x_position, y_position, xy,
                           flag, pos_target)

            s = len(logits) // 2
            c1 = torch.max(logits[:s], 1).indices.cpu().numpy()
            c2 = torch.max(logits[s:], 1).indices.cpu().numpy()

            for i in range(s):
                if c1[i] == c2[i]:
                    new_sample.append(self.vtData[bs * step + i])
                    rm_index.append(bs * step + i)

        self.vtData = [self.vtData[i] for i in range(len(self.vtData)) if i not in rm_index]
        self.trainData.extend(new_sample)
        print("添加{}个样本".format(len(new_sample)))
        params = {'batch_size': self.args.bs,
                  'shuffle': True,
                  'collate_fn': EventDataset.collate_fn}

        self.train_dataloader = DataLoader(EventDataset(self.trainData), **params)
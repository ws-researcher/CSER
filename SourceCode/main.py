
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import sys
import optuna
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
sys.path.append(os.path.dirname(sys.path[0]))
from SourceCode.IE import Objective


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='SEED', default=1741, type=int)
    parser.add_argument('--datasets', help="Name of dataset", action='append', required=True)
    parser.add_argument('--roberta_type', help="base or large", default='roberta-large', type=str)
    parser.add_argument('--best_path', help="Path for save model", type=str)
    parser.add_argument('--log_file', help="Path of log file", type=str)
    parser.add_argument('--bs', help='batch size', default=16, type=int)
    parser.add_argument('--epoche', help='epoches', default=7, type=int)
    parser.add_argument('--sc_epoche', help='Contrastive learning epoch', default=10, type=int)
    parser.add_argument('--b_lr', help='Robert lr', default=7e-06)
    parser.add_argument('--m_lr', help='MLP lr', default=5e-05)
    parser.add_argument('--h_lr', help='MLP lr', default=5e-05)
    parser.add_argument('--b_lr_decay_rate', help='robert lr decay', default=0.5)
    parser.add_argument('--sc_word_drop_rate', help='sentence word drop', default=0.1)
    parser.add_argument('--ce_word_drop_rate', help='sentence word drop', default=0.05)
    parser.add_argument('--fn_activate', help='MLP activate function', default='tanh')
    parser.add_argument('--drop_rate', help='', default=0.5)
    parser.add_argument('--weight_decay', help='', default=0.01)
    parser.add_argument('--views', help='', default=2)
    parser.add_argument('--CS', help='', default="R") # S:only symmetry negative sample; R:random negative sample; SR:symmetry negative sample and random negative sample

    parser.add_argument('-temperature', type=float, default=0.1)

    args = parser.parse_args()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(study_name="y", direction='maximize', sampler=sampler, storage='sqlite:///MATRES.db',
                                load_if_exists=True)

    study.optimize(Objective(args), n_trials=1)
    trial = study.best_trial

    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(args.datasets)
    print(df)

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))





    # sc_cutoffs = [0.35]
    # # sc_cutoffs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55]
    #
    # for cutoff in sc_cutoffs:
    #     args.sc_word_drop_rate = cutoff
    #     print("cutoff:{}".format(args.sc_word_drop_rate))
    #     sampler = optuna.samplers.TPESampler(seed=args.seed)
    #     study = optuna.create_study(study_name="y", direction='maximize', sampler=sampler, storage='sqlite:///MATRES.db', load_if_exists=True)
    #
    #     study.optimize(Objective(args), n_trials=1)
    #     trial = study.best_trial
    #
    #     df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    #     print(args.datasets)
    #     print(df)
    #
    #     print('Accuracy: {}'.format(trial.value))
    #     print("Best hyperparameters: {}".format(trial.params))
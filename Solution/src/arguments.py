import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='liver_preprocessed', # Classification task needs to add "_class"
                        help='{PPH_vag/ces_reg}, {wine}, {steam} {mimic_preprocessed_class} {mimic_ppc_class} {simulate} {thyroid} {examination} {liver_processed}')
    parser.add_argument('--nrun', type=int, default=1,
                        help='total number of runs[default: 1]')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='cross-validation fold, 1 refer not CV [default: 1]')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='proportion of test sets divided from training set, '
                             '0 refer dataset has its own test set [default: 0.2]')
    parser.add_argument('--val_ratio', type=float, default=0.,
                        help='proportion of test sets divided from training set [default: 0.2]')
    parser.add_argument('--method_mvi', type=str, default='s',
                        help='missing values imputation method [default: "mice"]')
    parser.add_argument('--dict_ManualRatio', type=dict, default={'drop_NanRatio': 1},
                        help='ratio of manual missing values [default: ""]')
    parser.add_argument('--missing_ratio', type=float, default=1.,
                        help='ratio of manual missing values [default: ""]')
    parser.add_argument('--Flag_LoadMetric', type=bool, default=False, metavar='N',
                        help='overload metric training before[default: False]')
    parser.add_argument('--Flag_DataPreprocessing', type=bool, default=True, metavar='N',
                        help='[default: True]')
    parser.add_argument('--Flag_MVI', type=bool, default=True, metavar='N',
                        help='[default: True]')
    parser.add_argument('--Flag_downstream', type=bool, default=True, metavar='N',
                        help='[default: True]')
    parser.add_argument('--Flag_Mask_Saving', type=bool, default=False, metavar='N',
                        help='[default: True]')
    parser.add_argument('--test', type=int, default=0, metavar='N',
                        help='[default: 0]')
    parser.add_argument('--cuda', action='store_true', help='If training is to be done on a GPU')
    args = parser.parse_args()

    return args

import copy
import time
import re
import pandas as pd
import arguments
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from preprocess import load_data
from preprocess.get_dataset import DataPreprocessing
from preprocess.missing_values_imputation import MVI
from preprocess.representation_learning import RepresentationLearning
from model.ReinforcementLearning import RL

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run(train_data, test_data, target, args, trial) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    target: dict
    if args.test_ratio == 0 or not test_data.empty:
        train_set = train_data
        test_set = test_data
    else:
        train_set, test_set = train_test_split(train_data, test_size=args.test_ratio, random_state=args.seed, shuffle=False)

    metric_df_all = pd.DataFrame([])
    q_train_df_all = pd.DataFrame([])
    q_test_df_all = pd.DataFrame([])
    history_df = pd.DataFrame([])
    q_test_df_box_all_flod = pd.DataFrame([])
    q_test_df_all_flod = pd.DataFrame([])
    kf = KFold(n_splits=args.n_splits)
    for k, (train_index, val_index) in enumerate(kf.split(train_set)):
        metric_all_fold = pd.DataFrame([])
        train_set_cv = train_set.iloc[train_index]
        val_set_cv = train_set.iloc[val_index]
        test_set_cv = copy.deepcopy(test_set)

        dp = DataPreprocessing(train_set_cv, val_set_cv, test_set_cv, None, seed=args.seed,
                               flag_label_onehot=False,
                               flag_ex_null=True, flag_ex_std_flag=False, flag_ex_occ=False,
                               flag_ca_co_sel=True, flag_ca_fac=True, flag_onehot=True, flag_nor=True,
                               flag_feat_emb=False, flag_RUS=False, flag_confusion=False, flaq_save=False)
        if args.Flag_DataPreprocessing:
            train_set_cv, val_set_cv, test_set_cv, ca_col, co_col, nor = dp.process()
            # train_set_cv.to_csv(f'./Train_KFlod[{k}]_{args.task_name}.csv', index_label=['index'])
            # val_set_cv.to_csv(f'./Val_KFlod[{k}]_{args.task_name}.csv', index_label=['index'])
            # test_set_cv.to_csv(f'./Test_KFlod[{k}]_{args.task_name}.csv', index_label=['index'])
        else:
            train_set_cv = pd.read_csv(f'../DataSet/{args.task_name}/processed_flod/data_kflod/Train_KFlod[{k}]_{args.task_name}.csv', index_col=['index'])
            val_set_cv = pd.read_csv(f'../DataSet/{args.task_name}/processed_flod/data_kflod/Val_KFlod[{k}]_{args.task_name}.csv', index_col=['index'])
            test_set_cv = pd.read_csv(f'../DataSet/{args.task_name}/processed_flod/data_kflod/Test_KFlod[{k}]_{args.task_name}.csv', index_col=['index'])
            nor = None

        col_drop = dp.features_ex(train_set_cv, args.dict_ManualRatio['drop_NanRatio']) # Drop useless features (high deletion rate, small variance, etc.)
        train_set_cv.drop(columns=col_drop, inplace=True)
        col_ = train_set_cv.columns
        val_set_cv = val_set_cv[col_]
        test_set_cv = test_set_cv[col_]
        ca_col = train_set_cv.filter(regex=r'sparse').columns.tolist()
        co_col = train_set_cv.filter(regex=r'dense').columns.tolist()

        train_label = train_set_cv[[x for x in target.values()]]
        val_label = val_set_cv[[x for x in target.values()]]
        test_label = test_set_cv[[x for x in target.values()]]
        train_x = train_set_cv.drop(columns=target.values())
        val_x = val_set_cv.drop(columns=target.values())
        test_x = test_set_cv.drop(columns=target.values())

        print(f'train_x shape {train_x.shape} | val_x shape {val_x.shape} | test_x shape {test_x.shape}')

        # missing values imputation start
        mvi = MVI(co_col, ca_col, args.seed)
        train_x_filled = mvi.fit_transform(train_x)
        val_x_filled = mvi.transform(val_x)
        test_x_filled = mvi.transform(test_x)
        # missing values imputation end

        # Dimension reduction Start
        represent = RepresentationLearning(dim_input_list=[24, 84, 158])
        train_x_hidden = represent.fit_transform(train_x_filled, val_x=val_x_filled)
        val_x_hidden = represent.transform(val_x_filled)
        test_x_hidden = represent.transform(test_x_filled)
        # Dimension reduction End

        # RL
        method_name_RL = 'ADRL_LC'
        state_dim = train_x_hidden.shape[1] - 3 - train_x_hidden.filter(regex=r'action').shape[1]
        model = RL(state_dim=state_dim,
                   action_dim=train_x_hidden.filter(regex=r'action').shape[1], max_episodes=400, seed=args.seed, ckpt_dir='', method_RL=method_name_RL)

        model.store_data(train_x_hidden, train_label)
        model.agent.learn_class(train_x_hidden.drop(columns=train_x_hidden.filter(regex=r'时间|label').columns), train_label[['label1']],
                                val_x_hidden.drop(columns=val_x_hidden.filter(regex=r'时间|label').columns), val_label[['label1']])
        critic_loss_all, actor_loss_all, q_test_list, q_test_df_mean_epoch = model.learn(val_x_hidden, val_label)

        history_df_tmp = pd.DataFrame(critic_loss_all, columns=[f'critic_{k}'])
        history_df_tmp[f'action_{k}'] = actor_loss_all
        history_df = pd.concat([history_df, history_df_tmp], axis=1)

        q_test_df = model.eval(copy.deepcopy(test_x_hidden), test_label, type_task='RL')
        if q_test_df_all.empty:
            q_test_df_all = copy.deepcopy(q_test_df)
        else:
            q_test_df_all += copy.deepcopy(q_test_df)

        # q_test_df.to_csv(f'./q_test_df_flod{k}.csv', index=False)
        q_test_df_all_flod = pd.concat([q_test_df_all_flod, q_test_df], axis=0)
        # q_test_df_all_flod.to_csv(f'./q_test_all_flod_{method_name_RL}.csv', index=False)
        q_test_df_box_all = pd.DataFrame([])
        for q_col in q_test_df.filter(regex=r'q').columns:
            if q_col == 'q_diff' or q_col == 'q_zero' or q_col == 'q_round':
                continue
            q_test_df_box = q_test_df[[q_col]]
            q_test_df_box.columns = ['q values']
            q_test_df_box['index'] = q_col
            q_test_df_box_all = pd.concat([q_test_df_box_all, q_test_df_box], axis=0)
        q_test_df_box = q_test_df[q_test_df['label1'] == 1][['q_record']]
        q_test_df_box.columns = ['q values']
        q_test_df_box['index'] = 'q_record_1'
        q_test_df_box_all = pd.concat([q_test_df_box_all, q_test_df_box], axis=0)
        q_test_df_box = q_test_df[q_test_df['label1'] == 0][['q_record']]
        q_test_df_box.columns = ['q values']
        q_test_df_box['index'] = 'q_record_0'
        q_test_df_box_all = pd.concat([q_test_df_box_all, q_test_df_box], axis=0)
        q_test_df_box_all.to_csv(f'./q_test_df_box_flod_{k}_{method_name_RL}.csv', index=False)
        q_test_df_box_all_flod = pd.concat([q_test_df_box_all_flod, q_test_df_box_all], axis=0)
        q_test_df_box_all_flod.to_csv(f'./q_test_box_all_flod_{method_name_RL}.csv', index=False)

        q_train_df = model.eval(copy.deepcopy(train_x_filled), train_label, type_task='RL')
        if q_train_df_all.empty:
            q_train_df_all = q_train_df
        else:
            q_train_df_all += q_train_df
        # RL end

        metric_df_all = pd.concat([metric_df_all, metric_all_fold], axis=0)
    history_df.to_csv(f'./history_df_{method_name_RL}.csv', index=False)
    q_test_df_mean = q_test_df_all/(k+1)
    q_test_df_mean.to_csv(f'./{args.task_name}_q_test_mean_{method_name_RL}.csv', index=False)
    # metric_df_all.to_csv(f'./{args.task_name}_RL_metric_{method_name_RL}.csv')
    return metric_df_all


if __name__ == "__main__":
    args = arguments.get_args()

    test_prediction_all = pd.DataFrame([])
    train_prediction_all = pd.DataFrame([])
    history_df_all = pd.DataFrame([])
    metric_df_all = pd.DataFrame([])
    metric_AL_Allrun = pd.DataFrame([])

    for trial in range(args.nrun):
        print('rnum : {}'.format(trial))
        args.seed = (trial * 55) % 2022 + 2# a different random seed for each run

        # data fetch
        # input: file path
        # output: data with DataFrame
        train_data, test_data, target = load_data.data_load(args.task_name, args.seed)

        # run model
        # input: train_data
        # output: metric, train_prediction, test_prediction
        metric_df = run(train_data, test_data, target, args, trial)

        metric_df_all = pd.concat([metric_df_all, metric_df], axis=0)
        local_time = time.strftime("%m_%d_%H_%M", time.localtime())
    metric_df_all.to_csv(f'./{args.task_name}_{local_time}.csv', index_label=['index'])
pass

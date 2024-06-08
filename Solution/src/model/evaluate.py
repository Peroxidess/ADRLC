import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, multilabel_confusion_matrix, \
    precision_score, recall_score, f1_score, r2_score, accuracy_score, balanced_accuracy_score, roc_curve, auc, \
    classification_report, matthews_corrcoef, fowlkes_mallows_score
import seaborn as sns
plt.rc('font', family='Times New Roman')


def mean_absolute_percentage_error(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def geometric_mean(y_true, y_pred):
    gm = y_true ** 2 + y_pred ** 2
    gm_ = gm ** 0.5
    return gm_


class Evaluate:
    def __init__(self, true, pred, data, eval_type, task_name, name_clf, nor_flag=False, nor_std=None):
        self.data = copy.deepcopy(data)
        self.eval_type = eval_type
        self.task_name = task_name
        self.name_clf = name_clf
        self.nor_flag = nor_flag
        self.nor_std = nor_std
        self.true, self.pred = self.nor_inver(true, pred)

    def nor_inver(self, true, pred):
        if self.nor_flag:
            data_inver = []
            for labelorpred in [true, pred]:
                self.data.loc[:, 'label0'] = labelorpred
                data_inver.append(self.nor_std.inverse_transform(self.data))
            true = data_inver[0][:, -1].reshape(-1, 1)
            pred = data_inver[1][:, -1].reshape(-1, 1)
        return true, pred


class Eval_Class(Evaluate):
    def __init__(self, true, pred, data, eval_type, task_name, name_clf, nor_flag=False, nor_std=None):
        Evaluate. __init__(self, true, pred, data, eval_type, task_name, name_clf, nor_flag, nor_std)
        if pred.shape[1] > 1:
            self.pred = np.argmax(pred, axis=1)
        else:
            self.pred = pred
        if len(true.shape) < 2 or true.shape[1] > 1:
            self.true = np.argmax(true, axis=1)
        else:
            self.true = true

    def eval(self):
        if np.unique(self.true).shape[0] < 3:
            auc_ = 0
            if 'class' in self.task_name:
                fpr, tpr, thresholds = roc_curve(self.true, self.pred)
                auc_ = auc(fpr, tpr)
                youden = (1 - fpr) + tpr - 1
                index_max = np.argmax(youden)
                threshold_max = thresholds[index_max]
                # threshold_max += 0.02
                if threshold_max > 1:
                    threshold_max = 0.5
        else:
            auc_ = 0
            threshold_max = 0.5
        pred = np.array([(lambda x: 0 if x < threshold_max else 1)(i) for i in self.pred])
        acc = accuracy_score(self.true, pred)
        acc_balanced = balanced_accuracy_score(self.true, pred)
        pre_weighted = precision_score(self.true, pred, average='weighted')
        pre_macro = precision_score(self.true, pred, average='macro')
        recall_weighted = recall_score(self.true, pred, average='weighted')
        recall_macro = recall_score(self.true, pred, average='macro')
        f1_weighted = f1_score(self.true, pred, average='weighted')
        f1_macro = f1_score(self.true, pred, average='macro')
        mcc = matthews_corrcoef(self.true, pred)
        fms = fowlkes_mallows_score(self.true.reshape(-1, ), pred.reshape(-1, ))
        # print(classification_report(self.true, pred))

        con_mat = confusion_matrix(self.true, pred)
        # sns.heatmap(con_mat, annot=True, fmt='g')
        # plt.show()
        metric_dict = dict(zip([
                                '{} acc'.format(self.eval_type),
                                '{} acc_balanced'.format(self.eval_type),
                                '{} pre_weighted'.format(self.eval_type),
                                '{} pre_macro'.format(self.eval_type),
                                '{} recall_weighted'.format(self.eval_type),
                                '{} recall_macro'.format(self.eval_type),
                                '{} f1_weighted'.format(self.eval_type), '{} f1_macro'.format(self.eval_type),
                                '{} auc_'.format(self.eval_type),
                                '{} mcc'.format(self.eval_type),
                                '{} fms'.format(self.eval_type),
                                ],
                               [acc, acc_balanced, pre_weighted, pre_macro, recall_weighted, recall_macro, f1_weighted,
                                f1_macro, auc_, mcc, fms]))
        return pd.DataFrame([metric_dict], index=[self.name_clf])


class Eval_Regre(Evaluate):
    def __init__(self, true, pred, data, eval_type, task_name, name_clf, nor_flag=False, nor_std=None):
        Evaluate. __init__(self, true, pred, data, eval_type, task_name, name_clf, nor_flag, nor_std)

    def eval(self):
        r2 = r2_score(self.true, self.pred)
        mse = mean_squared_error(self.true, self.pred)
        mae = mean_absolute_error(self.true, self.pred)
        mape = mean_absolute_percentage_error(self.true, self.pred)
        metric_dict = dict(zip(['{} r2'.format(self.eval_type), '{} mae'.format(self.eval_type),
                                '{} mse'.format(self.eval_type), '{} mape'.format(self.eval_type)],
                               [r2, mae, mse, mape]))
        return pd.DataFrame([metric_dict], index=[self.name_clf])


def mvi_rec_evalution(data_raw, index_manual_nan, data_filled, name_clf=0, name_metric='test'):
    data_raw = data_raw[data_filled.columns]
    if data_filled.shape[0] > data_raw.shape[0]:
        index_u = set(data_raw.index) & set(data_filled.index)
        data_filled = data_filled.loc[index_u]
        data_raw = data_raw.loc[index_u]
    elif data_filled.shape[0] == data_raw.shape[0]:
        data_filled.index = data_raw.index
    else:
        data_raw = data_raw.iloc[0: data_filled.shape[0]]
        data_filled.index = data_raw.index
        data_raw = data_raw.loc[data_filled.index]
    index_notna_ = index_manual_nan.loc[data_filled.index]

    data_raw_ = data_raw.fillna(data_filled)
    mae_byall = mean_absolute_error(data_raw_, data_filled)
    mse_byall = mean_squared_error(data_raw_, data_filled)
    mape_byall = mean_absolute_percentage_error(data_raw_, data_filled)
    rmse_byall = np.sqrt(mean_squared_error(data_raw_, data_filled))
    # index_notna_ = index_manual_nan
    index_notna = data_raw.notna()
    index_manul_nan = np.logical_xor(index_notna, index_notna_)
    data_raw_labeldrop = data_raw.drop(columns=data_raw.filter(regex=r'label').columns)
    data_filled_labeldrop = data_filled.drop(columns=data_filled.filter(regex=r'label').columns)
    mms = MinMaxScaler(feature_range=(0, 1))
    std = StandardScaler()
    nor = std
    data_raw_labeldrop = pd.DataFrame(nor.fit_transform(data_raw_labeldrop),
                                      columns=data_raw_labeldrop.columns,
                                      index=data_raw_labeldrop.index)
    data_filled_labeldrop = pd.DataFrame(nor.fit_transform(data_filled_labeldrop),
                                         columns=data_filled_labeldrop.columns,
                                         index=data_filled_labeldrop.index)
    mae_list = []
    mse_list = []
    rmse_list = []
    mape_list = []
    for index_, values in index_manul_nan.iteritems():
        data_raw_col = data_raw_labeldrop[[index_]][values == True]
        data_filled_col = data_filled_labeldrop[[index_]][values == True]
        if data_raw_col.empty:
            continue
        mae_bycol = mean_absolute_error(data_raw_col, data_filled_col)
        mse_bycol = mean_squared_error(data_raw_col, data_filled_col)
        mape_bycol = mean_absolute_percentage_error(data_raw_col, data_filled_col)
        rmse_bycol = np.sqrt(mean_squared_error(data_raw_col, data_filled_col))
        mae_list.append(mae_bycol)
        mse_list.append(mse_bycol)
        mape_list.append(mape_bycol)
        rmse_list.append(rmse_bycol)
    metric_dict = dict(zip([f'{name_metric} mae_bycol', f'{name_metric} mse_bycol', f'{name_metric} rmse_bycol', f'{name_metric} mae_byall', f'{name_metric} mse_byall', f'{name_metric} rmse_byall'],
                           [np.mean(mae_list), np.mean(mse_list), np.mean(rmse_list), mae_byall, mse_byall, rmse_byall]))
    metric_df = pd.DataFrame([metric_dict], index=[name_clf])
    print(f'metric{metric_df}')
    return metric_df


def plot_box(q_table, method='Ours', fold=''):
    sns.set(font_scale=1)
    ax = plt.plot()
    nor = MinMaxScaler()
    q_table['q values nor'] = nor.fit_transform(q_table['q values'].values.reshape(-1, 1))
    sns.boxplot(x='index', y='q values', data=q_table, hue=None, width=0.8, dodge=False)
    title = f'Q'
    plt.title(title, fontsize=9, y=0.98)
    plt.ylabel(f'q values', fontsize=9)
    # ax.yaxis.set_label_coords(-0.12, 0.5)
    plt.xlabel('')

    plt.legend(bbox_to_anchor=(1.15, 0.8), ncol=3, fontsize=9)
    # plt.subplots_adjust(wspace=0.23)
    # plt.subplots_adjust(hspace=0.25)
    # plt.savefig(f'./{method}_{title}_{fold}.png', dpi=334, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()
    pass


def features_plot(features, feature_cluster=None, model=None, flag_='tsne', order=0, method_name_AL='', seed=2022):
    if 'flat' in flag_:
        list_minmax = []
        for n in range(features.shape[1]):
            list_minmax.append([features[:, n].min() - 1, features[:, n].max() + 1])
        from itertools import product
        list_meshgrid = []
        list_1 = np.arange(list_minmax[0][0], list_minmax[0][1], (list_minmax[0][1] - list_minmax[0][0]) / 100)
        list_2 = np.arange(list_minmax[1][0], list_minmax[1][1], (list_minmax[1][1] - list_minmax[1][0]) / 100)
        for x in product(list_1, list_2):
            list_meshgrid.append(x)
        y_meshgrid, _ = model.predict(list_meshgrid)
        y_meshgrid = y_meshgrid[:, 0].reshape(-1, 1)
        # yy[np.where(yy > 0)] = 1
        # yy[np.where(yy < 0)] = 0
        y, _ = model.predict(features)
        y = y[:, 0].reshape(-1, 1)
        features_meshgrid = np.hstack((np.array(list_meshgrid), y_meshgrid))
        features_raw = np.hstack((features, y))
        features_ = np.vstack((features_meshgrid, features_raw))
        feature_cluster_ = np.vstack((np.zeros(shape=(features_meshgrid.shape[0], 1)), feature_cluster.reshape(-1, 1) + 1))
    else:
        features_ = features
        feature_cluster_ = feature_cluster * 1

    point_size = np.ones(shape=(feature_cluster.shape[0], 1))

    if 'tsne' in flag_:
        tsne = TSNE(n_components=2,
                    init='pca',
                    random_state=seed)
        x = tsne.fit_transform(features_)
        point_size[np.where(feature_cluster_ != 2)] = 16
        point_size[np.where(feature_cluster_ == 2)] = 72
    elif 'pca' in flag_:
        pca = PCA(n_components=2, whiten=True)
        x = pca.fit_transform(features_)
    elif 'raw' in flag_:
        x = features_
        point_size[np.where(feature_cluster != 0)] = 32
        point_size[np.where(feature_cluster == 0)] = 32
    if x.shape[1] > 2:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        feature_cluster_c = pd.DataFrame(feature_cluster).replace({0: 'red', 1: 'purple', 2: 'blue'}).values.reshape(-1, )
        feature_cluster_alpha = pd.DataFrame(feature_cluster_).replace({0: 0.2, 1: 0.9, 2: 0.9}).values.reshape(-1, )
        feature_cluster_marker = pd.DataFrame(feature_cluster_).replace({0: '.', 1: '.', 2: '.'}).values.reshape(-1, )
        scatter = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=feature_cluster_c, label=['1', '2'], marker='s', s=point_size)
        ax.legend(*scatter.legend_elements())
        # scatter = ax.scatter(features_meshgrid[:, 0], features_meshgrid[:, 1], features_meshgrid[:, 2], c='oldlace', alpha=0.1)
        # ax.legend(*scatter.legend_elements())

    else:
        scatter = plt.scatter(x=x[:, 0], y=x[:, 1], c=feature_cluster_, s=point_size, marker='.', cmap='winter')
        plt.legend(*scatter.legend_elements())
    # scatter = plt.scatter(x=y[:, 0], y=y[:, 1], c=y_cluster, cmap=plt.cm.Spectral, marker='v')
    plt.title(f'{flag_}_{order}')
    # plt.show()
    plt.savefig(f'features_plot_{method_name_AL}_{order}.png')
    plt.close()
    pass


def q_trend_plot(q_table, name_col_list, fold=''):
    q_table.sort_values(by=name_col_list[0], inplace=True)
    q_table.reset_index(drop=True, inplace=True)
    for name_col in name_col_list:
        q_ = q_table[name_col]
        nor = MinMaxScaler(feature_range=(0, 1))
        q_nor = nor.fit_transform(q_.values.reshape(-1, 1))
        plt.plot(q_.index.values, q_nor)
    plt.close()


def q_plot(q_table, name_groupby, name_method='', fold='', flag_save=True):
    def initPlot():
        sns.set(font_scale=1.2, style="white", palette="muted",
                color_codes=True, font='Times New Roman')  # set( )设置主题，调色板更常用
        # mpl.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题

    initPlot()
    data_col = q_table[name_groupby].values.reshape(-1,)
    set_data_col = set(data_col)
    bins = max(int(np.floor(len(set_data_col) / 4)), 2)
    histogram_data_col = np.histogram(data_col, bins=bins)

    histogram_per = pd.Series(histogram_data_col[1])
    histogram_post = pd.Series(histogram_data_col[1]).shift()
    histogram_ave_ = (histogram_per + histogram_post) / 2
    histogram_ave_ = histogram_ave_.dropna(axis=0, how='any')
    data_col_bins = pd.cut(data_col, bins=bins, labels=histogram_ave_, ordered=False)
    data_col_bins_df = pd.DataFrame(data_col_bins, index=q_table.index, columns=name_groupby)
    q_table['q_round'] = data_col_bins_df
    q_groupby = q_table.groupby(by=['q_round']).mean()
    q_groupby_count = q_table.groupby(by=['q_round']).count()
    q_groupby_count['label1_ave'] = q_groupby_count['label1'] / q_groupby_count['label1'].sum()
    q_groupby['index'] = q_groupby.index

    # sns.histplot(q_table[q_table['label1'] == 0], x=name_groupby[0], kde=True, bins=70, label='Non-relapse', color='blue')
    # sns.histplot(q_table[q_table['label1'] == 1], x=name_groupby[0], kde=True, bins=70, label='Relapse', color='orange')
    # name_tmp = name_groupby[0].split('_')
    # plt.title(f'Record Action')
    # plt.legend()
    # plt.xlabel('Q')
    # plt.show()
    # plt.savefig(f'{name_method}_q_with_class_{name_groupby[0]}_{fold}_new.png', dpi=456, bbox_inches='tight', transparent=True)
    # plt.close()

    metric_list = ['q_record', 'q', 'q_diff']
    sns.set_style('ticks')
    model = {}
    for metric in metric_list:
        if metric in ['q', 'q_diff']:
            pf = model['q_record']
        elif metric in ['action', 'action_diff']:
            pf = model['action_raw']
        else:
            pf = make_pipeline(PolynomialFeatures(degree=6), Ridge())
            pf.fit(q_table[f'{metric}'].values.reshape(-1, 1), q_table['label1'].values.reshape(-1, 1))
        if metric == 'q_record':
            model['q_record'] = pf
        if metric == 'action_raw':
            model['action_raw'] = pf
        if metric == 'action_diff':
            q_table['action_diff'] = abs(q_table['action_diff'])
        ratio = pf.predict(q_table[f'{metric}'].values.reshape(-1, 1))
        print(f'mortality_ratio_{metric} mean {ratio.mean()} std {ratio.std()}')
        q_table[f'mortality_ratio_{metric}'] = ratio
        sns.regplot(data=q_table, x=f'{metric}', y=f'mortality_ratio_{metric}', order=6, ci=True, truncate=True, n_boot=1500,
                    x_estimator=np.mean, scatter=False)
        plt.ylabel('mortality')
        metric_plot = metric.replace('_', ' ')
        plt.xlabel(f'{metric_plot}')
        plt.show()
        # plt.savefig(f'{name_method}_{metric}_mortality_{fold}.png', dpi=334)
        # plt.close()
    if flag_save:
        # q_table.to_csv(f'./q_table_test_mortality_ratio_{name_method}_fold{fold}.csv', index=False)
        pass
    return q_table


def RL_history_plot(loss_history_list, label_list, method_name_RL='', flod=''):
    loss_history_ = []
    for loss_history, label in zip(loss_history_list, label_list):
        loss_history.fillna(method='ffill', inplace=True)
        loss_history.fillna(method='bfill', inplace=True)
        metric_trend_mean = loss_history.mean(axis=1)
        metric_trend_std = loss_history.std(axis=1)
        plt.plot(metric_trend_mean, label=label)
        plt.fill_between(list(range(metric_trend_mean.shape[0])), metric_trend_mean - metric_trend_std,
                         metric_trend_mean + metric_trend_std, alpha=0.2)
    plt.title(f'{method_name_RL}')
    plt.legend()
    plt.savefig(f'{method_name_RL}_history_plot_{label_list[0]}_{flod}.png', dpi=334)
    # plt.show()
    plt.close()
    pass


def history_plot(loss_history_list, label_list, method_name_RL='', fold=''):
    loss_history_ = []
    for label in label_list:
        for loss_history in loss_history_list:
            if type(loss_history) == list:
                for loss in loss_history:
                    loss_ = loss[label].mean()
                    loss_history_.append(loss_)
                loss_history_ = pd.DataFrame(loss_history_, columns=[label])
            else:
                loss_history_ = loss_history[[label]]

            loss_history_.fillna(method='ffill', inplace=True)
            loss_history_.fillna(method='bfill', inplace=True)
            if loss_history_.shape[1] > 1:
                loss_history_mean = loss_history_.mean(axis=1)
                metric_history_std = loss_history_.std(axis=1)
            else:
                loss_history_mean = loss_history_.values
            plt.plot(loss_history_mean, label=method_name_RL)
            if loss_history_.shape[1] > 1:
                plt.fill_between(list(range(loss_history_mean.shape[0])), loss_history_mean - metric_history_std,
                                 loss_history_mean + metric_history_std, alpha=0.2)
    plt.title(f'{label}')
    plt.legend()
    # plt.savefig(f'{method_name_RL}_history_plot_{label_list[0]}_{fold}.png', dpi=334)
    # plt.show()
    plt.close()
    pass


def plot_corr(data):
    plt.figure(figsize=(24, 20))
    data_corr = data.corr()
    sns.heatmap(data_corr)
    plt.show()


def plot_pair(train_data, test_data=pd.DataFrame([]), label=pd.DataFrame([]), columns=None):
    if not test_data.empty:
        train_data['tag'] = 0
        test_data['tag'] = 1
        data = pd.concat([train_data, test_data], axis=0)
    else:
        train_data['tag'] = 0
        data = train_data
    if not label.empty:
        data_plot = pd.concat([data.loc[:, columns], label], axis=1)
    else:
        data_plot = data
    data_plot.drop(columns=data_plot.filter(regex=r'病例编号|次序').columns, inplace=True)
    col_1 = data_plot.filter(regex=r'功率').columns.values[0]
    col_2 = data_plot.filter(regex=r'消融范围').columns.values[0]
    col_3 = data_plot.filter(regex=r'消融范围（宽）').columns.values[0]
    col_4 = data_plot.filter(regex=r'消融范围（高）').columns.values[0]
    col_5 = data_plot.filter(regex=r'肿瘤（长）').columns.values[0]
    col_6 = data_plot.filter(regex=r'肿瘤（宽）').columns.values[0]
    col_7 = data_plot.filter(regex=r'肿瘤（高）').columns.values[0]
    data_plot.rename(columns={col_1: 'power', col_2: 'ablation', col_3: 'ablation_wide', col_4: 'ablation_height',
                              col_5: 'tumour_length', col_6: 'tumour_width', col_7: 'tumour_height'}, inplace=True)
    nacount = data_plot.describe()
    sns.set(font_scale=0.6)
    fig = sns.pairplot(data_plot, diag_kind='kde', hue='tag')
    plt.yticks(rotation=45)
    # fig.map_diag(sns.kdeplot)
    # fig.map_offdiag(sns.kdeplot)
    # plt.show()
    plt.savefig('pair_plot.png', dpi=334, bbox_inches='tight')
    pass


def plot_esd(data):
    mns = MinMaxScaler(feature_range=(0, 1))
    data = pd.DataFrame(mns.fit_transform(data), index=data.index, columns=data.columns)
    data_describe = data.describe()
    data_E = data_describe.loc['mean', :]
    data_SD = data_describe.loc['std', : ]
    plt.scatter(x=data_E, y=data_SD)
    plt.show()

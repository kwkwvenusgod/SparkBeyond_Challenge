import sklearn
import datetime, sys,os
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, \
    RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, Merge, regularizers, LSTM, TimeDistributed, GlobalAveragePooling2D
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import Adamax
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.convolutional import MaxPooling2D, MaxPooling1D
from CNN_LSTM import CNN_LSTM
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
import custom_metrics
import pandas as pd
import timeit
import numpy as np
import ml_metrics
import cPickle as Pickle
import itertools, pickle
import math
import random
from sklearn.pipeline import Pipeline
from collections import Counter
import inspect
import copy


def load_model_bin(model_file):
    model_file = open(model_file, 'rb')
    return Pickle.load(model_file)


def save_model_bin(model, model_file):
    model_file = open(model_file, 'wb')
    Pickle.dump(model, model_file, Pickle.HIGHEST_PROTOCOL)


def load_pd_df(file_name):
    ret_val = pd.read_csv(file_name)
    return ret_val


def to_2dim(array_val):
    return np.array(array_val, ndmin=2).transpose()


def nth(iterable, n):
    return next(itertools.islice(iterable, n, None))


def avg_eval_metric(eval_metric, test_y, prediction, metric_type):
    if prediction.shape[1] == 1:
        return eval_metric(test_y, prediction[:, 0])
    elif prediction.shape[1] == 2:
        return eval_metric(test_y, prediction[:, 1])
    else:
        metric_val = 0.0
        metric_count = 0.0
        if metric_type == "cumulative":
            cur_pred = np.zeros(prediction.shape[0])
            for c in xrange(prediction.shape[1] - 1):
                cur_actual = np.array(np.array(test_y) <= c).astype(int)
                cur_pred += prediction[:, c]
                metric_val += eval_metric(cur_actual, cur_pred)
                metric_count += 1.0
        else:
            for c in xrange(prediction.shape[1]):
                cur_actual = np.array(np.array(test_y) == c).astype(int)
                metric_val += eval_metric(cur_actual, prediction[:, c])
                metric_count += 1.0
        if metric_type == "sum":
            metric_count = 1.0
        return metric_val / metric_count


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def data_filter(data, filter_dict):
    if len(filter_dict) > 0:
        for filter_col in filter_dict:
            data = data[data[filter_col] == filter_dict[filter_col]]
    return data


def print_stages(test_y, stage_predictions, test_metric, metric_type, test_weight):
    if hasattr(custom_metrics, test_metric):
        eval_metric = getattr(custom_metrics, test_metric)
    elif hasattr(ml_metrics, test_metric):
        eval_metric = getattr(ml_metrics, test_metric)
    else:
        eval_metric = getattr(sk_metrics, test_metric)
    if test_weight is not None:
        metric_args = inspect.getargspec(eval_metric)[0]
        if 'weight' in metric_args:
            eval_metric_orig = eval_metric
            eval_metric = lambda act, pred: eval_metric_orig(act, pred, test_weight)
    count = 0
    iters = []
    loss = []
    count_factor = 50
    for prediction in stage_predictions:
        count += 1
        if count in [1, 10, 50] or count % count_factor == 0:
            iters.append(count)
            loss.append(avg_eval_metric(eval_metric, test_y, prediction, metric_type=metric_type))
        if count > 1000:
            count_factor = 500
        elif count > 500:
            count_factor = 200
        elif count > 250:
            count_factor = 100
    loss_df = pd.DataFrame({'Iteration': iters, 'Loss': loss})
    loss_df.rename(columns={'Loss': test_metric}, inplace=True)
    pd.set_option('max_columns', len(loss_df.columns))
    pd.set_option('max_rows', len(loss_df))
    print("Loss:")
    print(loss_df)


def process_date(date_list):
    date = []
    for tmp in date_list:
        tmp = tmp.split('-')
        week = str(int(tmp[2])/7)
        tmp = tmp[1] + '-' + week
        date.append(tmp)
    return date


def transfer_list(input_list):
    n_samples = len(input_list)
    n_features = len(input_list[0])
    res = np.zeros((n_features, n_samples))

    for i in range(n_samples):
        tmp = np.asarray(input_list[i])
        if tmp.dtype != 'object':
            res[:, i] = tmp
        else:
            print tmp
    return res


def process_spray_data(input_data, spray_data, add_spray = False):
    # feature_list = ["Date", "Species", "Longitude", "Latitude", "Trap", "Address", "Street", 'Tmax', 'Tmin',
    #                 'DewPoint', "WetBulb", "Heat", "Cool", "Sunrise", "Sunset",
    #                 "StnPressure", "SeaLevel", "ResultSpeed", "ResultDir", "AvgSpeed", "NumMosquitos"]
    feature_list = ["Date", "Species", "Longitude", "Latitude", "Trap", 'Tmax', 'Tmin',
                     'DewPoint', "WetBulb", "Heat", "Cool", "Sunrise", "Sunset",
                    "StnPressure", "SeaLevel", "ResultSpeed", "ResultDir", "AvgSpeed", "NumMosquitos"]

    input_data = input_data[feature_list]
    if add_spray is True:
        spray_indicator = np.zeros(len(input_data))
        thres_date = 5
        thres_area = 0.6
        train_data_dates = input_data["Date"].drop_duplicates().tolist()
        spray_la_long_list = list(spray_data[["Latitude", "Longitude"]].groupby(spray_data["Date"]))
        train_data_la_long_list = list(input_data[["Latitude", "Longitude"]].groupby(input_data["Date"]))
        for spray_la_long_date in spray_la_long_list:
            spray_date_tmp_start = datetime.datetime.strptime(spray_la_long_date[0], '%Y-%m-%d')
            spray_date_tmp_end = spray_date_tmp_start + datetime.timedelta(days=thres_date)
            spray_location_tmp = spray_la_long_date[1].get_values()

            for i in range(len(train_data_dates)):
                train_data_date = datetime.datetime.strptime(train_data_dates[i], '%Y-%m-%d')
                if (train_data_date > spray_date_tmp_start) & (train_data_date <= spray_date_tmp_end):
                    train_location = train_data_la_long_list[i][1].get_values()
                    train_location_indices = train_data_la_long_list[i][1].axes[0].get_values()
                    distances = euclidean_distances(spray_location_tmp, train_location)
                    indices = np.unique(np.where(distances < thres_area)[1], return_index=False).astype(int)
                    indices = train_location_indices[indices].astype(int)
                    spray_indicator[indices] = 1
        input_data['Spray'] = pd.Series(list(spray_indicator), index=input_data.index)
        feature_list.append('Spray')

    input_data.loc[:, 'Date'] = process_date(input_data['Date'].values)
    input_data['StnPressure'].loc[input_data['StnPressure'] == 'M'] = float('NaN')
    input_data.loc[:, 'StnPressure'] = map(float, input_data['StnPressure'].values)

    input_data['AvgSpeed'].loc[input_data['AvgSpeed'] == 'M'] = float('NaN')
    input_data.loc[:, 'AvgSpeed'] = map(float, input_data['AvgSpeed'].values)

    input_data['WetBulb'].loc[input_data['WetBulb'] == 'M'] = float('NaN')
    input_data.loc[:, 'WetBulb'] = map(float, input_data['WetBulb'].values)

    input_data['Heat'].loc[input_data['Heat'] == 'M'] = float('NaN')
    input_data.loc[:, 'Heat'] = map(float, input_data['Heat'].values)

    input_data['Cool'].loc[input_data['Cool'] == 'M'] = float('NaN')
    input_data.loc[:, 'Cool'] = map(float, input_data['Cool'].values)

    input_data['Sunrise'].loc[input_data['Sunrise'] == 'M'] = float('NaN')
    input_data.loc[:, 'Sunrise'] = map(float, input_data['Sunrise'].values)

    input_data['Sunset'].loc[input_data['Sunset'] == 'M'] = float('NaN')
    input_data.loc[:, 'Sunset'] = map(float, input_data['Sunset'].values)

    input_data['SeaLevel'].loc[input_data['SeaLevel'] == 'M'] = float('NaN')
    input_data['SeaLevel'] = pd.Series(map(float, input_data['SeaLevel'].values), index=input_data.index)

    total_mosq = np.sum(input_data['NumMosquitos'].fillna(0).values)
    perc_mosq = np.asarray(input_data['NumMosquitos'].fillna(0).values) / total_mosq
    perc_mosq = perc_mosq / np.max(perc_mosq)
    input_data['NumMosquitos'] = pd.Series(perc_mosq, index=input_data.index)



    # more features
    return input_data, feature_list


class WNV:
    def __init__(self, _input_dir, _train_data_file, _target_col,
                 _model_type='GradientBoostingRegressor',
                 _fit_args={"n_estimators": 10, "learning_rate": 0.001, "loss": "ls",
                           "max_features": 5, "max_depth": 7, "random_state": 788954,
                           "subsample": 1, "verbose": 50}, _test_metric='normalized_weighted_gini', _na_fill_value=-20000,
                 _silent=False, _skip_mapping=False, _load_model=None, _train_filter=None, _metric_type='auto',
                 _load_type='fit_more',_features=None,_data_balance=False,_feature_mapping_dict={},_feature_transform_=None,
                 _feature_size=[],_model=None,_staged_predict=None,_have_feat_importance=False,_batch_size=34,_predict=None,
                 _bootstrap=0, _bootstrap_seed=None, _weight_col=None, _feature_mode='label',_model_path=None):
        self._input_dir = _input_dir
        self._train_data_file = _train_data_file
        self._target_col = _target_col
        self._model_type = _model_type
        self._fit_args = _fit_args
        self._test_metric = _test_metric
        self._na_fill_value = _na_fill_value
        self._silent = _silent
        self._skip_mapping = _skip_mapping
        self._load_model = _load_model
        self._bootstrap = _bootstrap
        self._bootstrap_seed = _bootstrap_seed
        self._weight_col = _weight_col
        self._train_filter = _train_filter
        self._metric_type = _metric_type
        self._load_type = _load_type
        self._features = _features
        self._model = _model
        self._staged_predict = _staged_predict
        self._have_feat_importance = _have_feat_importance
        self._predict = _predict
        self._data_balance = _data_balance
        self._feature_mapping_dict = _feature_mapping_dict
        self._feature_transform_ = _feature_transform_
        self._feature_mode = _feature_mode
        self._feature_size = _feature_size
        self._batch_size = _batch_size
        self._model_path = _model_path

    def staged_pred_proba(self, x):
        for pred in self._model.staged_predict_proba(x):
            yield self.prob_pred(pred)

    def staged_pred_proba_at_n(self, x, n):
        return nth(self.staged_pred_proba(x=x), n)

    def pred_proba(self, x):
        return self.prob_pred(self._model.predict_proba(X=x))

    def prob_pred(self, pred):
        return pred

    def staged_pred_continuous(self, x):
        for pred in self._model.staged_predict(x):
            yield to_2dim(pred)

    def staged_pred_continuous_at_n(self, x, n):
        return nth(self.staged_pred_continuous(x=x), n)

    def continuous_predict(self, x):
        return to_2dim(self._model.predict(X=x))

    def generator(self, train_x, train_y, indices_list, max_len):
        n_step = int(train_x.shape[0]/self._batch_size)
        flag = False
        while 1:
            for i in range(n_step):
                feature_train = np.zeros((self._batch_size, train_x.shape[1], max_len))
                y_train_generator = np.zeros((self._batch_size, 1))

                for j in range(self._batch_size):
                    feature_tmp = np.zeros((train_x.shape[1], max_len))
                    if i * self._batch_size + j < train_x.shape[0]:
                        ind = indices_list[i * self._batch_size + j]
                        feat_len = len(ind)
                        start = max_len - feat_len
                        feature_tmp[:, start:] = train_x[ind, :].transpose()
                        y_train_generator[j, 0] = train_y[i * self._batch_size + j]
                        feature_train[j] = feature_tmp
                    else:
                        flag = True
                        break
                if flag:
                    feature_train = feature_train[i * self._batch_size:train_x.shape[0]]
                    y_train_generator = train_y[i * self._batch_size:train_x.shape[0]]
                feature_train = np.reshape(feature_train, (self._batch_size, 1, train_x.shape[1], max_len, 1))
                yield feature_train, y_train_generator

    def feature_generator(self,train_x,  indices_list, max_len):
        n_step = int(train_x.shape[0] / self._batch_size)
        flag = False
        for i in range(n_step):
            feature_train = np.zeros((self._batch_size, train_x.shape[1], max_len))
            for j in range(self._batch_size):
                feature_tmp = np.zeros((train_x.shape[1], max_len))
                if i * self._batch_size + j < train_x.shape[0]:
                    ind = indices_list[i * self._batch_size + j]
                    feat_len = len(ind)
                    if feat_len > max_len:
                        ind = ind[0:max_len]
                        print ind
                        feature_tmp = train_x[ind, :].transpose()
                        feature_train[j] = feature_tmp
                    else:
                        feat_len = len(ind)
                        start = max_len - feat_len
                        feature_tmp[:, start:] = train_x[ind, :].transpose()
                        feature_train[j] = feature_tmp
                else:
                    flag = True
                    break
            if flag:
                feature_train = feature_train[i * self._batch_size:train_x.shape[0]]
            yield np.reshape(feature_train,(self._batch_size,1,train_x.shape[1],max_len,1))

    def process_date_list(self, date_list):
        all_dates = date_list
        time_period = 3
        indices = []
        max_len = 0
        for select_data in all_dates:
            tmp_date = select_data - datetime.timedelta(days=time_period)
            ind = np.where((all_dates > tmp_date) & (all_dates <= select_data))[0]
            if len(ind) > max_len:
                max_len = len(ind)
            indices.append(ind)

        return indices, max_len

    def save_model(self, dirPath='.'):
        if self._model_type in  ['LSTM', 'CNN']:
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            self._model_path = dirPath + '/'+ self._model_type + '.h5'
            self._model.save(self._model_path)
            self._model = None
            wvn_model_path = dirPath + '/' + 'wvn.pickle'
            with open(wvn_model_path, 'wb') as wvn_model:
                tmp = self.__dict__
                pickle.dump(tmp, wvn_model, pickle.HIGHEST_PROTOCOL)
        else:
            wvn_model_path = dirPath + '/' + 'wvn.pickle'
            with open(wvn_model_path, 'wb') as wvn_model:
                pickle.dump(self, wvn_model, pickle.HIGHEST_PROTOCOL)

    def load_real_model(self, file_path):
        if self._model_type == 'LSTM':
            model = CNN_LSTM((self._feature_size[0], 4), (None, self._feature_size[0], self._feature_size[1], 1))
            model.load(file_path)
            self._model = model
        else:
            print("Only keras model need the method")

    def train(self):
        start = timeit.default_timer()
        train_x, train_y, feature_list = self.feature_extraction()
        self._feature_size = [train_x.shape[1],1]
        self._features = feature_list


        if not self._silent:
            print "Train has %d instances " % (len(train_x))

        counts = Counter(train_y)
        expectation_ratio = 1 / float(len(counts.keys()))
        n_samples = len(train_y)
        for key, value in counts.items():
            tmp = float(expectation_ratio) / (float(value) / float(n_samples))
            if (tmp > 6) | (tmp < (1.0 / 6.0)):
                self._data_balance = True

        extra_fit_args = dict()
        if self._weight_col is not None:
            extra_fit_args['sample_weight'] = train_x[self._weight_col].values
            del train_x[self._weight_col]

        if 0 < self._bootstrap < 1.0:
            if self._bootstrap_seed is not None:
                if not self._silent:
                    print "Setting bootstrap seed to %d" % self._bootstrap_seed
                np.random.seed(self._bootstrap_seed)
                random.seed(self._bootstrap_seed)
            bootstrap_len = int(math.floor(self._bootstrap * len(train_x)))
            bootstrap_ix = random.sample(range(len(train_x)), bootstrap_len)
            train_x = train_x.iloc[bootstrap_ix]
            train_x.reset_index()
            train_y = train_y.iloc[bootstrap_ix]
            train_y.reset_index()

        model = None

        if self._model_type == "RandomForestRegressor":
            if model is None:
                if self._data_balance is True:
                    self._fit_args.update({"class_weight": "balanced"})
                model = RandomForestRegressor(**self._fit_args)
                model.fit(X=train_x, y=train_y, **extra_fit_args)
                self._model = model
                self._predict = lambda (fitted_model, pred_x): self.continuous_predict(x=pred_x)
                self._have_feat_importance = True

        elif self._model_type == "RandomForestClassifier":
            if model is None:
                # if self._data_balance is True:
                #     self._fit_args.update({"class_weight": "balanced"})
                model = RandomForestClassifier(**self._fit_args)
                model.fit(X=train_x, y=train_y, **extra_fit_args)
                self._model = model
                self._predict = lambda (fitted_model, pred_x): self.pred_proba(x=pred_x)
            self._staged_predict = lambda (fitted_model, pred_x): [self._predict((fitted_model, pred_x))]
            self._have_feat_importance = True

        elif self._model_type == "ExtraTreesRegressor":
            if model is None:
                if self._data_balance is True:
                    self._fit_args.update({"class_weight": "balanced"})
                model = ExtraTreesRegressor(**self._fit_args)
                model.fit(X=train_x, y=train_y, **extra_fit_args)
                self._model = model
                self._predict = lambda (fitted_model, pred_x): self.continuous_predict(x=pred_x)
                self._have_feat_importance = True

        elif self._model_type == "ExtraTreesClassifier":
            if model is None:
                if self._data_balance is True:
                    self._fit_args.update({"class_weight": "balanced"})
                model = ExtraTreesClassifier(**self._fit_args)
                model.fit(X=train_x, y=train_y, **extra_fit_args)
            self._predict = lambda (fitted_model, pred_x): self.pred_proba(x=pred_x)
            self._staged_predict = lambda (fitted_model, pred_x): [self._predict((fitted_model, pred_x))]
            self._have_feat_importance = True

        elif self._model_type == "GradientBoostingRegressor":
            if model is None:
                model = GradientBoostingRegressor(**self._fit_args)
                model.fit(X=train_x, y=train_y, **extra_fit_args)
                self._model = model
            elif self._load_type == "fit_more":
                model.warm_start = True
                model.n_estimators += self._fit_args['n_estimators']
                model.fit(X=train_x, y=train_y)
                self._model = model
            self._predict = lambda (fitted_model, pred_x): self.continuous_predict(x=pred_x)
            self._staged_predict = lambda (fitted_model, pred_x): self.staged_pred_continuous(x=pred_x)
            if self._load_type == "pred_at" and self._fit_args['n_estimators'] < model.n_estimators:
                if not self._silent:
                    print ("Predict using %d trees" % self._fit_args['n_estimators'])
                self._predict = lambda (fitted_model, pred_x): self.staged_pred_continuous_at_n(x=pred_x,
                                                                                                n=self._fit_args[
                                                                                                    'n_estimators'])
        elif self._model_type == "GradientBoostingClassifier":
            if model is None:
                model = GradientBoostingClassifier(**self._fit_args)
                model.fit(X=train_x, y=train_y, **extra_fit_args)
                self._model = model
            elif self._load_type == "fit_more":
                model.warm_start = True
                model.n_estimators += self._fit_args['n_estimators']
                model.fit(X=train_x, y=train_y)
                self._model = model
                self._staged_predict = lambda (fitted_model, pred_x): self.staged_pred_proba(x=pred_x)
            self._predict = lambda (fitted_model, pred_x): self.pred_proba(x=pred_x)
            if self._load_type == "pred_at" and self._fit_args['n_estimators'] < model.n_estimators:
                if not self._silent:
                    print ("Predict using %d trees" % self._fit_args['n_estimators'])
                self._predict = lambda (fitted_model, pred_x): self.staged_pred_proba_at_n(x=pred_x,
                                                                                           n=self._fit_args[
                                                                                               'n_estimators'])
        elif self._model_type == "LogisticRegression":
            if model is None:
                if self._data_balance is True:
                    self._fit_args.update({"class_weight": "balanced"})
                model = LogisticRegression(**self._fit_args)
                model.fit(X=train_x, y=train_y)
                self._model = model
            self._predict = lambda (fitted_model, pred_x): self.pred_proba(x=pred_x)
            self._staged_predict = lambda (fitted_model, pred_x): [self._predict((fitted_model, pred_x))]

        elif self._model_type == "SVC":
            if model is None:
                if self._data_balance is True:
                    self._fit_args.update({"class_weight": "balanced"})
                model = sklearn.svm.SVC(**self._fit_args)
                model.fit(X=train_x, y=train_y)
                self._model = model
            self._predict = lambda (fitted_model, pred_x): self.pred_proba(x=pred_x)
            self._staged_predict = lambda (fitted_model, pred_x): [self._predict((fitted_model, pred_x))]
        elif self._model_type == "CNN":
            if model is None:
                train_data = load_pd_df(self._input_dir + '/train.csv')
                indices, max_len = self.process_date_list(
                    train_data['Date'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')))
                self._feature_size = [train_x.shape[1], max_len]

                NB_FILTER = [64, 128]
                NB_Size = [4, 3, 3]
                FULLY_CONNECTED_UNIT = 256
                model = Sequential()
                model.add(Conv2D(NB_FILTER[0], (train_x.shape[1], NB_Size[0]), input_shape=train_x.shape, border_mode='valid',
                                 activation='relu'))
                model.add(MaxPooling2D(pool_size=(1, 3)))
                model.add(
                    Conv2D(NB_FILTER[1], (1, NB_Size[1]), border_mode='valid'))
                model.add(MaxPooling2D(pool_size=(1, 3)))
                model.add(Flatten())
                model.add(Dense(FULLY_CONNECTED_UNIT, activation='relu', W_constraint=maxnorm(3),
                                kernel_regularizer=regularizers.l2(0.01)))
                model.add(Dense(2, activation='softmax'))
                model.compile(loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])
                model.fit(train_x, train_y, batch_size=16, epochs=50, verbose=1)
        elif self._model_type == "LSTM":
            if model is None:
                train_data = load_pd_df(self._input_dir + '/train.csv')
                indices, max_len = self.process_date_list(
                    train_data['Date'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')))
                self._feature_size = [train_x.shape[1], max_len]

                # class_weight = {1: np.divide(float(n_samples) , float((len(counts) * counts[1]))),
                #                 0: np.divide(float(n_samples) , float((len(counts) * counts[0])))}
                class_weight = {1: 10,
                                0: 1}
                model = CNN_LSTM((self._feature_size[0],4), (None,self._feature_size[0],self._feature_size[1],1))
                model.fit_generator(generator=self.generator(train_x, train_y, indices, max_len),
                                    epochs=20, class_weight=class_weight, steps_per_epoch=train_x.shape[0]/self._batch_size)
                # model.fit_generator(generator=self.generator(train_x, train_y, indices, max_len),
                #                     epochs=1, class_weight=class_weight, steps_per_epoch=1)
                self._model = model

        elif self._model_type == "Pipeline":
            if model is None:
                model = Pipeline([
                    ('pre_process',
                     get_class(self._fit_args['pre_process']['name'])(self._fit_args['pre_process']['args'])),
                    ('model', get_class(self._fit_args['model']['name'])(self._fit_args['model']['args']))
                ])
                model.fit(X=train_x, y=train_y)
                self._model = model
            self._predict = lambda (fitted_model, pred_x): self.pred_proba(x=pred_x)
            self._staged_predict = lambda (fitted_model, pred_x): [self._predict((fitted_model, pred_x))]

        if not self._silent:
            stop = timeit.default_timer()
            print "Train time: %d s" % (stop - start)

        del train_x, train_y

    def predict_test(self, test_data_file, output_file=None):

        test_x = load_pd_df(test_data_file)
        mappings = None if self._skip_mapping else dict()
        if mappings is not None:
            for col in test_x.columns:
                if col in mappings:
                    test_x[col] = test_x[col].map(mappings[col]).fillna(self._na_fill_value)
                else:
                    test_x[col] = test_x[col].fillna(self._na_fill_value)
        test_y = None
        if self._target_col in test_x.columns:
            test_y = test_x[self._target_col][test_x[self._target_col] != self._na_fill_value]
            test_y2 = test_x[self._target_col][pd.notnull(test_x[self._target_col])]
            if len(test_y) != len(test_x) or len(test_y2) != len(test_x):
                test_y = None
            del test_y2

        test_weight = None
        if self._weight_col is not None:
            if self._weight_col in test_x.columns:
                test_weight = test_x[self._weight_col]
                del test_x[self._weight_col]

        test_x = test_x[self._features]

        test_pred = self._predict((self._model, test_x))
        if test_pred.shape[1] == 1:
            test_pred = pd.DataFrame({'pred': test_pred[:, 0]})
        elif test_pred.shape[1] == 2:
            test_pred = pd.DataFrame({'pred': test_pred[:, 1]})
        else:
            test_pred_df = None
            for c in xrange(test_pred.shape[1]):
                if test_pred_df is None:
                    test_pred_df = pd.DataFrame({'pred0': test_pred[:, c]})
                else:
                    test_pred_df['pred' + str(c)] = test_pred[:, c]
            test_pred = test_pred_df

        if not self._silent and test_y is not None:
            print_stages(test_y=test_y, stage_predictions=self._staged_predict((self._model, test_x)),
                         test_metric=self._test_metric, metric_type=self._metric_type, test_weight=test_weight)

        if not self._silent:
            if self._have_feat_importance:
                feat_importance = self._model.feature_importance_
            else:
                feat_importance = None
            if feat_importance is not None:
                feat_importance = pd.DataFrame({'Features': self._features,
                                                'Importance': feat_importance})
                pd.set_option('max_columns', len(test_x.columns))
                pd.set_option('max_rows', len(test_x))
                print("Feature importances:")
                feat_importance.sort(columns='Importance', ascending=False, inplace=True)
                feat_importance.index = range(1, len(feat_importance) + 1)
                print(feat_importance)

        if output_file is not None:
            test_pred.to_csv(output_file, index=False)

    def get_feat_importance(self, fname=None):
        if (self._model is not None) & self._have_feat_importance:
            importances = self._model.feature_importances_
            std = np.std([tree.feature_importances_ for tree in self._model.estimators_], axis=0)
            indices = np.argsort(importances)

            fig = plt.figure()
            plt.title("Feature importance")
            plt.bar(range(len(self._features)), importances[indices], color="r", yerr=std[indices], align="center")
            plt.xticks(range(len(self._features)), np.asarray(self._features)[indices], rotation=90)
            fig.tight_layout()
            if fname is None:
                fig.savefig("feature-importance.png", format="png")
            else:
                fig.savefig(fname, format="jpg")
        else:
            print("The chosen algo does not process feature importance function")

    def feature_extraction(self, mode="train", input_test=None):
        feature_mode = self._feature_mode
        weather_data = load_pd_df(self._input_dir + '/weather.csv')
        weather_data = weather_data.loc[weather_data['Station'] == 1]
        spray_data = load_pd_df(self._input_dir + '/spray.csv')[["Date", "Latitude", "Longitude"]].drop_duplicates()
        spray_data = spray_data.drop_duplicates()

        if mode == 'train':
            train_data = load_pd_df(self._input_dir + '/train.csv')
            train_data = train_data.sort_values(["Date", "Latitude", "Longitude"], ascending=[1, 1, 1])
            test_data = load_pd_df(self._input_dir + '/test.csv')
            train_weather_data = train_data.merge(weather_data, on='Date', how='left')
            all_data = train_data.append(test_data)
            all_data_weather = all_data.merge(weather_data, on='Date', how='left')

            all_data_proc, feature_list = process_spray_data(all_data_weather, spray_data)
            train_data_proc, feature_list = process_spray_data(train_weather_data, spray_data)
            self._features = feature_list
            x = copy.deepcopy(train_data_proc)
            if feature_mode == 'one_hot':
                for col in x.columns:
                    if all_data_proc[col].dtype == np.dtype('object'):
                        mapping_dataframe = pd.get_dummies(all_data_proc[col].drop_duplicates(), dummy_na=True)
                        self._feature_mapping_dict.update({col: mapping_dataframe})
                        x[col] = pd.Series(x[col].map(lambda t: map(float, mapping_dataframe[t].values)), index=x.index)
                    else:
                        x[col] = x[col].fillna(self._na_fill_value)
                feature_x = transfer_list(x[x.columns.values[0]].values)
                for i in range(1, len(x.columns.values)):
                    if x.columns.values[i] in self._feature_mapping_dict.keys():
                        tmp = transfer_list(x[x.columns.values[i]].values)
                        feature_x = np.concatenate((feature_x, tmp))
                    else:
                        tmp = np.reshape(x[x.columns.values[i]].values, (1, len(x)))
                        feature_x = np.concatenate((feature_x, tmp))

                x = feature_x.transpose()
            elif feature_mode == 'label':
                mappings = dict()
                if mappings is not None:
                    for col in train_data_proc.columns:
                        if all_data_proc[col].dtype == np.dtype('object'):
                            s = np.unique(all_data_proc[col].fillna(self._na_fill_value).values)
                            mappings[col] = pd.Series([t[0] for t in enumerate(s)], index=s)
                            x[col] = x[col].map(mappings[col]).fillna(self._na_fill_value)
                        else:
                            x[col] = x[col].fillna(self._na_fill_value)
                self._feature_mapping_dict = mappings
            y = train_data[self._target_col].values
        elif mode == 'eval':
            input_data = load_pd_df(self._input_dir + '/train.csv')
            input_data = input_data.merge(weather_data, on='Date', how='left')
            input_data_proc, feature_list = process_spray_data(input_data, spray_data)
            x = copy.deepcopy(input_data_proc)

            if feature_mode == 'one_hot':
                for col in x.columns:
                    if input_data_proc[col].dtype == np.dtype('object'):
                        mapping_dataframe = self._feature_mapping_dict[col]
                        x[col] = pd.Series(x[col].map(lambda t: map(float, mapping_dataframe[t].values)), index=x.index)
                    else:
                        x[col] = x[col].fillna(self._na_fill_value)
                feature_x = transfer_list(x[x.columns.values[0]].values)
                for i in range(1, len(x.columns.values)):
                    if x.columns.values[i] in self._feature_mapping_dict.keys():
                        tmp = transfer_list(x[x.columns.values[i]].values)
                        feature_x = np.concatenate((feature_x, tmp))
                    else:
                        tmp = np.reshape(x[x.columns.values[i]].values, (1, len(x)))
                        feature_x = np.concatenate((feature_x, tmp))
                x = feature_x.transpose()
                y = input_data[self._target_col]

            elif feature_mode == 'label':
                for col in x.columns:
                    if x[col].dtype == np.dtype('object'):
                        x[col] = x[col].map(self._feature_mapping_dict[col]).fillna(self._na_fill_value)
                    else:
                        x[col] = x[col].fillna(self._na_fill_value)
                y = input_data[self._target_col]
        elif (mode == 'test') & (input_test is not None):
            input_test = input_test.merge(weather_data, on='Date', how='left')
            input_data_proc, feature_list = process_spray_data(input_test, spray_data)
            x = copy.deepcopy(input_data_proc)

            if feature_mode == 'one_hot':
                for col in x.columns:
                    if input_data_proc[col].dtype == np.dtype('object'):
                        mapping_dataframe = self._feature_mapping_dict[col]
                        x[col] = pd.Series(x[col].map(lambda t: map(float, mapping_dataframe[t].values)), index=x.index)
                    else:
                        x[col] = x[col].fillna(self._na_fill_value)
                feature_x = transfer_list(x[x.columns.values[0]].values)
                for i in range(1, len(x.columns.values)):
                    if x.columns.values[i] in self._feature_mapping_dict.keys():
                        tmp = transfer_list(x[x.columns.values[i]].values)
                        feature_x = np.concatenate((feature_x, tmp))
                    else:
                        tmp = np.reshape(x[x.columns.values[i]].values, (1, len(x)))
                        feature_x = np.concatenate((feature_x, tmp))
                x = feature_x.transpose()
                y = []

            elif feature_mode == 'label':
                for col in x.columns:
                    if x[col].dtype == np.dtype('object'):
                        x[col] = x[col].map(self._feature_mapping_dict[col]).fillna(self._na_fill_value)
                    else:
                        x[col] = x[col].fillna(self._na_fill_value)
                y = []
        else:
            sys.exit()

        return x, y, feature_list

    def transform_categorical_numerical(self, ind, xind, mode='train'):
        if mode == 'train':
            keys = np.unique(xind)
            values = range(len(keys))
            mappings_tmp = dict(zip(keys, values))
            self._feature_mapping_dict.update({ind: mappings_tmp})
            new_xind = np.asarray([float(mappings_tmp[key]) for key in xind])
        else:
            mappings_tmp = self._feature_mapping_dict[ind]
            new_xind = np.asarray([mappings_tmp[key] for key in xind])
        return new_xind

    def transform_categorical_onehot(self, input_data, all_data=None):
        if self._feature_transform_ is not None:
            x_new = self._feature_transform_.transform(input_data)
        else:
            onehot = preprocessing.OneHotEncoder()
            onehot.fit(all_data)
            x_new = onehot.transform(input_data)
            self._feature_transform_ = onehot
        return x_new

    def predict(self, x, date_indices=None):
        if (self._model_type == 'LSTM')|(self._model_type == 'CNN'):
            y_pred = self._model.predict_generator(generator=self.feature_generator(x, date_indices,self._feature_size[1]),steps=x.shape[0]/self._batch_size)
        else:
            y_pred = self._model.predict(x)
        return y_pred

    def evaluation(self):
        train_file_name = self._input_dir + '/train.csv'
        train_x, y_true, feature_list = self.feature_extraction(mode='eval')
        y_true = load_pd_df(train_file_name)[self._target_col]
        date_list = load_pd_df(train_file_name)['Date'].map(lambda t: datetime.datetime.strptime(t, '%Y-%m-%d'))
        date_indices, max_len = self.process_date_list(date_list)
        y_pred = self.predict(train_x,date_indices=date_indices)
        y_pred[np.where(y_pred<0.5)] = 0
        y_pred[np.where(y_pred>=0.5)] = 1
        cfm = confusion_matrix(y_true=y_true, y_pred=y_pred)

        print cfm
        return cfm

    def basic_evaluation(self):
        train_data = load_pd_df(self._input_dir + '/train.csv')
        pres_train_data = train_data.loc[train_data['WnvPresent'] == 1]

        mapdata = np.loadtxt("../west_nile/input/mapdata_copyright_openstreetmap_contributors.txt")

        aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
        lon_lat_box = (-88, -87.5, 41.6, 42.1)

        plt.figure(figsize=(10, 14))
        plt.imshow(mapdata,
                   cmap=plt.get_cmap('gray'),
                   extent=lon_lat_box,
                   aspect=aspect)

        pres_traps = pres_train_data[['Date', 'Trap', 'Longitude', 'Latitude', 'WnvPresent']]
        pres_locations = pres_traps[['Longitude', 'Latitude']].drop_duplicates().values

        all_traps = train_data[['Date', 'Trap', 'Longitude', 'Latitude', 'WnvPresent']]
        all_locations = all_traps[['Longitude', 'Latitude']].drop_duplicates().values

        plt.scatter(all_locations[:, 0], all_locations[:, 1], marker='o', color='blue')
        plt.scatter(pres_locations[:, 0], pres_locations[:, 1], marker='x', color='red')
        plt.savefig('heatmap.png')

    def spray_effectiveness_eval(self):
        train_data = load_pd_df(self._input_dir + '/train.csv')
        spray_data = load_pd_df(self._input_dir + '/spray.csv')

        spray_dates = spray_data['Date'].drop_duplicates().values
        spray_duration = 5

        for i in range(len(spray_dates)):
            spray_date = spray_dates[i]
            spray_effective_day_list = [
                datetime.datetime.strptime(spray_date, '%Y-%m-%d') + datetime.timedelta(days=j + 1)
                for j in range(spray_duration)]
            before_spray_day_list = [datetime.datetime.strptime(spray_date, '%Y-%m-%d') - datetime.timedelta(days=j + 1)
                                     for j in range(spray_duration)]

            mapdata = np.loadtxt("../west_nile/input/mapdata_copyright_openstreetmap_contributors.txt")

            aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
            lon_lat_box = (-88, -87.5, 41.6, 42.1)

            fig = plt.figure(figsize=(10, 28))
            before_spray = fig.add_subplot(121)
            after_spray = fig.add_subplot(122)
            before_spray.imshow(mapdata,
                                cmap=plt.get_cmap('gray'),
                                extent=lon_lat_box,
                                aspect=aspect)
            after_spray.imshow(mapdata,
                               cmap=plt.get_cmap('gray'),
                               extent=lon_lat_box,
                               aspect=aspect)

            spray_locations = spray_data[['Longitude', 'Latitude']].loc[
                spray_data['Date'] == spray_dates[i]].drop_duplicates().values
            before_spray.scatter(spray_locations[:, 0], spray_locations[:, 1], marker="D", color='green')
            after_spray.scatter(spray_locations[:, 0], spray_locations[:, 1], marker="D", color='green')

            for spray_effective_day in spray_effective_day_list:
                train_spray_data = train_data.loc[
                    train_data['Date'] == datetime.datetime.strftime(spray_effective_day, '%Y-%m-%d')]
                if len(train_spray_data) != 0:
                    pres_locations = train_spray_data[['Longitude', 'Latitude']].loc[
                        train_spray_data['WnvPresent'] == 1].drop_duplicates().values
                    all_locations = train_spray_data[['Longitude', 'Latitude']].drop_duplicates().values

                    after_spray.scatter(all_locations[:, 0], all_locations[:, 1], marker='o', color='blue')
                    after_spray.scatter(pres_locations[:, 0], pres_locations[:, 1], marker='x', color='red')

            for before_spray_day in before_spray_day_list:
                train_spray_data = train_data.loc[
                    train_data['Date'] == datetime.datetime.strftime(before_spray_day, '%Y-%m-%d')]
                if len(train_spray_data) != 0:
                    pres_locations = train_spray_data[['Longitude', 'Latitude']].loc[
                        train_spray_data['WnvPresent'] == 1].drop_duplicates().values
                    all_locations = train_spray_data[['Longitude', 'Latitude']].drop_duplicates().values

                    before_spray.scatter(all_locations[:, 0], all_locations[:, 1], marker='o', color='blue')
                    before_spray.scatter(pres_locations[:, 0], pres_locations[:, 1], marker='x', color='red')
            after_spray_fname = 'virus distribution after ' + spray_dates[i]
            after_spray.set_title(after_spray_fname)

            before_spray_fname = 'virus distribution after ' + spray_dates[i]
            before_spray.set_title(before_spray_fname)
            plt.savefig('spray_effectiveness/' + spray_dates[i] + '.png')

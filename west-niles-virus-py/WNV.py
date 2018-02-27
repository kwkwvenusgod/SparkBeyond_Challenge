import sklearn
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, \
    RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as sk_metrics
import custom_metrics
import pandas as pd
import timeit
import numpy as np
import ml_metrics
import cPickle as Pickle
import itertools
import os
import math
import random
from sklearn.pipeline import Pipeline
import inspect


def load_model_bin(model_file):
    model_file = open(model_file, 'rb')
    return Pickle.load(model_file)


def save_model_bin(model, model_file):
    model_file = open(model_file, 'wb')
    Pickle.dump(model, model_file, Pickle.HIGHEST_PROTOCOL)


def load_pd_df(file_name, del_old=False, bin_suffix='.bin.pkl'):
    ret_val = None
    bin_file_name = file_name + bin_suffix
    if os.path.isfile(bin_file_name):
        if not os.path.isfile(file_name) or os.path.getmtime(bin_file_name) > os.path.getmtime(file_name):
            ret_val = load_model_bin(model_file=bin_file_name)
            print "Loading %s cache file" % bin_file_name

    if ret_val is None:
        print "Loading %s raw file" % file_name
        ret_val = pd.read_csv(file_name)
        print "Saving %s cache file" % bin_file_name
        save_model_bin(model=ret_val, model_file=bin_file_name)
        if del_old:
            print "Erasing %s raw file" % file_name
            os.remove(file_name)
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


class WNV:
    def __init__(self, train_data_file, target_col,
                 model_type='GradientBoostingRegressor',
                 fit_args={"n_estimators": 10, "learning_rate": 0.001, "loss": "ls",
                           "max_features": 5, "max_depth": 7, "random_state": 788954,
                           "subsample": 1, "verbose": 50}, test_metric='normalized_weighted_gini', na_fill_value=-20000,
                 silent=False, skip_mapping=False, load_model=None, train_filter=None, metric_type='auto',
                 load_type='fit_more',
                 bootstrap=0, bootstrap_seed=None, weight_col=None):

        self._train_data_file = train_data_file
        self._target_col = target_col
        self._model_type = model_type
        self._fit_args = fit_args
        self._test_metric = test_metric
        self._na_fill_value = na_fill_value
        self._silent = silent
        self._skip_mapping = skip_mapping
        self._load_model = load_model
        self._bootstrap = bootstrap
        self._bootstrap_seed = bootstrap_seed
        self._weight_col = weight_col
        self._train_filter = train_filter
        self._metric_type = metric_type
        self._load_type = load_type
        self._features = None
        self._model = None
        self._staged_predict = None
        self._feat_importance_fun = None
        self._predict = None

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

    def train(self):
        start = timeit.default_timer()

        train_x = load_pd_df(self._train_data_file)

        len_train_before = len(train_x)
        train_x = data_filter(train_x, self._train_filter)
        if not self._silent:
            print "Train has %d instances (was %d before filtering)" % (len(train_x), len_train_before)

        mappings = None if self._skip_mapping else dict()

        train_y = train_x[self._target_col]
        del train_x[self._target_col]

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

        x_cols = train_x.columns.values
        self._features = x_cols
        self._feat_importance_fun = lambda (fitted_model): fitted_model.feature_importances_
        self._predict = lambda (fitted_model, pred_x): fitted_model.predict(pred_x)
        self._staged_predict = lambda (fitted_model, pred_x): [self._predict((fitted_model, pred_x))]

        model = None

        if self._model_type == "RandomForestRegressor":
            if model is None:
                model = RandomForestRegressor(self._fit_args)
                model.fit(X=train_x, y=train_y, **extra_fit_args)
                self._model = model
                self._predict = lambda (fitted_model, pred_x): self.continuous_predict(x=pred_x)

        elif self._model_type == "RandomForestClassifier":
            if model is None:
                model = RandomForestClassifier(self._fit_args)
                model.fit(X=train_x, y=train_y, **extra_fit_args)
                self._model = model
                self._predict = lambda (fitted_model, pred_x): self.pred_proba(x=pred_x)
            self._staged_predict = lambda (fitted_model, pred_x): [self._predict((fitted_model, pred_x))]

        elif self._model_type == "ExtraTreesRegressor":
            if model is None:
                model = ExtraTreesRegressor(self._fit_args)
                model.fit(X=train_x, y=train_y, **extra_fit_args)
                self._model = model
                self._predict = lambda (fitted_model, pred_x): self.continuous_predict(x=pred_x)

        elif self._model_type == "ExtraTreesClassifier":
            if model is None:
                model = ExtraTreesClassifier(self._fit_args)
                model.fit(X=train_x, y=train_y, **extra_fit_args)
            self._predict = lambda (fitted_model, pred_x): self.pred_proba(x=pred_x)
            self._staged_predict = lambda (fitted_model, pred_x): [self._predict((fitted_model, pred_x))]

        elif self._model_type == "GradientBoostingRegressor":
            if model is None:
                model = GradientBoostingRegressor(self._fit_args)
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
                model = GradientBoostingClassifier(self._fit_args)
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
                model = LogisticRegression(self._fit_args)
                model.fit(X=train_x, y=train_y)
                self._model = model
            self._predict = lambda (fitted_model, pred_x): self.pred_proba(x=pred_x)
            self._staged_predict = lambda (fitted_model, pred_x): [self._predict((fitted_model, pred_x))]
            self._feat_importance_fun = lambda (fitted_model): None

        elif self._model_type == "SVC":
            if model is None:
                model = sklearn.svm.SVC(self._fit_args)
                model.fit(X=train_x, y=train_y)
                self._model = model
            self._predict = lambda (fitted_model, pred_x): self.pred_proba(x=pred_x)
            self._staged_predict = lambda (fitted_model, pred_x): [self._predict((fitted_model, pred_x))]
            self._feat_importance_fun = lambda (fitted_model): None

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
            self._feat_importance_fun = lambda (fitted_model): None

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
            feat_importance = self._feat_importance_fun(self._model)
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

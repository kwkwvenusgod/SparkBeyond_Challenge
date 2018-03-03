import sklearn
import datetime
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, \
    RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
import custom_metrics
import pandas as pd
import timeit
import numpy as np
import ml_metrics
import cPickle as Pickle
import itertools
import math
import random
from sklearn.pipeline import Pipeline
from collections import Counter
import inspect


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

class WNV:
    def __init__(self, input_dir ,train_data_file, target_col,
                 model_type='GradientBoostingRegressor',
                 fit_args={"n_estimators": 10, "learning_rate": 0.001, "loss": "ls",
                           "max_features": 5, "max_depth": 7, "random_state": 788954,
                           "subsample": 1, "verbose": 50}, test_metric='normalized_weighted_gini', na_fill_value=-20000,
                 silent=False, skip_mapping=False, load_model=None, train_filter=None, metric_type='auto',
                 load_type='fit_more',
                 bootstrap=0, bootstrap_seed=None, weight_col=None, del_cols=[]):
        self._input_dir = input_dir
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
        self._have_feat_importance = False
        self._predict = None
        self._del_cols = del_cols
        self._data_balance = False
        self._feature_mapping_dict = {}

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

        train_x,train_y,feature_list = self.feature_extraction()
        self._features = feature_list
        if not self._silent:
            print "Train has %d instances " % (len(train_x))

        counts = Counter(train_y)
        expectation_ratio = 1/float(len(counts.keys()))
        n_samples = len(train_y)
        for key,value in counts.items():
            tmp = float(expectation_ratio)/(float(value)/float(n_samples))
            if (tmp>6)|(tmp<(1.0/6.0)):
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

        self._predict = lambda (fitted_model, pred_x): fitted_model.predict(pred_x)
        self._staged_predict = lambda (fitted_model, pred_x): [self._predict((fitted_model, pred_x))]

        model = None

        if self._model_type == "RandomForestRegressor":
            if model is None:
                if self._data_balance is True:
                    self._fit_args.update({"class_weight":"balanced"})
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
                model = sklearn.svm.SVC(self._fit_args)
                model.fit(X=train_x, y=train_y)
                self._model = model
            self._predict = lambda (fitted_model, pred_x): self.pred_proba(x=pred_x)
            self._staged_predict = lambda (fitted_model, pred_x): [self._predict((fitted_model, pred_x))]

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

    def feature_extraction(self, mode="train"):
        weather_data = load_pd_df(self._input_dir+'/weather.csv')
        spray_data = load_pd_df(self._input_dir+ '/spray.csv')

        del spray_data["Time"]

        spray_data = spray_data.drop_duplicates()

        if mode=="test":
            train_data = load_pd_df(self._input_dir+'/test.csv')
        else:
            train_data = load_pd_df(self._input_dir+'/train.csv')

        data = []
        feature_list = ["Date", "Species","Longitude","Latitude","Trap", "Address", "Street"]

        #=================== Parse Train File =========================
        date = []
        for tmp in train_data["Date"]:
            tmp = tmp.split('-')
            tmp = tmp[1]+'-'+tmp[2]
            date.append(tmp)
        data.append(date)

        species = train_data["Species"].tolist()
        data.append(species)

        longitude = train_data["Longitude"].tolist()
        data.append(longitude)

        latitude = train_data["Latitude"].tolist()
        data.append(latitude)

        trap = train_data["Trap"].tolist()
        data.append(trap)

        addr = train_data["Address"].tolist()
        data.append(addr)

        street = train_data["Street"].tolist()
        data.append(street)

        data = np.asarray(data)

        # #=================== Parse weather data ===============
        weather_data_feature = ['Tmax','Tmin','DewPoint',"WetBulb","Heat","Cool","Sunrise","Sunset","StnPressure",
                                    "SeaLevel","ResultSpeed","ResultDir","AvgSpeed"]
        feature_list.extend(weather_data_feature)
        w_selection = list(weather_data[weather_data_feature].groupby(weather_data["Date"]))
        w_record_date = weather_data["Date"].drop_duplicates().tolist()
        weather_feature = []
        for i in range(len(train_data["Date"])):
            indices = w_record_date.index(train_data["Date"][i])
            w_selection_tmp = w_selection[indices][1].get_values()[0]
            weather_feature.append(w_selection_tmp)

        weather_feature = np.asarray(weather_feature).transpose()

        data=np.concatenate([data,weather_feature])

        spray_indicator = np.zeros((1,len(date)))
        thres_date = 5
        thres_area = 0.6
        train_data_dates = train_data["Date"].drop_duplicates().tolist()
        spray_la_long_list = list(spray_data[["Latitude","Longitude"]].groupby(spray_data["Date"]))
        train_data_la_long_list = list(train_data[["Latitude","Longitude"]].groupby(train_data["Date"]))
        for spray_la_long_date in spray_la_long_list:
            spray_date_tmp_start = datetime.datetime.strptime(spray_la_long_date[0], '%Y-%m-%d')
            spray_date_tmp_end = spray_date_tmp_start + datetime.timedelta(days=thres_date)
            spray_location_tmp = spray_la_long_date[1].get_values()

            for i in range(len(train_data_dates)):
                train_data_date = datetime.datetime.strptime(train_data_dates[i], '%Y-%m-%d')
                if (train_data_date>spray_date_tmp_start)&(train_data_date<=spray_date_tmp_end):
                    train_location= train_data_la_long_list[i][1].get_values()
                    train_location_indices = train_data_la_long_list[i][1].axes[0].get_values()
                    distances = euclidean_distances(spray_location_tmp,train_location)
                    indices = np.unique(np.where(distances<thres_area)[1], return_index=False).astype(int)
                    indices = train_location_indices[indices].astype(int)
                    spray_indicator[0,indices] = 1
        data = np.concatenate([data, spray_indicator])
        feature_list.extend(["Spray"])

        data_feature = np.zeros(data.shape)
        for i in range(data.shape[0]):
            if (data[i].dtype != np.dtype('float'))|(data[i].dtype != np.dtype('int'))|(data[i].dtype != np.dtype('double')):
                data_feature[i] = self.transform_categorical_numerical(i, data[i],mode=mode)
            else:
                data_feature[i] = data[i]

        return data_feature.transpose(), list(train_data[self._target_col]),feature_list

    def transform_categorical_numerical(self, ind, xind, mode="train"):

        if mode == "train":
            keys = np.unique(xind)
            values = range(len(keys))
            mappings_tmp = dict(zip(keys,values))
            self._feature_mapping_dict.update({ind: mappings_tmp})
            new_xind = np.asarray([float(mappings_tmp[key]) for key in xind])
        else:
            mappings_tmp = self._feature_mapping_dict[ind]
            new_xind = np.asarray([mappings_tmp[key] for key in xind])
        return new_xind

    def evaluation(self):
        train_x,y_true, feature_list = self.feature_extraction()
        y_pred = self._model.predict(train_x)
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

        pres_traps = pres_train_data[['Date', 'Trap','Longitude', 'Latitude', 'WnvPresent']]
        pres_locations = pres_traps[['Longitude', 'Latitude']].drop_duplicates().values

        all_traps = train_data[['Date', 'Trap','Longitude', 'Latitude', 'WnvPresent']]
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
            spray_effective_day_list = [datetime.datetime.strptime(spray_date,'%Y-%m-%d') + datetime.timedelta(days=j)
                                         for j in range(spray_duration)]

            mapdata = np.loadtxt("../west_nile/input/mapdata_copyright_openstreetmap_contributors.txt")

            aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
            lon_lat_box = (-88, -87.5, 41.6, 42.1)

            plt.figure(figsize=(10, 14))
            plt.imshow(mapdata,
                       cmap=plt.get_cmap('gray'),
                       extent=lon_lat_box,
                       aspect=aspect)

            spray_locations = spray_data[['Longitude', 'Latitude']].loc[spray_data['Date'] == spray_dates[i]].drop_duplicates().values
            plt.scatter(spray_locations[:, 0], spray_locations[:, 1], marker="D", color='green')

            for spray_effective_day in spray_effective_day_list:
                train_spray_data = train_data.loc[train_data['Date'] == datetime.datetime.strftime(spray_effective_day,'%Y-%m-%d')]
                if len(train_spray_data) != 0:
                    pres_locations = train_spray_data[['Longitude', 'Latitude']].loc[train_spray_data['WnvPresent'] == 1].drop_duplicates().values
                    all_locations = train_spray_data[['Longitude', 'Latitude']].drop_duplicates().values

                    plt.scatter(all_locations[:, 0], all_locations[:, 1], marker='o', color='blue')
                    plt.scatter(pres_locations[:, 0], pres_locations[:, 1], marker='x', color='red')
            plt.title(spray_dates[i])
            plt.savefig('spray_effectiveness/'+spray_dates[i]+'.png')








            









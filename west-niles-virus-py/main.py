import simplejson
import pandas as pd
import utils
from WNV import WNV

if __name__ == "__main__":
    config_file_path = 'args_config.json'
    with open(config_file_path, 'rb') as config_file:
        config = simplejson.load(config_file)

    wnv = WNV(input_dir=config["input_dir"],
              train_data_file=config['train_file_path'],
              test_metric=config['test_metric'],
              target_col=config['target_col'],
              weight_col=config['weight_col'],
              metric_type=config['metric_type'],
              model_type=config['model_type'],
              na_fill_value=config['na_fill_value'],
              skip_mapping=config['skip_mapping'],
              fit_args=config['fit_args'],
              silent=config['silent'],
              train_filter=config['train_filter'],
              load_model=config['load_model'],
              load_type=config['load_type'],
              bootstrap=config['bootstrap'],
              bootstrap_seed=config['bootstrap_seed'],
              del_cols=config['del_cols'])

    # wnv.basic_evaluation()
    # wnv.spray_effectiveness_eval()
    wnv.train()
    wnv.evaluation()
    wnv.get_feat_importance()

    testx,testy,featurelist = wnv.feature_extraction(mode='test')
    y_test_pred = wnv.predict(testx)

    test_origin = pd.read_csv('../west_nile/input/test.csv')
    test_origin['WnvPresent'] = pd.Series(y_test_pred, index=test_origin.index)

    utils.draw_affected_area(test_origin,'predict_distribution')
    utils.draw_affected_area(test_origin,'predict_distribution', mode='year')
    utils.draw_affected_area(test_origin,'predict_distribution', mode='year_month')





    #draw result



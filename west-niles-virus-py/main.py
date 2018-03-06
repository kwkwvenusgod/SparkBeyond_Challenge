import simplejson
import pandas as pd
import utils, datetime
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
              feature_mode=config['feature_mode'])

    # wnv.basic_evaluation()
    # wnv.spray_effectiveness_eval()
    wnv.train()
    wnv.evaluation()
    wnv.get_feat_importance()

    test_origin = pd.read_csv('../west_nile/input/test.csv')
    group_features = ["Date","Address","Species","Block","Street","Trap",
                      "AddressNumberAndStreet","Latitude","Longitude","AddressAccuracy"]
    numMosqs = test_origin.groupby(group_features)['Id'].transform("count")
    test_origin['NumMosquitos'] = numMosqs
    testx,testy,featurelist = wnv.feature_extraction(mode='test', input_test=test_origin)
    test_date_list = test_origin['Date'].map(lambda t:datetime.datetime.strptime(t,'%Y-%m-%d'))
    date_indices = wnv.process_date_list(test_date_list)
    y_test_pred = wnv.predict(testx, date_indices=date_indices)

    test_origin['WnvPresent'] = pd.Series(y_test_pred, index=test_origin.index)

    test_origin[['Id','WnvPresent']].to_csv('sampleSubmission.csv',index=False)

    utils.draw_affected_area(test_origin,'predict_distribution')
    utils.draw_affected_area(test_origin,'predict_distribution', mode='year')
    utils.draw_affected_area(test_origin,'predict_distribution', mode='year_month')





    #draw result



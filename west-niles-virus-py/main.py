import simplejson
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

    wnv.basic_evaluation()
    wnv.spray_effectiveness_eval()
    wnv.train()
    wnv.evaluation()
    wnv.get_feat_importance()


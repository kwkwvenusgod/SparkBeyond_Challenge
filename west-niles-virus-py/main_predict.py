import pickle
from CNN_LSTM import CNN_LSTM
from WNV import WNV


file_path = ''
with open(file_path, 'rb') as wnv_param_file:
    wnv_param = pickle.load(wnv_param_file)

wnv = WNV(**wnv_param)
real_model_path = ''
wnv.load_real_model(real_model_path)


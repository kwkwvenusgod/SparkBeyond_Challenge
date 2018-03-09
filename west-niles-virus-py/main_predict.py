import pickle
from CNN_LSTM import CNN_LSTM
from WNV import WNV
import simplejson


file_path = './cnn_lstm/wvn.pickle'
with open(file_path, 'rb') as wnv_param_file:
    wnv_param = pickle.load(wnv_param_file)

wnv = WNV(**wnv_param)
real_model_path = './cnn_lstm/LSTM.h5'
wnv.load_real_model(real_model_path)

wnv.evaluation()


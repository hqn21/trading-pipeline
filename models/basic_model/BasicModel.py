import torch
import torch.nn as nn
from models.basic_model import iTransformer, DLinear, LSTM, CATS, TimesNet


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
        
class BasicModel(nn.Module):
    def __init__(self, configs):
        super(BasicModel, self).__init__()
        
        if not hasattr(configs, 'n_features') and hasattr(configs, 'feature_dim'):
            configs.n_features = configs.feature_dim
    
        self.configs = configs
        model_dict = {
            'iTransformer': iTransformer.Model,
            'DLinear': DLinear.Model,
            'LSTM': LSTM.Model,
            'TimesNet': TimesNet.Model,
            'CATS': CATS.CATS
        }
        self.model_class = model_dict[configs.model]

        self.decomp_method = configs.decomp_method
        self.pred_len = configs.pred_len
        
        if self.decomp_method == 'moving_avg':
            kernel_size = configs.decomp_kernel_size
            self.decomposition = series_decomp(kernel_size)
            
            self.Model_Seasonal = self._build_model()
            self.Model_Trend = self._build_model()
            
        elif self.decomp_method == "":
            self.Model = self._build_model()
            
        else:
            raise Exception(f"decomp_method: {self.decomp_method} does not exist")
            
    def _build_model(self):
        model = self.model_class(self.configs).float()
        return model
    
    def forward(self, x, x_broker=None, x_general=None, stock_idx=None):
        B, N, L, F = x.shape
        x = x.reshape(B*N, L, F)
        
        x_mark_enc = None
        x_dec = None
        x_mark_dec = None
        batch_static = None
            
        if self.decomp_method == 'moving_avg':
            seasonal_init, trend_init = self.decomposition(x)
            
            seasonal_output = self.Model_Seasonal(seasonal_init, x_mark_enc, x_dec, x_mark_dec, batch_static)
            trend_output = self.Model_Trend(trend_init, x_mark_enc, x_dec, x_mark_dec, batch_static)
            
            output = seasonal_output + trend_output
        else:
            output = self.Model(x, x_mark_enc, x_dec, x_mark_dec, batch_static)
            
        return output.reshape(B, N)
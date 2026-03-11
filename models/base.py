import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, model, preprocesser_model: nn.Module=None):
        super(BaseModel, self).__init__()
        self.model = model
        
        if preprocesser_model:
            self.preprocesser_model = preprocesser_model
            
            for param in self.preprocesser_model.parameters():
                param.requires_grad = False
            # （可選）切到 eval 模式以關閉 dropout / batchnorm 更新
            self.preprocesser_model.eval()
            
            self.preprocessor_linear = torch.nn.Linear(
                self.preprocesser_model.feature_dim * 2, 
                self.preprocesser_model.feature_dim
            )
        else:
            self.preprocesser_model = None
    
    def preprocess_and_combine(self, x):
        preprocessed_x = self.preprocesser_model(x)
        x = torch.cat((x, preprocessed_x), dim=-1)
        x = self.preprocessor_linear(x)

        return x
    
    def forward(self, x, x_broker=None, x_general=None, stock_idx=None):
        if self.preprocesser_model: # B * n_stock * n_feature * seq_len
            x = self.preprocess_and_combine(x)
    
        if stock_idx:    
            output = self.model(x, x_broker, x_general, stock_idx)
        else:
            output = self.model(x, x_broker, x_general)
            
        return output
            
        

    
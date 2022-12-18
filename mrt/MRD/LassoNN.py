import torch
from torch import nn


class CancelOut(nn.Module):
    '''
    CancelOut Layer
    
    x - an input data (vector, matrix, tensor)
    '''
    def __init__(self,inp, *kargs, **kwargs):
        super(CancelOut, self).__init__()
        self.weights = nn.Parameter(torch.zeros(inp,requires_grad = True)+4)
    def forward(self, x):
        return (x * torch.sigmoid(self.weights.float()))

class LassoNN(nn.Module):
    def __init__(self, features, is_linear=True):
        super(LassoNN, self).__init__()
        if is_linear:
            self.is_linear = True
            self.fc_layer = nn.Sequential(
                # nn.Dropout(p=0.1),
                nn.Linear(features, 1, bias=False),
            )
        else:
          self.is_linear = False
          self.cancelout = CancelOut(features)
          self.fc_layer = nn.Sequential(
              nn.Linear(features, 16,bias=True),
              nn.ReLU(),
              nn.Dropout(p=0.5),
              nn.Linear(16, 1,bias=False)
          )

    def predict(self, x):
        return self.forward(torch.tensor(x).type(torch.FloatTensor)).clone().detach().numpy()

    def forward(self, x):
        if not self.is_linear:
          x = self.cancelout(x)
        # fc layer
        x = self.fc_layer(x)
        return x

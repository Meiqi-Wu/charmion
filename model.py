import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)

class DenseBlock(nn.Module):
    '''
    A block of a fully-connected layer with relu activation and dropout.
    '''
    def __init__(self, input_size, output_size, dropout_ratio):
        super(DenseBlock, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout_ratio)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        return F.relu(x)
    
class DenseNet(nn.Module):
    '''
    A few fully-connected layers, with sigmoid activation at the output layer.
    '''
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(DenseNet, self).__init__()
        if isinstance(dropout, list):
            self.dropout = dropout
        else:
            self.dropout = [dropout] * len(hidden_size)
            
        self.layers = nn.ModuleList()
        layer_size = [input_size] + hidden_size + [output_size]
        for i in range(len(layer_size)-2):
            self.layers.append(DenseBlock(layer_size[i], layer_size[i+1], self.dropout[i]))
        self.layers.append(nn.Linear(layer_size[-2], layer_size[-1]))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.sigmoid(x)
    
    def regular_loss(self, L1, L2):
        l1, l2 = 0, 0
        for layer in self.layers[:-1]:
            w = layer.linear.weight.data
            l1 += torch.sum(w.abs())
            l2 += torch.sum(torch.mul(w,w))
            
        w = self.layers[-1].weight.data
        l1 += torch.sum(w.abs())
        l2 += torch.sum(torch.mul(w,w))
        return L1*l1 + L2*l2
       
        
class Model(object):
    '''
    The Model class.
    '''
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        

    def fit(self, X, y, epoch, lr, batch_size, L2, pos_weight, verbose=True):
        
        n_samples = len(X)
        if type(X)!=np.ndarray:
            X = X.values
            y = y.values
            
        optimizer = optim.Adam(self.net.parameters(), lr = lr, weight_decay=L2)    
        criterian = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(pos_weight))
        
        for e in range(epoch):
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:min(i+batch_size, n_samples)]
                y_target = y[i:min(i+batch_size, n_samples)]
                X_batch = torch.from_numpy(X_batch).float()
                y_target = torch.from_numpy(y_target).float()
                
                optimizer.zero_grad()
                y_prob = self.net(X_batch)
                
                y_prob = y_prob.view(-1,1)
                y_target = y_target.view(-1,1)
            
                loss = criterian(y_prob, y_target)
                
                loss.backward()
                
#                 gradient = self.net.layers[0].linear.weight.grad
#                 print(torch.min(gradient), torch.max(gradient))
                
#                 if (gradient != gradient).any():
#                     print('gradient')
#                     print(gradient)
                        
                optimizer.step()
                
                if verbose:
                    if i%(30*batch_size)==0:
                        print(f"Epoch [{e+1}, {i}] : loss {loss}")       
    
    def predict(self, X, thresh = 0.5):
        prob = self.predict_prob(X)
        res = (prob>=thresh)*1
        return res
    
    def predict_prob(self, X):
        if type(X)==pd.core.frame.DataFrame:
            X = torch.from_numpy(X.values).float()
        elif type(X)== np.ndarray:
            X = torch.from_numpy(X).float()
        
        output = self.net(X)
        return output.detach().numpy()


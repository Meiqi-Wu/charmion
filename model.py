import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from callbacks import EarlyStopping
from utils import make_dataloader
from sklearn.model_selection import train_test_split

SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)

class DenseBlock(nn.Module):
    '''
    A block of a fully-connected layer with relu activation and dropout.
    '''
    def __init__(self, input_size, output_size, dropout_ratio, activate):
        super(DenseBlock, self).__init__()
        self.batchnorm = nn.BatchNorm1d(input_size)
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout_ratio)
        self.activate = activate
        
    def forward(self, x):
        x = self.batchnorm(x)
        x = self.linear(x)
        x = self.dropout(x)
        if self.activate:
            return self.activate(x)
        else:
            return x
    
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
            self.layers.append(DenseBlock(layer_size[i], layer_size[i+1], self.dropout[i], F.relu))
        self.layers.append(DenseBlock(layer_size[-2], layer_size[-1], 0, None))

    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def regular_loss(self, L1, L2):
        all_params = torch.tensor([])
        for layer in self.layers:
            layer_params = list(layer.linear.parameters())[0].view(-1)
            all_params = torch.cat([all_params, layer_params])

        return L1*torch.norm(all_params, 1) + L2*torch.norm(all_params, 2)
    
        
        
class Model(object):
    '''
    The Model class.
    '''
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.net.to(self.device)
       

    def fit(self, X, y, epoch, lr, batch_size, L1, L2, patience=5, pos_weight=1, verbose=True):
        if type(X)!=np.ndarray:
            X = X.values
            y = y.values
        # split train and validation set
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        X_val = torch.from_numpy(X_val).float().to(self.device)
        y_val = torch.from_numpy(y_val).float().to(self.device)
        train_loader = make_dataloader(X_train, y_train, batch_size)
        
        optimizer = optim.Adam(self.net.parameters(), lr = lr)    
        criterian = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        n_features = X_train.shape[1]
        for e in range(epoch):
            counter = 0
            for data in train_loader:
                counter += 1
                X_batch = data[:,:n_features].float().to(self.device)
                y_target = data[:,n_features:].float().to(self.device)

                optimizer.zero_grad()
                
                y_prob = self.net(X_batch)
                loss = criterian(y_prob.view(-1,1), y_target.view(-1,1)) + self.net.regular_loss(L1, L2)
                loss.backward()
                optimizer.step()
    
                if verbose:
                    if counter % 40 ==0:
                        print(f"Epoch [{e+1}, {counter}] : train loss {loss}") 
                        
            # validation loss
            y_prob = self.net(X_val)
            valid_loss = criterian(y_prob.view(-1,1), y_val.view(-1,1)) + self.net.regular_loss(L1, L2)
            
            early_stopping(valid_loss, self.net)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        self.net.load_state_dict(torch.load('checkpoint.pt'))
    
    def predict(self, X, thresh = 0.5):
        prob = self.predict_prob(X)
        res = (prob>=thresh)*1
        return res
    
    def predict_proba(self, X):
        if type(X)==pd.core.frame.DataFrame:
            X = torch.from_numpy(X.values).float()
        elif type(X)== np.ndarray:
            X = torch.from_numpy(X).float()
        
        output = F.sigmoid(self.net(X))
        return output.detach().numpy().astype('float64')


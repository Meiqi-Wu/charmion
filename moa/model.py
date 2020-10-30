import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from moa.callbacks import EarlyStopping
from moa.utils import make_dataloader
from moa.metrics import SmoothCrossEntropyLoss
from sklearn.model_selection import train_test_split



class DenseBlock(nn.Module):
    '''
    A block of a fully-connected layer with relu activation and dropout.
    '''
    def __init__(self, input_size, output_size, dropout_ratio, activate, batch=True):
        super(DenseBlock, self).__init__()
        self.linear = nn.utils.weight_norm(nn.Linear(input_size, output_size))
        self.activate = activate
        if batch:
            self.batchnorm = nn.BatchNorm1d(output_size)
        else:
            self.batchnorm = None
        
        self.dropout = nn.Dropout(dropout_ratio)
        
        
    def forward(self, x):
        x = self.linear(x)
        
        if self.activate:
            x = self.activate(x)
            
        if self.batchnorm:
            x = self.batchnorm(x)
        
        x = self.dropout(x)
        return x
    
class DenseNet(nn.Module):
    '''
    A few fully-connected layers, with sigmoid activation at the output layer.
    '''
    def __init__(self, input_size=1200, hidden_size=[2048, 2048], output_size=206, dropout=0.2):
        super(DenseNet, self).__init__()
        if isinstance(dropout, list):
            self.dropout = dropout
        else:
            self.dropout = [dropout] * (1+len(hidden_size))
            
        self.layers = nn.ModuleList()
        layer_size = [input_size] + hidden_size + [output_size]
        
        self.layers.append(nn.BatchNorm1d(input_size))
        self.layers.append(nn.Dropout(self.dropout[0]))
        
        for i in range(len(layer_size)-2):
            self.layers.append(DenseBlock(layer_size[i], layer_size[i+1], self.dropout[i+1], F.relu))
        self.layers.append(DenseBlock(layer_size[-2], layer_size[-1], 0, None, False))
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def regular_loss(self, L1, L2):
        all_params = torch.tensor([]).to(self.device)
        for layer in self.layers[2:]:
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
       

    def fit(self, X_train, y_train, X_val, y_val, epoch, lr, batch_size, L1=0, L2=0, weight_decay=1e-5, patience=10, smoothing=0.001, p_min=0.001, scheduler='ReduceLROnPlateau', verbose=True):
        if type(X_train)!=np.ndarray :
            X_train, y_train = X_train.values, y_train.values
            X_val, y_val = X_val.values, y_val.values

        X_val = torch.from_numpy(X_val).float().to(self.device)
        y_val = torch.from_numpy(y_val).float().to(self.device)
        train_loader = make_dataloader(X_train, y_train, batch_size)
        
        optimizer = optim.Adam(self.net.parameters(), lr = lr, weight_decay=weight_decay)    
        if scheduler in ['OneCycleLR', 'both']:
            lr_scheduler_1 = optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.1, div_factor=1e3, max_lr=1e-2, 
                                                         epochs=epoch, steps_per_epoch=len(train_loader))
        if scheduler in ['ReduceLROnPlateau', 'both']:
            lr_scheduler_2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(patience-7, 2), 
                                                                factor=0.5, verbose=verbose)
#         criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.))
        criterion = SmoothCrossEntropyLoss(smoothing=smoothing)
        metric = lambda inputs, targets: F.binary_cross_entropy((torch.clamp(F.sigmoid(inputs), p_min, 1-p_min)), targets)
        
        early_stopping = EarlyStopping(patience=patience, verbose=False)
        
        n_features = X_train.shape[1]
        for e in range(epoch):
            counter = 0
            for data in train_loader:
                X_batch = data[:,:n_features].float().to(self.device)
                y_target = data[:,n_features:].float().to(self.device)
                if X_batch.shape[0] <= 1:
                    continue
                self.net.train()
                optimizer.zero_grad()
                
                y_prob = self.net(X_batch)
                train_loss = criterion(y_prob, y_target) + self.net.regular_loss(L1, L2)
                train_metric = metric(y_prob, y_target)
                train_loss.backward()
                optimizer.step()
                if scheduler in ['OneCycleLR', 'both']:
                    lr_scheduler_1.step()
                        
            # validation loss
            self.net.eval()
            y_prob = self.net(X_val)
            valid_loss = criterion(y_prob, y_val) + self.net.regular_loss(L1, L2)
            valid_metric = metric(y_prob, y_val)
            
            if verbose:
                if (e+1) % 2 ==0:
                    print("Epoch [{}/{}] : train loss {:5f}, train metric {:5f}, val loss {:5f}, val metric {:5f}".format(e+1, epoch, train_loss, train_metric, valid_loss, valid_metric)) 
            
            early_stopping(valid_metric, self.net)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            if scheduler in ['ReduceLROnPlateau', 'both']:
                lr_scheduler.step(valid_metric)
                
        self.net.load_state_dict(torch.load('checkpoint.pt'))

    
    def predict(self, X, thresh = 0.5):
        prob = self.predict_prob(X)
        res = (prob>=thresh)*1
        return res
    
    def predict_proba(self, X):
        if type(X)==pd.core.frame.DataFrame:
            X = torch.from_numpy(X.values).float().to(self.device)
        elif type(X)== np.ndarray:
            X = torch.from_numpy(X).float().to(self.device)
        
        output = F.sigmoid(self.net(X))
        return output.cpu().detach().numpy().astype('float64')
    
    def save_model(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load_model(self, path):
        self.net.load_state_dict(torch.load(path))
        


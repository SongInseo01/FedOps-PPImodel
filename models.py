from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import bayes_layers as bl
from copy import deepcopy
from sklearn.metrics import roc_curve
import math
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm 

random_seed = 72
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device, torch.cuda.is_available())


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


# Define MNIST Model    
DROP_OUT = 0.45
N_MC_ITER = 20

class PPIBayesNet(nn.Module):
    def __init__(self, output_size, **bayes_args):
        super().__init__()
        self.model = nn.Sequential (
            bl.Linear(30, 128, prior_mu=0, prior_logsigmasq=0,
                      bias=0.1, approx_post='Radial', kl_method='repar',
                      n_mc_iter=N_MC_ITER, **bayes_args),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),

            bl.Linear(128, 64, prior_mu=0, prior_logsigmasq=0,
                      bias=False, approx_post='Radial', kl_method='repar',
                      n_mc_iter=N_MC_ITER, **bayes_args),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),

            bl.Linear(64, 32, prior_mu=0, prior_logsigmasq=0,
                      bias=False, approx_post='Radial', kl_method='repar',
                      n_mc_iter=N_MC_ITER, **bayes_args),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),

            bl.Linear(32, 16, prior_mu=0, prior_logsigmasq=0,
                      bias=False, approx_post='Radial', kl_method='repar',
                      n_mc_iter=N_MC_ITER, **bayes_args),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),

            bl.Linear(16, output_size, **bayes_args)
        )

    def forward(self, x):
        kl = 0
        for layer in self.model:
            # 넘파이 배열을 파이토치 텐서로 변환
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32).to(device)
            
            # 베이지안 선형 레이어인 경우
            if isinstance(layer, bl.Linear):
                x, kl_ = layer(x)
                kl += kl_
            else:
                # 다른 레이어의 경우 (활성화, 정규화, 드롭아웃)
                x = layer(x)
        return x, kl

# Set the loss function and optimizer
# def set_model_hyperparameter(model, lr):
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=2, eta_max=0.01, T_up=10, gamma=0.5)

#     return criterion, optimizer, scheduler

# Set the torch train & test
# torch train
def train_torch():
    def custom_train_torch(model, train_loader, val_loader, epochs, device: str = "cpu"):

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=2, eta_max=0.01, T_up=10, gamma=0.5)
        """Train the network on the training set."""
        print("Starting training...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        train_losses, valid_losses, lowest_loss = list(), list(), np.inf

        
        for epoch in range(epochs):
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
                train_loss, valid_loss = 0, 0

                model.trian()
                for x_minibatch, y_minibatch in train_loader:
                    x_minibatch = x_minibatch.to(device)
                    y_minibatch = y_minibatch.view(-1, 1).to(device) # 타겟 레이블 형태 조정

                    outputs = model(x_minibatch)
                    y_minibatch_pred = outputs[0]
                    kl_div = outputs[1]

                    kl_weight = 0.1
                    loss = criterion(y_minibatch_pred, y_minibatch) + kl_weight * kl_div

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                scheduler.step()
                    
                train_losses.append(train_loss / len(train_loader))

                model.eval()
                with torch.no_grad():
                    for x_minibatch, y_minibatch in val_loader:
                        x_minibatch = x_minibatch.to(device)
                        y_minibatch = y_minibatch.view(-1, 1).to(device)

                        outputs = model(x_minibatch)
                        y_minibatch_pred = outputs[0]
                        kl_div = outputs[1]

                        loss = criterion(y_minibatch_pred, y_minibatch)
                        valid_loss += loss.item()
                
                valid_loss = valid_loss / len(val_loader)
                valid_losses.append(valid_loss)

                if valid_losses[-1] < lowest_loss:
                    lowest_loss = valid_losses[-1]
                    lowest_epoch = epoch
                    best_model = deepcopy(model.state_dict())
                # else:
                #     if (early_stop > 0) and lowest_epoch + early_stop < epoch:
                #         print("Early Stopped", epoch, "epochs")
                #         break

                current_lr = optimizer.param_groups[0]['lr']

                # print(f'Epoch {epoch + 1}/{epochs} | Train loss:{train_losses[-1]:.4f} | Valid loss:{valid_losses[-1]:.4f} | Lowest loss:{lowest_loss:.4f} | lowest_epoch:{lowest_epoch} | LR: {current_lr:.6f}')
                pbar.update()

            model.load_state_dict(best_model)
        model.to("cpu")
            
        return model
    
    return custom_train_torch

# torch test
def test_torch():
    
    def custom_test_torch(model, test_loader, device: str = "cpu"):
        criterion = nn.BCEWithLogitsLoss()
        """Validate the network on the entire test set."""
        print("Starting evalutation...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        total_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_targets = []    
        
        model.to(device)
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.view(-1, 1).to(device)

                    outputs = model(x_batch)
                    y_pred = outputs[0]
                    y_pred = torch.sigmoid(y_pred)

                    # 손실 계산
                    loss = criterion(y_pred, y_batch)
                    total_loss += loss.item()

                    predicted = y_pred >= 0.5

                    correct += (predicted.float() == y_batch).sum().item()
                    total += y_batch.size(0)

                    probs = torch.sigmoid(y_pred).cpu().numpy()
                    all_preds.extend(predicted.cpu().numpy().flatten())
                    all_targets.extend(y_batch.cpu().numpy().flatten())

            avg_loss = total_loss / len(test_loader)
            accuracy = correct / total * 100
            auc_score = roc_auc_score(all_targets, all_preds)
            conf_matrix = confusion_matrix(all_targets, all_preds)

        
        # metrics=None    
        
        model.to("cpu")  # move model back to CPU
        return avg_loss, auc_score, conf_matrix
    
    return custom_test_torch

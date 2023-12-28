import json
import logging
from collections import Counter
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


# set log format
handlers_list = [logging.StreamHandler()]

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)


"""
Create your data loader for training/testing.
Keep the value of the return variable for normal operation.
----------------------------------------------------------
dataset example
"""


# Pytorch version

# Define a custom Dataset class
class ChristDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels
        
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

# MNIST
def load_partition(dataset, validation_split, label_count, batch_size):
    # 데이터 로드 및 전처리
    chris_data = pd.read_csv('./ischris.csv')

    X_chris = chris_data.iloc[:, :-1]
    y_chris = chris_data.iloc[:, -1]
    imputer = SimpleImputer(strategy='mean')
    X_chris_imputed = imputer.fit_transform(X_chris)
    X_chris_imputed = pd.DataFrame(X_chris_imputed, columns=X_chris.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_chris_imputed, y_chris, test_size=0.2, random_state=random_seed, stratify=y_chris)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_seed, stratify=y_train)

    # Dataset 객체 생성
    train_dataset = ChristDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr.values, dtype=torch.float32))
    val_dataset = ChristDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32))
    test_dataset = ChristDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))

    minibatch_size = 64 # @@ 32로 테스트 해보기
    # DataLoader 설정
    train_loader = DataLoader(train_dataset, batch_size = minibatch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=minibatch_size, shuffle=False) # shuffle False로 테스트 해보기
    test_loader = DataLoader(test_dataset, batch_size=minibatch_size, shuffle=False)
    y_label_counter = 1

    return train_loader, val_loader, test_loader, y_label_counter

def gl_model_torch_validation(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Adjusted for grayscale
    ])

    chris_data = pd.read_csv('./ischris.csv')

    X_chris = chris_data.iloc[:, :-1]
    y_chris = chris_data.iloc[:, -1]
    imputer = SimpleImputer(strategy='mean')
    X_chris_imputed = imputer.fit_transform(X_chris)
    X_chris_imputed = pd.DataFrame(X_chris_imputed, columns=X_chris.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_chris_imputed, y_chris, test_size=0.2, random_state=random_seed, stratify=y_chris)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_seed, stratify=y_train)

    # Load the test set of MNIST Dataset
    val_dataset = ChristDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32))

    # DataLoader for validation
    gl_val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return gl_val_loader
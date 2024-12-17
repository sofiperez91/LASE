import sys
import pickle
import argparse
import time
import numpy as np
import torch
from torch import nn
from scipy.stats import sem
sys.path.append("../")

from models.FCNN import FullyConnectedNet

parser = argparse.ArgumentParser(description='Classifier')
parser.add_argument('--dataset', type=str, default='cora', help='[cora, citeseer, amazon, chameleon, squirrel, cornell]')
parser.add_argument('--iter', type=int, default=10)


args = parser.parse_args()
dataset = args.dataset
total_iter = args.iter

DATASET_FILE = f'../data/real_dataset/{dataset}_dataset.pkl'
MODEL_FILE_FCNN = f'../saved_models/{dataset}/{dataset}_fcnn.pt'
FCNN_RESULTS = f'./results/{dataset}/{dataset}_FCNN_results.pkl'
device = 'cuda'

## LOAD DATASET 
with open(DATASET_FILE, 'rb') as f:
    data = pickle.load(f)


data = data.to(device)

input_dim = data.x.shape[1] # dimensionality of the feature embeddings
hidden_dim = 64 
output_dim = torch.unique(data.y).shape[0] # number of classes
lr=1e-2
epochs = 1000


acc_fcnn_test = []
for iter in range(total_iter):
    train_index = data.train_idx[:,iter].to(device)
    val_index = data.val_idx[:,iter].to(device)
    test_index = data.test_idx[:,iter].to(device)

    model = FullyConnectedNet(input_dim, hidden_dim, output_dim)
    model.to(device) 

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loss_epoch=[]
    val_loss_epoch=[]
    test_loss_epoch=[]
    best_valid_acc = 0
    best_epoch = 0

    start = time.time()

    for epoch in range(epochs):
        ## Train
        model.train()
        optimizer.zero_grad()
        out = model(data.x[train_index])
        loss =  criterion(out, data.y[train_index].squeeze().to(device))
        loss.backward() 
        optimizer.step() 
        
        # Calculate accuracy
        _, predicted_labels = torch.max(out.squeeze(),1)
        total_train_correct = (predicted_labels.squeeze() == data.y[train_index].squeeze().to(device)).sum().item()
        total_train_samples = len(train_index)
        train_acc = total_train_correct / total_train_samples
        total_train_loss = loss.item()    

        ## Val
        model.eval()
        out = model(data.x[val_index])
        loss =  criterion(out, data.y[val_index].squeeze().to(device))
            
        # Calculate accuracy
        _, predicted_labels = torch.max(out.squeeze(),1)
        total_val_correct = (predicted_labels.squeeze() == data.y[val_index].squeeze().to(device)).sum().item()
        total_val_samples = len(val_index)
        valid_acc = total_val_correct / total_val_samples
        total_val_loss = loss.item()        
        
        ## Test
        out = model(data.x[test_index])
        loss =  criterion(out, data.y[test_index].squeeze().to(device))
            
        # Calculate accuracy
        _, predicted_labels = torch.max(out.squeeze(),1)
        total_test_correct = (predicted_labels.squeeze() == data.y[test_index].squeeze().to(device)).sum().item()
        total_test_samples = len(test_index)
        test_acc = total_test_correct / total_test_samples
        total_test_loss = loss.item()      
                
        train_loss_epoch.append(total_train_loss)
        val_loss_epoch.append(total_val_loss)
        test_loss_epoch.append(total_test_loss)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), MODEL_FILE_FCNN)
            best_epoch = epoch

        if epoch % 100 == 0:        
            # Print epoch statistics
            print(f'Epoch: {epoch:02d}, '
                f'Loss: {total_train_loss:.4f}, '
                f'Train: {100 * train_acc:.2f}%, '
                f'Valid: {100 * valid_acc:.2f}% '
                f'Test: {100 * test_acc:.2f}%')

    stop = time.time()
    print(f"Training time: {stop - start}s")


    model.load_state_dict(torch.load(MODEL_FILE_FCNN))
    model.to(device)

    model.eval()
    out = model(data.x[test_index])
    loss =  criterion(out, data.y[test_index].squeeze().to(device))

    # Calculate accuracy
    _, predicted_labels = torch.max(out.squeeze(),1)
    total_test_correct = (predicted_labels.squeeze() == data.y[test_index].squeeze().to(device)).sum().item()
    total_test_samples = len(test_index)
    test_acc = total_test_correct / total_test_samples
    total_test_loss = loss.item()      
    acc_fcnn_test.append(test_acc)
    
with open(FCNN_RESULTS, 'wb') as f:
    pickle.dump(acc_fcnn_test, f)

print(f'{np.array(acc_fcnn_test).mean()*100} +/- {sem(np.array(acc_fcnn_test))*100}')
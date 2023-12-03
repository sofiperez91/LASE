import sys
sys.path.append("../")

import argparse
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import to_dense_adj
from models.glase_classifier import glaseClassifierGAT, FeatureClassifierGAT
from models.GLASE_unshared_normalized import gLASE
import pickle
import time
import math
from training.get_init import get_x_init
from graspologic.embed import AdjacencySpectralEmbed 


parser = argparse.ArgumentParser(description='Classifier')
parser.add_argument('--dataset', type=str, default='cora', help='[cora, amazon, chameleon]')


args = parser.parse_args()
dataset = args.dataset
    

GLASE_EMBEDDINGS=f'../saved_models/glase_unshared_{dataset}_5steps.pt'
Q_FILE = f'../data/{dataset}_q.pkl'
DATASET_FILE = f'../data/{dataset}_dataset.pkl'
MODEL_FILE_GLASE = f'../saved_models/glase_{dataset}_classifier_GAT_GLASE_ER02.pt'
MODEL_FILE_ASE = f'../saved_models/glase_{dataset}_classifier_GAT_ASE_ER02.pt'
MODEL_FILE_FEAT = f'../saved_models/glase_{dataset}_classifier_GAT_FEAT_ER02.pt'
MASK_FILE = f'../data/{dataset}_mask_ER02.pkl'
GLASE_RESULTS = f'./results/glase_{dataset}_GAT_results_ER02.pkl'
ASE_RESULTS = f'./results/ase_{dataset}_GAT_results_ER02.pkl'
FEATURE_RESULTS = f'./results/feat_{dataset}_GAT_results_ER02.pkl'

## LOAD DATASET 
with open(DATASET_FILE, 'rb') as f:
    data = pickle.load(f)

print(data)

## GENERATE EMBEDDINGS
d = torch.unique(data.y).shape[0]
print(d)
gd_steps = 5
device = 'cpu'

num_nodes = data.num_nodes

glase_model = gLASE(d,d, gd_steps)
glase_model.load_state_dict(torch.load(GLASE_EMBEDDINGS))
glase_model.to(device)

with open(Q_FILE, 'rb') as f:
    q = pickle.load(f)

Q = torch.diag(q).to(device)
edge_index_2 = torch.ones([num_nodes,num_nodes],).nonzero().t().contiguous()
mask = (torch.ones([num_nodes,num_nodes],)-torch.eye(num_nodes)).nonzero().t().contiguous().to(device)
with open(MASK_FILE, 'rb') as f:
    mask = pickle.load(f)

x = get_x_init(num_nodes, d, 0, math.pi/2, 0, math.pi/2).to(device)
x_glase = glase_model(x, data.edge_index, edge_index_2, Q, mask)
x_glase = x_glase.detach()

## ASE EMBEDDINGS
adj_matrix = to_dense_adj(data.edge_index.to('cpu')).squeeze(0)
mask_matrix = to_dense_adj(mask).squeeze(0)
masked_adj = adj_matrix*mask_matrix
ase = AdjacencySpectralEmbed(n_components=d, diag_aug=True, algorithm='full')
x_ase = ase.fit_transform(masked_adj.numpy())
x_ase = torch.from_numpy(x_ase)

edge_index = masked_adj.nonzero().t().contiguous().to(device)


feature_dim = data.x.shape[1] # dimensionality of the word embeddings
embedding_dim = d  # dimensionality of the graph embeddings
hidden_dim = 32  # number of hidden units
h_embedding_dim = 32  ## TODO: SACAR
output_dim = d  # number of classes
n_layers = 3
dropout1 = 0.5
dropout2 = 0.5 ## TODO: SACAR
device = 'cpu'
epochs = 1000
lr=1e-2


## FEATURES
acc_feat_test = []
for iter in range(10):
    train_index = data.train_idx[:,iter]
    val_index = data.val_idx[:,iter]
    test_index = data.test_idx[:,iter]

    # model = FeatureClassifier(feature_dim, hidden_dim,output_dim, n_layers, dropout1, dropout2)
    model = FeatureClassifierGAT(feature_dim, hidden_dim,output_dim, n_layers, dropout1, dropout2, num_heads=1)
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
        # out = model(data.x)
        out = model(data.x, edge_index)
        loss =  criterion(out[train_index], data.y[train_index].squeeze().to(device))
        loss.backward() 
        optimizer.step() 
        
        # Calculate accuracy
        _, predicted_labels = torch.max(out[train_index].squeeze(),1)
        total_train_correct = (predicted_labels.squeeze() == data.y[train_index].squeeze().to(device)).sum().item()
        total_train_samples = len(train_index)
        train_acc = total_train_correct / total_train_samples
        total_train_loss = loss.item()    

        ## Val
        model.eval()
        # out = model(data.x)
        out = model(data.x, edge_index)
        loss =  criterion(out[val_index], data.y[val_index].squeeze().to(device))
            
        # Calculate accuracy
        _, predicted_labels = torch.max(out[val_index].squeeze(),1)
        total_val_correct = (predicted_labels.squeeze() == data.y[val_index].squeeze().to(device)).sum().item()
        total_val_samples = len(val_index)
        valid_acc = total_val_correct / total_val_samples
        total_val_loss = loss.item()        
        
        ## Test
        loss =  criterion(out[test_index], data.y[test_index].squeeze().to(device))
            
        # Calculate accuracy
        _, predicted_labels = torch.max(out[test_index].squeeze(),1)
        total_test_correct = (predicted_labels.squeeze() == data.y[test_index].squeeze().to(device)).sum().item()
        total_test_samples = len(test_index)
        test_acc = total_test_correct / total_test_samples
        total_test_loss = loss.item()      
                
        train_loss_epoch.append(total_train_loss)
        val_loss_epoch.append(total_val_loss)
        test_loss_epoch.append(total_test_loss)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), MODEL_FILE_FEAT)
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


    model.load_state_dict(torch.load(MODEL_FILE_FEAT))
    model.to(device)

    model.eval()
    # out = model(data.x)
    out = model(data.x, edge_index)
    loss =  criterion(out[test_index], data.y[test_index].squeeze().to(device))

    # Calculate accuracy
    _, predicted_labels = torch.max(out[test_index].squeeze(),1)
    total_test_correct = (predicted_labels.squeeze() == data.y[test_index].squeeze().to(device)).sum().item()
    total_test_samples = len(test_index)
    test_acc = total_test_correct / total_test_samples
    total_test_loss = loss.item()      

    print('Best epoch: ', best_epoch)
    print(f'Test accuracy:  {100 * test_acc:.2f}')
    acc_feat_test.append(test_acc)
    
with open(FEATURE_RESULTS, 'wb') as f:
    pickle.dump(acc_feat_test, f)


## TRAIN ASE
acc_ase_test = []
for iter in range(10):
    
    train_index = data.train_idx[:,iter]
    val_index = data.val_idx[:,iter]
    test_index = data.test_idx[:,iter]
    
    model = glaseClassifierGAT(feature_dim, embedding_dim, hidden_dim, h_embedding_dim,output_dim, n_layers, dropout1, dropout2, num_heads=1)
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
        out = model(data.x, x_ase, edge_index)
        loss =  criterion(out[train_index], data.y[train_index].squeeze().to(device))
        loss.backward() 
        optimizer.step() 
        
        # Calculate accuracy
        _, predicted_labels = torch.max(out[train_index].squeeze(),1)
        total_train_correct = (predicted_labels.squeeze() == data.y[train_index].squeeze().to(device)).sum().item()
        total_train_samples = len(train_index)
        train_acc = total_train_correct / total_train_samples
        total_train_loss = loss.item()    

        ## Val
        model.eval()
        out = model(data.x, x_ase, edge_index)
        loss =  criterion(out[val_index], data.y[val_index].squeeze().to(device))
            
        # Calculate accuracy
        _, predicted_labels = torch.max(out[val_index].squeeze(),1)
        total_val_correct = (predicted_labels.squeeze() == data.y[val_index].squeeze().to(device)).sum().item()
        total_val_samples = len(val_index)
        valid_acc = total_val_correct / total_val_samples
        total_val_loss = loss.item()        
        
        ## Test
        loss =  criterion(out[test_index], data.y[test_index].squeeze().to(device))
            
        # Calculate accuracy
        _, predicted_labels = torch.max(out[test_index].squeeze(),1)
        total_test_correct = (predicted_labels.squeeze() == data.y[test_index].squeeze().to(device)).sum().item()
        total_test_samples = len(test_index)
        test_acc = total_test_correct / total_test_samples
        total_test_loss = loss.item()      
                
        train_loss_epoch.append(total_train_loss)
        val_loss_epoch.append(total_val_loss)
        test_loss_epoch.append(total_test_loss)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), MODEL_FILE_ASE)
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


    model.load_state_dict(torch.load(MODEL_FILE_ASE))
    model.to(device)

    model.eval()
    out = model(data.x, x_ase, edge_index)
    loss =  criterion(out[test_index], data.y[test_index].squeeze().to(device))

    # Calculate accuracy
    _, predicted_labels = torch.max(out[test_index].squeeze(),1)
    total_test_correct = (predicted_labels.squeeze() == data.y[test_index].squeeze().to(device)).sum().item()
    total_test_samples = len(test_index)
    test_acc = total_test_correct / total_test_samples
    total_test_loss = loss.item()      

    print('Best epoch: ', best_epoch)
    print(f'Test accuracy:  {100 * test_acc:.2f}')
    acc_ase_test.append(test_acc)

with open(ASE_RESULTS, 'wb') as f:
    pickle.dump(acc_ase_test, f)


## GLASE
acc_glase_test = []
for iter in range(10):
    
    train_index = data.train_idx[:,iter]
    val_index = data.val_idx[:,iter]
    test_index = data.test_idx[:,iter]
    
    model = glaseClassifierGAT(feature_dim, embedding_dim, hidden_dim, h_embedding_dim,output_dim, n_layers, dropout1, dropout2, num_heads=1)
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
        out = model(data.x, x_glase, edge_index)
        loss =  criterion(out[train_index], data.y[train_index].squeeze().to(device))
        loss.backward() 
        optimizer.step() 
        
        # Calculate accuracy
        _, predicted_labels = torch.max(out[train_index].squeeze(),1)
        total_train_correct = (predicted_labels.squeeze() == data.y[train_index].squeeze().to(device)).sum().item()
        total_train_samples = len(train_index)
        train_acc = total_train_correct / total_train_samples
        total_train_loss = loss.item()    

        ## Val
        model.eval()
        out = model(data.x, x_glase, edge_index)
        loss =  criterion(out[val_index], data.y[val_index].squeeze().to(device))
            
        # Calculate accuracy
        _, predicted_labels = torch.max(out[val_index].squeeze(),1)
        total_val_correct = (predicted_labels.squeeze() == data.y[val_index].squeeze().to(device)).sum().item()
        total_val_samples = len(val_index)
        valid_acc = total_val_correct / total_val_samples
        total_val_loss = loss.item()        
        
        ## Test
        loss =  criterion(out[test_index], data.y[test_index].squeeze().to(device))
            
        # Calculate accuracy
        _, predicted_labels = torch.max(out[test_index].squeeze(),1)
        total_test_correct = (predicted_labels.squeeze() == data.y[test_index].squeeze().to(device)).sum().item()
        total_test_samples = len(test_index)
        test_acc = total_test_correct / total_test_samples
        total_test_loss = loss.item()      
                
        train_loss_epoch.append(total_train_loss)
        val_loss_epoch.append(total_val_loss)
        test_loss_epoch.append(total_test_loss)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), MODEL_FILE_GLASE)
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


    model.load_state_dict(torch.load(MODEL_FILE_GLASE))
    model.to(device)

    model.eval()
    out = model(data.x, x_glase, edge_index)
    loss =  criterion(out[test_index], data.y[test_index].squeeze().to(device))

    # Calculate accuracy
    _, predicted_labels = torch.max(out[test_index].squeeze(),1)
    total_test_correct = (predicted_labels.squeeze() == data.y[test_index].squeeze().to(device)).sum().item()
    total_test_samples = len(test_index)
    test_acc = total_test_correct / total_test_samples
    total_test_loss = loss.item()      

    print('Best epoch: ', best_epoch)
    print(f'Test accuracy:  {100 * test_acc:.2f}')
    acc_glase_test.append(test_acc)
    
with open(GLASE_RESULTS, 'wb') as f:
    pickle.dump(acc_glase_test, f)

print(f'{np.array(acc_feat_test).mean()*100} +/- {np.array(acc_feat_test).std()*100}')
print(f'{np.array(acc_ase_test).mean()*100} +/- {np.array(acc_ase_test).std()*100}')
print(f'{np.array(acc_glase_test).mean()*100} +/- {np.array(acc_glase_test).std()*100}')


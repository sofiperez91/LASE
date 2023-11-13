import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import to_dense_adj, stochastic_blockmodel_graph
from models.glase_classifier import glaseClassifier
from models.GRDPG_GD_Unroll_unshared_normalized import gLASE
import pickle
from models.early_stopper import EarlyStopper
import time
import math
from training.get_init import get_x_init
from graspologic.embed import AdjacencySpectralEmbed 
from sklearn.model_selection import KFold



GLASE_EMBEDDINGS='../../saved_models/paper/glase_unshared_d7_cora.pt'
Q_FILE = '../data/cora_q.pkl'
DATASET_FILE = '../dataset/Cora/cora_dataset.pkl'
MODEL_FILE='../../saved_models/paper/glase_classifier_cora_ER02.pt'
MASK_FILE = '../dataset/Cora/cora_mask_ER02.pkl'
GLASE_RESULTS = './results/glase_cora_d7_results_ER02.pkl'
ASE_RESULTS = './results/ase_cora_d7_results_ER02.pkl'

## LOAD DATASET 
with open(DATASET_FILE, 'rb') as f:
    data = pickle.load(f)

print(data)

## GENERATE EMBEDDINGS
d = 7
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
# mask = (torch.ones([num_nodes,num_nodes],)-torch.eye(num_nodes)).nonzero().t().contiguous().to(device)
with open(MASK_FILE, 'rb') as f:
    mask = pickle.load(f)


x = get_x_init(num_nodes, d, 0, math.pi/2, 0, math.pi/2).to(device)
x_glase = glase_model(x, data.edge_index, edge_index_2, Q, mask)
x_glase = x_glase.detach()

## ASE EMBEDDINGS
adj_matrix = to_dense_adj(data.edge_index.to('cpu')).squeeze(0).numpy()
mask_matrix = to_dense_adj(mask).squeeze(0).numpy()
masked_adj = adj_matrix*mask_matrix
ase = AdjacencySpectralEmbed(n_components=d, diag_aug=True, algorithm='full')
x_ase = ase.fit_transform(masked_adj)
x_ase = torch.from_numpy(x_ase)


feature_dim = 1433 # dimensionality of the word embeddings
embedding_dim = 7  # dimensionality of the graph embeddings
hidden_dim = 256  # number of hidden units\
h_embedding_dim = 32
output_dim = 7  # number of classes
n_layers = 3
dropout1 = 0.5
dropout2 = 0.5
device = 'cpu'
epochs = 500
lr=1e-2

acc_glase_test = []

## SPLIT TRAIN, VAL
train_samples = np.array(data.train_idx)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Split the data into 10 folds
for train_, val_ in kf.split(train_samples):
    train_index = train_samples[train_]
    val_index = train_samples[val_]
    test_index = data.test_idx
    
    model = glaseClassifier(feature_dim, embedding_dim, hidden_dim, h_embedding_dim,output_dim, n_layers, dropout1, dropout2, gd_steps)
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
        out = model(data.x, x_glase)
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
        out = model(data.x, x_glase)
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
            torch.save(model.state_dict(), MODEL_FILE)
            best_epoch = epoch
        
        # Print epoch statistics
        print(f'Epoch: {epoch:02d}, '
            f'Loss: {total_train_loss:.4f}, '
            f'Train: {100 * train_acc:.2f}%, '
            f'Valid: {100 * valid_acc:.2f}% '
            f'Test: {100 * test_acc:.2f}%')

    stop = time.time()
    print(f"Training time: {stop - start}s")


    model.load_state_dict(torch.load(MODEL_FILE))
    model.to(device)

    model.eval()
    out = model(data.x, x_glase)
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

## TRAIN ASE

acc_ase_test = []

for train_, val_ in kf.split(train_samples):
    train_index = train_samples[train_]
    val_index = train_samples[val_]
    test_index = data.test_idx
    
    model = glaseClassifier(feature_dim, embedding_dim, hidden_dim, h_embedding_dim,output_dim, n_layers, dropout1, dropout2, gd_steps)
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
        out = model(data.x, x_ase)
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
        out = model(data.x, x_ase)
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
            torch.save(model.state_dict(), MODEL_FILE)
            best_epoch = epoch
        
        # Print epoch statistics
        print(f'Epoch: {epoch:02d}, '
            f'Loss: {total_train_loss:.4f}, '
            f'Train: {100 * train_acc:.2f}%, '
            f'Valid: {100 * valid_acc:.2f}% '
            f'Test: {100 * test_acc:.2f}%')

    stop = time.time()
    print(f"Training time: {stop - start}s")


    model.load_state_dict(torch.load(MODEL_FILE))
    model.to(device)

    model.eval()
    out = model(data.x, x_ase)
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
    
print(f'{np.array(acc_glase_test).mean()*100} +/- {np.array(acc_glase_test).std()*100}')
print(f'{np.array(acc_ase_test).mean()*100} +/- {np.array(acc_ase_test).std()*100}')
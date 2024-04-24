![LASE](docs/imgs/graph_networks.png)
[![License](https://img.shields.io/github/license/sofiperez91/LASE)](LICENSE)

# Learned Adyacency Spectral Embeddings (LASE)

## Dependencies
- Python 3.10+
- Pytorch 2.0+
- Pytorch-geometric 2.5+
- CUDA 12.0 

## Setup
Clone repo and create a dedicated Python environment:
```
python -m venv lase
source lase/bin/activate
pip install -r requirements.txt
```

## Data
### Synthetic datasets
To build an SBM dataset first use the `~/LASE/data/data_config.json` to load its configuration. Structure should be as follows:

**Simple training**
```
    "sbm2_unbalanced_positive": {
        "d": 2,
        "n": [70, 30],
        "p": [
                [0.9, 0.1],
                [0.1, 0.5]
            ],
        "total_samples": 1000,
        "train_samples": 800,
        "mode": "simple"
    }
```
**Subgraph training**
```
    "sbm3_unbalanced_positive_subgraphs": {
        "d": 3,
        "n": [6000, 4000, 2000],
        "p": [
                [0.9, 0.2, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.2, 0.7]
            ],
        "total_samples": 1000, 
        "train_samples": 800,
        "mode": "subgraphs",
        "dropout": 95
    },
```
Once loaded run the following script:

```
cd ~/LASE/data
python build_sbm_dataset.py --dataset='sbm2_unbalanced_positive'
```

### Real datasets
Use the following to load the available real world datasets. Options include: `cora`, `amazon`, `squirrel`, `crocodile`, `cornell`, `texas` and `wisconsin`

```
cd ~/LASE/data
python build_real_dataset.py --dataset='cora' --d=6
```

## Training 

### LASE embeddings on SBM

```
cd ~/LASE/training
python train_lase_unshared_normalized.py --dataset='sbm2_unbalanced_positive' --gd_steps=5 --epochs=200 --init='random' --att='FULL'
```

Options include:
- **dataset**: Any of the configurations loaded in `~/LASE/data/data_config.json`
- **glase_steps**: number of GLASE layers
- **init**: `ones`, `random` or `glase_init`
- **att**: `FULL`, `MO8`, `M06`, `M04` and `M02`

### LASE embeddings on SBM with missing information

```
cd ~/LASE/training
python train_lase_unshared_normalized_missing.py --dataset='sbm2_unbalanced_positive' --gd_steps=5 --epochs=200 --init='random' --mask_threshold=0.7
```

Options include:
- **dataset**: Any of the configurations loaded in `~/LASE/data/data_config.json`
- **glase_steps**: number of GLASE layers
- **init**: `ones`, `random` or `glase_init`
- **mask_threshold**: A float indicating the degree of sparcity of the mask matrix


### GLASE embeddings on SBM


```
cd ~/LASE/training
python train_glase_unshared_normalized.py --dataset='sbm5_unbalanced_negative' --gd_steps=5 --epochs=200 --init='glase_init'
```

Options include:
- **dataset**: Any of the configurations loaded in `~/LASE/data/data_config.json`
- **glase_steps**: number of GLASE layers
- **init**: `ones`, `random` or `glase_init`



### Node classification use case:

To train GLASE embeddings from the real world datasets run the following snippet of code:
```
cd ~/LASE/training
python train_glase_dataset_embeddings.py --dataset='cora' --mask='FULL' --d=6 --glase_steps=5
```

Options include:
- **dataset**: `cora`, `amazon`, `squirrel`, `crocodile`, `cornell`, `texas` and `wisconsin`.
- **mask**: `FULL`, `MO8`, `M06`, `M04` and `M02`.
- **d**: embedding dimension
- **glase_steps**: number of GLASE layers

To train the GAT-based node classifier use the following:

```
cd ~/LASE/training
python train_glase_dataset_embeddings.py --dataset='cora' --mask='FULL' --d=6 --glase_steps=5
```

Options include:
- **dataset**: `cora`, `amazon`, `squirrel`, `crocodile`, `cornell`, `texas` and `wisconsin`.
- **mask**: `FULL`, `MO8`, `M06`, `M04` and `M02`.
- **d**: embedding dimension
- **glase_steps**: number of GLASE layers

This script will train 3 models:
- using only node features
- using node features and concatenating ASE PE
- using node features and concatenating GLASE PE


To train the e2e variant use the following:

```
cd ~/LASE/training
python train_glase_e2e_classifier.py --dataset="cora" --mask="FULL" --d=6
```
Options include:
- **dataset**: `cora`, `amazon`, `squirrel`, `crocodile`, `cornell`, `texas` and `wisconsin`.
- **mask**: `FULL`, `MO8`, `M06`, `M04` and `M02`.
- **d**: embedding dimension


### Link prediction use case:

**Senators:**
```
cd ~/LASE/training
python link_prediction_senators.py --mask_threshold=0.7 --iter=10
```

**ONU:**
```
cd ~/LASE/training
python link_prediction_onu.py
```


## Measure performance
To evaluate the performance of LASE against other models use the following:

```
python measure_lase_performance.py --model='LASE_BB03' --dataset='sbm3_unbalanced_positive_v2'
```

Options include:
- **model**: `svd`, `cgd`, `LASE_FULL`, `LASE_ER05`, `LASE_WS03` and `LASE_BB03`
- **dataset**: Any of the configurations loaded in `~/LASE/data/data_config.json`

## Visualizations
In the `notebook` folder there are several of the visualizations used in the documentation. 

### SBM visualizations:
- `LASE_section.ipynb`
- `LASE_subgraphs_section.ipynb`
- `LASE_missing_data.ipynb`
- `LASE_sparse_attention_section.ipynb`
- `GLASE_section.ipynb`
- `GLASE_performance.ipynb`
- `senators_link_prediction.ipynb`

### Real-world datasets visualizations
- `GLASE_real_datasets.ipynb`
- `onu_link_prediction.ipynb`

### GCN eigenvector analysis:
- `GCN_eigenvector_analysis.ipynb`
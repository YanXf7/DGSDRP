# DGSDRP
The source code of paper "deep graph and sequence representation learning for drug response prediction"
# Resources:
* README.md: this file.
* data: GDSC dataset
## source codes:
* preprocess.py: create data in pytorch format.
* utils.py: include TestbedDataset used by create_data.py to create data, performance measures and functions to draw loss, pearson by epoch.
* models/ginconv.py, gat.py, gat_gcn.py, and gcn.py: proposed models GINConvNet and GAT_GCN receiving graphs as input for drugs.
* training.py: train a GraphDRP model.

# Dependencies
* Torch
* Pytorch_geometric
* Rdkit
* Matplotlib
* Pandas
* Numpy
* Scipy
# Step-by-step running:
* Create data in pytorch format
```
python preprocess.py --choice 0 
```
choice:    0: create mixed test dataset     1: create saliency map dataset     2: create blind drug dataset      3: create blind cell dataset  
  
This returns file pytorch format (.pt) stored at data/processed including training, validation, test set.
* Train a GraphDRP model
```
python training.py --model 0 --train_batch 1024 --val_batch 1024 --test_batch 1024 --lr 0.0005 --num_epoch 1000 --log_interval 20 --cuda_name "cuda:0"
```
model:       0: GINConvNet       1: GAT_GCN  
  
To train a model using training data. The model is chosen if it gains the best MSE for testing data.  

This returns the model and result files for the modelling achieving the best MSE for testing data throughout the training.

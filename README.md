# distance_constrained_ANN

Source code for my master thesis project: Implementing a Brain-inspired Distance 
Constraint into Artificial Neural Networks

This README file provides an overview of how to run the model with a distance
constraint on a fully-connected network or a sparse one. Additionally, the
script also produces an $L^p$ regularized model as a contrast to the
distance-modulated impacts. 

## A distance-constrained model
In each sub-project folder, you can find a `requirements.txt` for all
required dependencies to run each sub-project. Training models require a machine
with graphics accelaration. 

Scripts for the fully connected neural network (FCNN) and recurrent neural
network (RNN) are stored in directories `~/fcnn` and `~/rnn` respectively. 

### Dataset
The FCNN model is trained on MNIST dataset, and the RNN model on a 3-bit memory
task. Both dataset will be automatically downloaded or generated once the
training is initialized.

### Experiments 
* A json file (`params.json` for FCNN models and `hps.json` for RNN
  models) stores the value of parameters and hyperparameters used for the
  training and inference. Before training, the user can manually change the
  setting in the json file if needed and store file together with the trained
  models for inference.
* `train.py` produces a trained model with either a distance constraint
  ($\beta = 1.0$) or an $L^p$ regularization ($\beta = 0.0$). 
* `train_sparse.py` uses the same framework as `train.py` but it adds a
  binary search pruning algorithm (see Chapter 2.3) to remove weights smaller
  than a threshold. This script is used for getting a network model with the
  sparset connection topology while maintaining the performance over the
  bottom-line accuracy.

To initialize the training, the user can type the following in the terminal:
```python3 train*.py --seed $seed```
Please refer to the corresponding script for more settings. 

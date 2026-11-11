import pickle as pkl
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

# Define the fully connected neural network
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        # x = torch.softmax(self.fc3(x))  # softmax for multilabel classification
        return x

    
# load the meta classifier
def meta_classifier(nn_struct_preds,model,lb):
    """ Get predictions from the integrated (meta classifier) model"""
    from meetneighbors.embedding_pipeline import METADATA

    with model.open('rb') as handle:
        classifier = pkl.load(handle)
    # Preserve the trained feature order, independently of metadata columns.
    columns = [f'{category}_{source}' for source in ('x', 'y') for category in lb.classes_]
    features = nn_struct_preds[columns].copy()
    features.columns = [f'{i}_{source}' for source in ('x', 'y') for i in range(len(lb.classes_))]
    meta_preds = pd.DataFrame(classifier.predict_proba(features), columns=list(lb.classes_))
    metadata = [column for column in METADATA if column in nn_struct_preds.columns]
    return pd.concat([nn_struct_preds[metadata].reset_index(drop=True), meta_preds], axis=1)

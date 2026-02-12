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
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
#         x = torch.softmax(self.fc3(x))  # softmax for multilabel classification
        return x

    
# load the meta classifier
def meta_classifier(nn_struct_preds,model,lb):
    """ Get predictions from the integrated (meta classifier) model"""
    model = pkl.load(model.open('rb'))
    first_feature_ind,num_features = 2,12
    # first_feature_ind,num_features = 1,12 # for centroids
    newcolnames = {col:str(i)+'_x' if '_x' in col else str(i-num_features)+'_y'
               for i,col in enumerate(nn_struct_preds.iloc[:,first_feature_ind:].columns)} # gotta match colnames to what was seen during fit
    nn_struct_preds.rename(columns=newcolnames,inplace=True)
    meta_preds = model.predict_proba(nn_struct_preds.iloc[:,first_feature_ind:])
                        
                        
    meta_preds = pd.DataFrame(meta_preds)
    meta_preds.columns = [cat for cat in lb.classes_]
    # meta_preds = pd.concat([nn_struct_preds[['query','neighborhood_name','seq_annotations']],meta_preds],axis=1)
    # meta_preds = pd.concat([nn_struct_preds[['query','neighborhood_name']],meta_preds],axis=1)
    meta_preds = pd.concat([nn_struct_preds[['query']],meta_preds],axis=1)
    return meta_preds
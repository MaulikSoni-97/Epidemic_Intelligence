import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from Utils import get_max_lengths,get_evaluation
from Data_Loader import MEDWEB
from Hierarchy_Attation import HierAttNet
import argparse
import shutil
import numpy as np


def train(batch_size,num_epochs,lr,word_hidden_size,train_set,word2vec_path,saved_path,momentum,metric_Id):

    #checking for gpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2)
    else:
        torch.manual_seed(2)

    training_params = {"batch_size": batch_size,
                       "shuffle": True,
                       "drop_last": False}
    

    max_word_length = get_max_lengths(train_set)
    training_set = MEDWEB(train_set,word2vec_path, max_word_length)
    training_generator = DataLoader(training_set, **training_params)
    
    model = HierAttNet(word_hidden_size, batch_size,word2vec_path,max_word_length)

    if torch.cuda.is_available():
        model.cuda()

    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    model.train()
    num_iter_per_epoch = len(training_generator)
    for epoch in range(num_epochs):
        for iter, (feature, label) in enumerate(training_generator):
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            model._init_hidden_state()
            predictions = model(feature)
            loss = get_evaluation(metric_Id,batch_size,predictions,label)
            loss.backward()
            optimizer.step()
            # training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
            print("Epoch: {}/{}, Iteration: {},loss={} ".format(
                epoch + 1,
                num_epochs,
                iter + 1,
                loss      
                ))
        if epoch==num_epochs-1:
            torch.save(model,saved_path)


if __name__ == '__main__':
    
    batch_size = 64
    num_epochs = 15
    lr = 0.01  
    momentum = 0.9
    train_set = 'train dataset file with .pkl extension'

    # change the path of glove pretrained word2vec library as preferable
    # download twitter word2vec library from 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    # use 100 dimension txt file 
    word_hidden_size = 100 #dimension should be match with given  

    # mention the path of word2vec library
    word2vec_path = 'glove text file having  100 dimension , .txt extension'

    # give the path to saved model(.spt extension) which will required in future use
    # three models so three diffrent paths for three loss function
    # change at each running type
    saved_path = 'path to save model with .pt extension'
    
    # 1 for NLL, 2 for Hinge, 3 for HingeSqr loss function
    metric_Id = 1
    train(batch_size,num_epochs,lr,word_hidden_size,train_set,word2vec_path,saved_path,momentum,metric_Id)
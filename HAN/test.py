import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Utils import get_max_lengths
from Data_Loader import MEDWEB
from Hierarchy_Attation import HierAttNet
from Word_Attation import WordAttNet
import argparse
import shutil
import csv
import numpy as np

def test(batch_size,pre_trained_model,train_set,test_set,word2vec_path,prediction_result_path):
    test_params = {"batch_size":batch_size,
                   "shuffle": False,
                   "drop_last": False}
    
    max_word_length = get_max_lengths(train_set)
    if torch.cuda.is_available():
        model = torch.load(pre_trained_model)
        model = HierAttNet(word_hidden_size, batch_size,word2vec_path,max_word_length)
        model.load_state_dict(torch.load(pre_trained_model))
    else:
        model = torch.load(pre_trained_model, map_location=lambda storage, loc: storage)

    test_set = MEDWEB(test_set, word2vec_path, max_word_length)
    test_generator = DataLoader(test_set, **test_params)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    te_label_ls = []
    te_pred_ls = []
    for te_feature, te_label in test_generator:
        num_sample = len(te_label)
        if torch.cuda.is_available():
            te_feature = te_feature.cuda()
            te_label = te_label.cuda()
        with torch.no_grad():
            model._init_hidden_state()
            te_predictions = model(te_feature)
          
        te_label_ls.append(te_label.clone().cpu())
        te_pred_ls.append(te_predictions.clone().cpu())

    te_pred = torch.cat(te_pred_ls, 0).numpy()
    te_label = torch.cat(te_label_ls,0).numpy()
  
    np.save(prediction_result_path',te_pred)
    te_pred_new=[]
    for i in range(len(te_pred)):
      temp=[]
      for j in range(8):
        if te_pred[i][j] < 0:
          temp.append(-1)
        else:
          temp.append(1)
      te_pred_new.append(temp)
    

    ExactMatch = 0
    for i in range(len(te_pred_new)):
      correctCount = 0
      for j in range(0,8):
        if int(te_pred_new[i][j]) == int(te_label[i][j]):
          correctCount+=1
      if correctCount == 8:
        ExactMatch += 1
    print((ExactMatch/len(te_pred))*100)

if __name__ == '__main__':
    batch_size = 64
    
    # three pretrained model path due to three losses
    pre_trained_model =  'pre trained model path with .pt extension' 
  
    train_set = 'train dataset file with .pkl extension'

    test_set = 'test dataset with .pkl extension'

    word2vec_path = 'glove text file having  100 dimension , .txt extension'
    
    # three path there will be due to three models 
    prediction_result_path = "give this path with .npy extension"

    test(batch_size,pre_trained_model,train_set,test_set,word2vec_path,prediction_result_path)
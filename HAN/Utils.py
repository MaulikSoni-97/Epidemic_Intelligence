import torch
import sys
import csv
# from torch.nn import SoftMarginLoss
import torch.nn as nn
csv.field_size_limit(sys.maxsize)
from nltk.tokenize import  word_tokenize
from sklearn import metrics
import numpy as np
import pandas as pd

def get_labels(data_path):
  'Function to get label for each tweets'
  data = pd.read_pickle(data_path)

  LabelList = []
  Labels = data[['Influenza', 'Diarrhea', 'Hayfever', 'Cough', 'Headache',
    'Fever', 'Runnynose', 'Cold']]

  for i in range(len(Labels)):
    TweetLabel = []
    for j in range(8):
      if Labels.iloc[i,j] == 'n': 
        TweetLabel.append(-1)
      else:
        TweetLabel.append(1)
    LabelList.append(np.array(TweetLabel))

  LabelList = np.array(LabelList)
  return LabelList

def get_evaluation(metric_Id,batch_size,predictions,labels):
  '''
  Three loss finction for the evaluation
  '''
  
  if metric_Id == 1:
      loss_Func = nn.SoftMarginLoss(reduction='sum')
      loss = 0
      for i in range(batch_size):
        loss += loss_Func(predictions[i].float(),labels[i].float())
      return loss

  elif metric_Id == 2:
     
      hinge_loss = 1 - torch.mul(predictions, labels)
      hinge_loss[hinge_loss < 0] = 0
  
      return torch.sum(hinge_loss)
  
  else:
      hinge_loss = 1 - torch.mul(predictions, labels)
      hinge_loss[hinge_loss < 0] = 0
      hinge_loss = torch.mul(hinge_loss,hinge_loss)
      return torch.sum(hinge_loss)

def matrix_mul1(input, weight, bias=False):
    '''
    special multiplication required as torch.mul wasn't useful
    '''
    feature_list = []
    flag=0
    
    for feature in input:
        feature = torch.mm(feature, weight)
       
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze() #squeez coverts (A, 1, B) -> (A, B) 

def matrix_mul2(input, weight, bias=False):
    '''
    special multiplication required as torch.mul wasn't useful
    '''
    feature_list = []
    flag=0
   
    for feature in input:
        feature = torch.mm(feature, weight)
        
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
            feature = torch.tanh(feature)
        feature_list.append(feature)

    return torch.cat(feature_list,1)

def element_wise_mul(input1, input2):
    '''
    multiplication of two matrix having different shape with additional utiity
    '''
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1) 
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0) 

    return torch.sum(output, 0).unsqueeze(0)

def get_max_lengths(data_path):
  '''
  gives the tweet length with maximum tokens
  '''
  data = pd.read_pickle(data_path) #try read csv in case not working

  tweets_len = []
  for i in range(len(data['Tweet'])):
      tweets_len.append(len(word_tokenize(data.loc[i,'Tweet'])))
      
  tweets_len.sort()
  print('max_len=',tweets_len[int(1*len(tweets_len))-1])

  return tweets_len[int(1*len(tweets_len))-1]

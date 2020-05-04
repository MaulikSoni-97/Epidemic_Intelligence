import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import csv
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from Utils import get_labels
nltk.download('punkt')

class MEDWEB(Dataset):
    """
      A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, data_path ,dict_path,max_tweet_length, is_train=True):
      
      self.data = pd.read_pickle(data_path)
      self.tweets = list(self.data['Tweet'])
      self.labels = get_labels(data_path) #shape (num_of_tweets,8)
      self.dict_path = dict_path
      self.max_tweet_length = max_tweet_length
      self.is_train = is_train

      self.dict = pd.read_csv(filepath_or_buffer=dict_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
      self.dict = [word[0] for word in self.dict]


    def __len__(self):
      return len(self.tweets)

    def __getitem__(self, index):

      label = self.labels[index] #shape (1,8)
      tweet = self.tweets[index] 

      tweet_encode = [self.dict.index(word) if word in self.dict else -1 
                      for word in word_tokenize(text=tweet)] 
      
      if len(tweet_encode) < self.max_tweet_length: #equaling length of each tweet
          extended_words = [-1 for _ in range(self.max_tweet_length - len(tweet_encode))]
          tweet_encode.extend(extended_words)

      tweet_encode = tweet_encode[:self.max_tweet_length] #may be give error if not in required length

      tweet_encode = np.stack(arrays=tweet_encode, axis=0) #list to numpy array
      tweet_encode += 1 #replacing -1 with 0 may be

      return tweet_encode.astype(np.int64), label

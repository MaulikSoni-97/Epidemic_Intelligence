import torch
import torch.nn as nn
from Word_Attation import WordAttNet

class HierAttNet(nn.Module):
    '''
    This class mainly intilize hidden states and intilize the wordAttNetwork
    Becuse of the single step it only intilize the Word Attation network otherwise
    for documnent classification class sentAttNetwork should be defined added to the 
    forward function.
    '''
    def __init__(self, word_hidden_size, batch_size, pretrained_word2vec_path,max_word_length,num_classes = 8):
         
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size #embedding size
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(pretrained_word2vec_path,num_classes,word_hidden_size)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda()
            

    def forward(self, input):

        output, self.word_hidden_state = self.word_att_net(input.permute(1,0), self.word_hidden_state)

        return output
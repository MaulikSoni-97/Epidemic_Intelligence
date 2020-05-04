## Run HAN Model:
### 1 Train.py <br />
- first set path to the 'Train_set' to the train data <br />
- second set path to the glove  word2vec text library which you can download from here.http://nlp.stanford.edu/data/glove.twitter.27B.zip.  please take care you choose 100 dimension text file which I have previously set 100D in 'word hidden size'. <br />
- Third give the path 'saved_path' to save the model. <br />
- Fourth, there is 'metric_Id' option which is for three different loss function , Negative log likelihood, Hinge an Hinge square loss as 1, 2, 3 number respectively. <br />
- So for three loss function you have to run the train file three times and you will get three models for three losses. <br />

### 2 Test.py  <br />
- First set 'pre-trained model' to any one of the path of the model. <br />
- Second provide path to 'train-set' and 'test-set' to path to the train and test datafile respectively. <br />
- Third set path to glove word2vec path library as done in train model section. <br />
- Fourth provide prediction result path as path to predition which would be required in .npy extension.  <br />
agian repeat for three other loss function. This way you wiil get three .npy files for three model respectively<br />

### 3 Baggig.py <br />
- Just provide test dataset path to 'test' and run it. :)

## *Run Stastical methods dirctly by running statistical_methods.ipynb file directly.*

## *Run Bert model by dirctly by running Bert_Method.ipynb file directly.*
- please note that for bert model please use data which is given in BERT folder and not the one in folder 'Data'.

## References:
- https://github.com/uvipen/Hierarchical-attention-networks-pytorch
- https://github.com/kaushaltrivedi/bert-toxic-comments-multilabel
- https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5
- https://towardsdatascience.com/multi-label-text-classification-5c505fdedca8


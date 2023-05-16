import torch
import os
from sklearn import svm
import numpy as np
from sklearn.decomposition import PCA
from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder

# CLS averaging processing function
def getCls(vector):
  vector=vector.mean(axis=0)
  return vector

# CLS data generation and writing functions
def data_write(input_data,output_file_name):
  embedder = ProtTransBertBFDEmbedder()
  cnt=0
  for i in input_data:
    embedding = embedder.embed(i)
    cls=getCls(embedding)
    cnt+=1
    print(cnt)
    print(cls)
    if not os.path.exists(output_file_name):
      os.system(r"touch {}".format(output_file_name))
    with open(output_file_name,'a') as f:
      f.write(str(cnt)+"\n")
      for j in cls:
        f.write(str(j)+" ")
      f.write("\n")

# CLS data reading function
def readClsData(cls_datafile,cls_data):
    cnt=0
    with open(cls_datafile,"r") as f:
        d=f.readline()
        while d:
            d=f.readline().split() 
            if d:
                temp=[]
                for i in d:
                    i=float(i)
                    temp.append(i)
                cnt+=1
                cls_data.append(temp)
            f.readline()

# Fasta data reading
path_ne_train="fastadata/Train/negative_train_sequence.fasta"
path_ne_test="fastadata/Independent_Test/negative_test_sequence.fasta"
path_po_train="fastadata/Train/positive_train_sequence.fasta"
path_po_test="fastadata/Independent_Test/positive_test_sequence.fasta"
negative_train=[]
negative_test=[]
positive_train=[]
positive_test=[]

with open (path_ne_train,"r") as f:
    d=f.readline()
    while d:
        d=f.readline().strip()
        f.readline()
        if d:
          negative_train.append(d)
    print(len(negative_train))
    
with open (path_ne_test,"r") as f:
    d=f.readline()
    while d:
        d=f.readline().strip()
        f.readline()
        if d:
            negative_test.append(d)
    print(len(negative_test))

with open (path_po_train,"r") as f:
    d=f.readline()
    while d:
        d=f.readline().strip()
        f.readline()
        if d:
            positive_train.append(d)
    print(len(positive_train))
    
with open (path_po_test,"r") as f:
    d=f.readline()
    while d:
        d=f.readline().strip()
        f.readline()
        if d:
            positive_test.append(d)
    print(len(positive_test))

# Generate CLS variables
path_ne_train_cls="clsdata/negative_train_cls.txt"
path_ne_test_cls="clsdata/negative_test_cls.txt"
path_po_train_cls="clsdata/positive_train_cls.txt"
path_po_test_cls="clsdata/positive_test_cls.txt"
data_write(negative_train,path_ne_train_cls)
data_write(negative_test,path_ne_test_cls)
data_write(positive_train,path_po_train_cls)
data_write(positive_test,path_po_test_cls)
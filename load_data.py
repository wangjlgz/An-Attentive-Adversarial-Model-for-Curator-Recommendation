import numpy
from keras.utils.data_utils import get_file
from keras.utils.np_utils import to_categorical
from zipfile import ZipFile
import numpy as np
import gensim, logging


def load_data_unified(dataset):
    
    train_R = {} #feedback on curators
    test_R = {}
    curator_set = set()
    item_set = set()

    train_R2 = {} #feedback on items
    path = 'data/' + dataset + '/'
    with open(path + 'train_user_curator.txt', 'r') as file:
        for line in file:
            user_id, curator_id = line.rstrip().split('\t')
            if not int(user_id) in train_R:
                train_R[int(user_id)] = []
                train_R2[int(user_id)] = []
                test_R[int(user_id)] = []
            train_R[int(user_id)].append(int(curator_id))
            curator_set.add(int(curator_id))

    with open(path + 'test_user_curator.txt', 'r') as file:
        for line in file:
            user_id, curator_id = line.rstrip().split('\t')
            test_R[int(user_id)].append(int(curator_id))
            curator_set.add(int(curator_id))

    num_curator = len(curator_set)
    num_user = len(train_R)

    for cu in curator_set:
        train_R[cu+num_user] = [cu]  #curator follow herself/himself
        train_R2[cu+num_user] = []
    
    with open(path + 'train_user_item.txt', 'r') as file:
        for line in file:
            user_id, item_id = line.rstrip().split('\t')                
            train_R2[int(user_id)].append(int(item_id))
            item_set.add(int(item_id))
    
    
    with open(path + 'train_curator_item.txt', 'r') as file:
        for line in file:
            user_id, item_id = line.rstrip().split('\t')
            train_R2[int(user_id)+num_user].append(int(item_id))
            item_set.add(int(item_id))

    num_item = len(item_set)

    return train_R,test_R,train_R2,num_user,num_item,num_curator
import sys
import os
import numpy as np
import numpy.random as npr
import pandas as pd
import random
from sklearn.model_selection import train_test_split

# Module file with functions that you fill in so that they can be
# called by the notebook.  This file should be in the same
# directory/folder as the notebook when you test with the notebook.

# You can modify anything here, add "helper" functions, more imports,
# and so on, as long as the notebook runs and produces a meaningful
# result (though not necessarily a good classifier).  Document your
# code with reasonable comments.

# Function for Part 1
def preprocess(inputfile):
    return inputfile.readlines()

# Code for part 2
class Instance:
    def __init__(self, neclass, features):
        self.neclass = neclass
        self.features = features

    def __str__(self):
        return "Class: {} Features: {}".format(self.neclass, self.features)

    def __repr__(self):
        return str(self)

def create_instances(data, pos_include = False, split_pos_vs_words = False):
    """
    Will use start tokens (<s1>,<s2>,<s3>,<s4>,<s5>) and end tokens (<e1>,<e2>,<e3>,<e4>,<e5>).
    These will encode the distance from the start and end of sentence. 
    We will also include and go past other named entities (NEs). I have decided that the actual
     names of other neighbouring entities themselves aren't relevant, so will have a generic <ne> tag for these. 
    The <ne> tag can represent a named entity spanning more than one token. 
    """
    #need to append extra rows to data in order to avoid out-of-index errors. 
    data_len = len(data)
    for i in range(1,6):
        data.insert(0,[-i,0,f'<s{6-i}>','NONE','O'])
        data.append([data[len(data)-1][0]+1,data[data_len-1][1]+1,f'<s{i}>','NONE','O'])

    #we iterate over the data not including the fake ones we just added (5 at beginning and end of list)
    instances = list()
    if split_pos_vs_words:
        instances_pos = list()
        
    for index in range(5, len(data)-5):
        entity_type = data[index][4]
        if entity_type == 'O':#dont do anything if the word isnt a NE
            continue
        if entity_type.split('-')[0] == 'I':
            continue
    #we only do the next bit for a word if it its the head (or B- tag) of a NE
        
        features_list = list()
        if split_pos_vs_words:
            features_list_pos_only = list()
        
        sent_num = data[index][1]
        crawl_count = 1 
        #the following generates the features list for before the target word
        while True:
            compared_entity_type = data[index - crawl_count][4]
            if sent_num == data[index - crawl_count][1]: #if word in the same sentence
                if pos_include == True:
                    if split_pos_vs_words:
                        features_list_pos_only.insert(0,data[index - crawl_count][3])#insert pos for neighbouring word
                    else:
                        features_list.insert(0,data[index - crawl_count][3])#insert pos for neighbouring word
                if compared_entity_type == 'O':
                    features_list.insert(0,data[index - crawl_count][2])#insert neighbouring word (to feature)
                elif compared_entity_type.split('-')[0] == 'B':
                    features_list.insert(0,'<ne>')#insert NE tag, the specific NE isn't relevant imo
            else:
                for i in range(1, 7 - crawl_count):
                    features_list.insert(0,f'<s{i}>')
                break
                    
            if crawl_count==5:
                break
            else:
                crawl_count+=1
        
        crawl_count = 1     
        while True:
            compared_entity_type = data[index - crawl_count][4]
            if sent_num == data[index + crawl_count][1]: #if word in the same sentence
                if pos_include == True:
                    if split_pos_vs_words:
                        features_list_pos_only.append(data[index - crawl_count][3])#insert pos for neighbouring word
                    else:
                        features_list.append(data[index - crawl_count][3])#insert pos for neighbouring word
                if compared_entity_type == 'O':
                    features_list.append(data[index + crawl_count][2])
                elif compared_entity_type.split('-')[0] == 'B':
                    features_list.append('<ne>')
            else:
                for i in range(1, 7 - crawl_count):
                    features_list.append(f'<e{i}>')
                break
                
            if crawl_count==5:
                break
            else:
                crawl_count+=1
        
        
        instances.append(Instance(entity_type.split('-')[1], features_list))
        if split_pos_vs_words:
            instances_pos.append(Instance(entity_type.split('-')[1], features_list_pos_only))
        
        
    if split_pos_vs_words:
        return instances, instances_pos
    else:
        return instances

# Code for part 3
def create_table(instances, top_freq=3000):
    
    #get most frequent words by iterating over all the documents
    #each word is an key/index; each value is the count itself
    all_words_count = dict()
    for instance in instances:
        for word in instance.features:
            all_words_count[word] = all_words_count.get(word, 0) + 1
    
    #following is an ordered list of the most frequent words in all texts:
    top_freq_words = [word_freq[0] for word_freq in sorted(all_words_count.items(), key=lambda x: x[1], reverse=True)][0:top_freq]
    
    #we can now generate the matrix row by row (doc by doc)
    #each row will be a list() in all_data
    all_data = list()
    for instance in instances:
        word_counts = np.zeros(len(top_freq_words)) #scaffold the row
        for word in instance.features:
            try:
                #for each word in the doc, we try to increment the count of each word in word_counts
                #using top_freq_words as an index, will pass if a word isn't in the top 3000 words
                word_counts[top_freq_words.index(word)] += 1
            except:
                pass
        #throw the whole row in all_data once finished with the sentence, adding the filename in the first column
        all_data.append( [instance.neclass] + list(word_counts) )
    #we is done, yeeee
    return pd.DataFrame(all_data, columns= ['class'] + top_freq_words )

def ttsplit(bigdf):
    df_train, df_test = train_test_split(bigdf, test_size=0.2)
        
    return df_train.drop('class', axis=1).to_numpy(), df_train['class'], df_test.drop('class', axis=1).to_numpy(), df_test['class']

# Code for part 5

def confusion_matrix(truth, predictions):
    truth = list(truth)
    predictions = list(predictions)
    
    classes_list = list()
    for i in range(len(truth)):
        if truth[i] not in classes_list:
            classes_list.append(truth[i])
        if predictions[i] not in classes_list:
            classes_list.append(predictions[i])
    
    classes_nested_dict = dict()
    for i in classes_list:
        classes_nested_dict[i] = dict()
        for j in classes_list:
            classes_nested_dict[i][j] = 0
    
    for i in range(len(truth)):
        classes_nested_dict[truth[i]][predictions[i]] += 1
    
    truth_list = list()
    for key_truth, value_truth in classes_nested_dict.items():
        pred_list = list()
        for key_pred, value_pred in value_truth.items():
            pred_list.append(value_pred)
        truth_list.append(pred_list)
        
    df_confusion = pd.DataFrame(np.array(truth_list),columns=classes_list,index=classes_list)
    
    return df_confusion

def info_associated_w_cf(truth, predictions):
    truth = list(truth)
    predictions = list(predictions)
    confusion_dict = dict()
    for i in range(len(truth)):
        if truth[i] not in confusion_dict:
            confusion_dict[truth[i]] = { 'tp':0, 'tn':0, 'fn':0, 'fp':0 }
        if predictions[i] not in confusion_dict:
            confusion_dict[predictions[i]] = { 'tp':0, 'tn':0, 'fn':0, 'fp':0 }
    
    for key in confusion_dict.keys():
        for i in range(len(truth)):
            if (key == truth[i]) and (key == predictions[i]):
                confusion_dict[key]['tp'] += 1
            elif (key == truth[i]):
                confusion_dict[key]['fn'] += 1
            elif (key == predictions[i]):
                confusion_dict[key]['fp'] += 1
            else:
                confusion_dict[key]['tn'] += 1
                
    for key in confusion_dict.keys():
        correct_sum = confusion_dict[key]['tp'] + confusion_dict[key]['tn']
        wrong_sum = confusion_dict[key]['fp'] + confusion_dict[key]['fn']
        confusion_dict[key]['correct_percentage'] = correct_sum/(correct_sum+wrong_sum)
        confusion_dict[key]['wrong_percentage'] = wrong_sum/(correct_sum+wrong_sum)
                
    return confusion_dict

# Code for bonus part B
def bonusb(filename):
    pass

import collections
import collections
import csv
from io import open
import json
import logging
import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm, tqdm_notebook

import pandas as pd
import random


#Read anli label file
def read_lst(input_file):
    """Reads a lst file."""
    with open(input_file, "r") as file:
        lines = file.read().splitlines()
        file.close()
    return lines

#For Sampling FewShot Data
def Fewshot_sample(data_path,label_path,save_path, sample_number=100, seed=68):
    random.seed(seed)
    train_data = pd.read_json(data_path, lines=True)
    train_labels = read_lst(label_path)
    train_data['label'] = [int(label) for label in train_labels]
    
    storyid_list = train_data["story_id"].unique()
    sampled_id = random.sample(list(storyid_list), sample_number)
    sampled_data = train_data.loc[train_data["story_id"].isin(sampled_id)].set_index('story_id')
    sampled_data = sampled_data.loc[sampled_id][:sample_number]
    sampled_data = sampled_data.reset_index()
    sampled_data.to_json(save_path,orient="records")
    return sampled_data, sampled_id


def _preprocess(sample_file):
    stories = collections.defaultdict(dict)
    num_duplicate = 0
    #for sample in tqdm(get_samples(sample_file, label_file)):
    for sample in tqdm(get_samples(sample_file)):
        label = sample['label']
        if sample['hyp1'] == sample['hyp2']:
            num_duplicate += 1
            continue
        if sample['story_id'] not in stories:
            stories[sample['story_id']]['cnt'] = 1
            stories[sample['story_id']]['obs1'] = sample['obs1']
            stories[sample['story_id']]['obs2'] = sample['obs2']
            stories[sample['story_id']]['hypes'] = collections.defaultdict(lambda: [0, 0])
            stories[sample['story_id']]['hypes'][sample['hyp1']][int(label == 1)] += 1
            stories[sample['story_id']]['hypes'][sample['hyp2']][int(label == 2)] += 1
        else:
            stories[sample['story_id']]['cnt'] += 1
            assert stories[sample['story_id']]['obs1'] == sample['obs1']
            assert stories[sample['story_id']]['obs2'] == sample['obs2']
            stories[sample['story_id']]['hypes'][sample['hyp1']][int(label == 1)] += 1
            stories[sample['story_id']]['hypes'][sample['hyp2']][int(label == 2)] += 1
    examples = []
    for _id, story in stories.items():
        examples.append(StoryExample(_id, story['obs1'], story['obs2'], 
        #list(story['hypes'].keys()),[n_pos / (n_pos + n_neg) for n_neg, n_pos in story['hypes'].values()]))
        list(story['hypes'].keys()),[ (n_pos - n_neg) for n_neg, n_pos in story['hypes'].values()]))
    return stories, examples


class StoryExample(object):

    def __init__(self, _id, obs1, obs2, hypes, labels=None):
        """
        Args:
            _id (str):
            obs1 (str):
            obs2 (str):
            hypes (list[str]):
            labels (list[float]):
        """
        self._id = _id
        self.obs1 = obs1
        self.obs2 = obs2
        self.hypes = hypes
        self.hyp2idx = dict([(hyp, i) for i, hyp in enumerate(hypes)])
        self.labels = labels

    @property
    def id(self):
        return self._id

    @classmethod
    def from_dict(cls, dic):
        return cls(**dic)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return json.dumps(self.__dict__, ensure_ascii=False)


def read_jsonl(input_file):
    """Reads a jsonl file."""
    lines = []
    with open(input_file, "r") as f:
        for line in f:
            dic = json.loads(line)
            lines.append(dic)
    return lines

# def read_lst(input_file):
#     """Reads a lst file."""
#     with open(input_file, "r") as f:
#         lines = f.readlines()
#     return lines


def get_samples(sample_file):
    file = open(sample_file)
    samples= json.load(file)
    return samples

def get_hypothesis_frequence(examples):
    all_dataset=[]
    for i in range(len(examples)):
        dataset = dict()
        dataset['_id'] = examples[i]._id
        dataset['obs1'] = examples[i].obs1
        dataset['obs2'] = examples[i].obs2
        dataset['obs2'] = examples[i].obs2
        dataset['hypes'] = examples[i].hypes
        dataset['labels'] = examples[i].labels
        hyplabel = {}
        for h, lab in zip(examples[i].hypes, examples[i].labels):
            hyplabel[h] = lab
            #hyplabel_1 = sorted(hyplabel.items(), key=lambda x: x[1], reverse=True)
        dataset['hyplabel'] = hyplabel
        all_dataset.append(dataset)

    for x in range(len(all_dataset)):
        hypess = []
        for i in all_dataset[x]['hyplabel'].items():
            hyp2 = []
            for j in all_dataset[x]['hyplabel'].items():
                if i == j:
                    continue
                if i[1] > j[1]:
                    hyp2.append(j[0])
            hypess.append(hyp2)
        all_dataset[x]['hypess'] = hypess
    return all_dataset


def assign_hypothesis(all_dataset, save_path):
    train_data = get_samples(save_path)
    for num in range(len(all_dataset)):
        for ii in range(len(all_dataset[num]['hypes'])):
            if len(all_dataset[num]['hypess'][ii]) == 0:
                continue
            for xx in range(len(train_data)):
                if train_data[xx]['hyp1'] == all_dataset[num]['hypes'][ii]:
                        train_data[xx]['hyp2'] = all_dataset[num]['hypess'][ii]
                if train_data[xx]['hyp2'] == all_dataset[num]['hypes'][ii]:
                        train_data[xx]['hyp1'] = all_dataset[num]['hypess'][ii]
    return train_data

# Generate Save No Random Version of dataset
def generate_save(train_data,save_path):
    train_data_100 = pd.DataFrame.from_dict(train_data)
    train_data_100_2 = train_data_100.explode('hyp1').reset_index(drop=True)
    df_3 = train_data_100_2.explode('hyp2').reset_index(drop=True)

    #print("Length of augmentated dataset",len(df_3))
    df_3 = df_3.drop_duplicates()
    #print("After Drop Duplicates, Length of augmentated dataset",len(df_3))

    df_3_idx = range(len(df_3))
    df_3['idx'] = df_3_idx
    #df_3.to_json(save_path,orient="records")
    return df_3

# Generate Save Random Version of dataset
def generate_save_random(df_3,save_processed_path_random,seed):
    df_4 = df_3.drop(columns=['idx'])
    df_5 = df_4.sample(frac=1, random_state=seed)
    df_5_idx = range(len(df_5))
    df_5['idx'] = df_5_idx
    df_5.to_json(save_processed_path_random,orient="records")
    # return df_5

def data_read_for_train(dev_data_file,dev_label_file):
    dev_data = pd.read_json(dev_data_file, lines=True)
    dev_labels = read_lst(dev_label_file)
    dev_data['label'] = [int(label) for label in dev_labels]
    dev_data = dev_data.reset_index()
    dev_data = dev_data.rename(columns={"index":'idx'})
    dev_data.to_json("./train.json",orient="records")

def data_processing_trian(data_path,label_path,save_path, sample_number,seed):
    if sample_number < 10000:
        sampled_data, sampled_id = Fewshot_sample(data_path,label_path,save_path, sample_number,seed)
        # Perform the data_precessing part
        stories, examples = _preprocess(save_path)
        all_dataset = get_hypothesis_frequence(examples)
        train_data = assign_hypothesis(all_dataset, save_path)
        
        # Convert to dataframe for further processing 
        # Non-Random Version of dataset
        save_processed_path = "./train.json"
        save_processed_path_random = "./train.json"
        df_3 = generate_save(train_data,save_processed_path)

        #Random Version of dataset
        generate_save_random(df_3,save_processed_path_random,seed)
    else:
        data_read_for_train(data_path,label_path)
        # Perform the data_precessing part
        stories, examples = _preprocess(save_path)
        all_dataset = get_hypothesis_frequence(examples)
        train_data = assign_hypothesis(all_dataset, save_path)
        
        # Convert to dataframe for further processing 
        # Non-Random Version of dataset
        save_processed_path = "./train.json"
        save_processed_path_random = "./train.json"
        df_3 = generate_save(train_data,save_processed_path)

        #Random Version of dataset
        generate_save_random(df_3,save_processed_path_random,seed)

def data_processing_dev(dev_data_file,dev_label_file):
    dev_data = pd.read_json(dev_data_file, lines=True)
    dev_labels = read_lst(dev_label_file)
    dev_data['label'] = [int(label) for label in dev_labels]
    dev_data = dev_data.reset_index()
    dev_data = dev_data.rename(columns={"index":'idx'})
    dev_data.to_json("./dev32.json",orient="records")

def data_processing_test(test_data_file,test_label_file):
    test_data = pd.read_json(test_data_file, lines=True)
    test_labels = read_lst(test_label_file)
    test_data['label'] = [int(label) for label in test_labels]
    test_data = test_data.reset_index()
    test_data = test_data.rename(columns={"index":'idx'})
    test_data.to_json("./val.json",orient="records")

if __name__ == "__main__":
    #Processing training/Sampling FewShot Data
    train_data_path = "./anli/train.jsonl"
    train_label_path= './anli/train-labels.lst'
    save_path= "./Preprocess_instance.json"
    sample_number = 100
    seed = 72
    print("Seed Number :", seed)
    
    # sampled_id is the sampled story id
    sampled_data, sampled_id = Fewshot_sample(train_data_path,train_label_path,save_path, sample_number,seed)

    # Perform the data_precessing part
    stories, examples = _preprocess(save_path)
    all_dataset = get_hypothesis_frequence(examples)
    train_data = assign_hypothesis(all_dataset, save_path)

    
    # Convert to dataframe for further processing 
    # No Random Version of dataset
    save_processed_path = "./Seed72_100instance.json"
    save_processed_path_random = "./Seed72_100instance_random.json"
    print("Start NoRandom Version of dataset")
    df_3 = generate_save(train_data,save_processed_path)

    #Random Version of dataset
    print("Start Random Version of dataset")
    generate_save_random(df_3,save_processed_path_random)


    ### Procedure ###
    # 1. save_path                  "./Seed42_100instances.json"
    # 2. seed                       42
    # 3. save_processed_path        "Seed42_100instance_extentrows.json"
    # 4. save_processed_path_random "Seed42_100instance_extentrows_random.json"
    #if need, change sample number






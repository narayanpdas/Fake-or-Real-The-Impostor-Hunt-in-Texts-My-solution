import pandas as pd
import numpy as np
import os 
# Note: This is the File Path If you import the data in your Notebook
#       If you downloaded the data Provid the Local Paths
path_train = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train'
path_test = '/kaggle/input/fake-or-real-the-impostor-hunt/data/test'
train_labels_path = '/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv'
train_labels = pd.read_csv(train_labels_path)
def make_data_csv(path):
    df = []
    for articles in os.listdir(path):
        data_point = []
        data_point.append(articles)
        for text in os.listdir(path+'/'+articles):
            with open(path+'/'+articles+'/'+text,'r') as file:
                contents = file.read()
                data_point.append(contents)
                file.close()
        df.append(data_point)
    df = pd.DataFrame(df)
    df.columns = ['article_number','text_2','text_1']
    df['id'] = df['article_number'].str.split('_').str[1].astype(int)
    df = df.sort_values(by='id').reset_index(drop=True)
    return df
train_df = make_data_csv(path_train)
test_df = make_data_csv(path_test)
train_df['labels'] = train_labels['real_text_id']
train_df.to_csv('train_dataset.csv')
test_df.to_csv('test_data.csv')
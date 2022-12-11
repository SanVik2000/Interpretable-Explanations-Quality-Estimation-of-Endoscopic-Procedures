import os
import glob
import shutil
import pandas as pd
from tqdm import tqdm

def get_label_from_file(file_path):
    f = open(file_path, "r")
    file_contents = f.read()
    if 'correct' in file_contents:
        return 1
    else:
        return 0 

def read_folder():
    video_label_list = []
    file_name_list = []
    
    root_path = '/media/sanvik/Data/Dual_Degree_Project/full/'
    subfolders = [ f.path for f in os.scandir(root_path) if f.is_dir() ]
    for item in tqdm(subfolders):
        label = get_label_from_file(os.path.join(root_path, item, 'label.txt'))
        video_label_list.append(label)
        file_name_list.append(os.path.join(root_path, item))

    df = pd.DataFrame(columns=['File', 'Label'])
    for x,y in zip(file_name_list, video_label_list):
        df = df.append({'File' : x, 'Label' : y}, ignore_index = True)
    print(df)
    df.to_csv("dataset_metadata.csv")

def process(df, phase):
    dest_data_dir = f"/media/sanvik/Data/Dual_Degree_Project/{phase}/"
    for i in range(len(df)):
        file = df['File'].iloc[i]
        trial_name = file.split('/')[-1]
        shutil.copytree(file, os.path.join(dest_data_dir, trial_name))


def main():
    df = pd.read_csv("dataset_metadata.csv")
    
    negative_df = df.loc[df['Label'] == 0]
    negative_train_df, negative_val_df = negative_df[:15], negative_df[15:]
    positive_df = df.loc[df['Label'] == 1]
    positive_train_df , positive_val_df = positive_df[:60], positive_df[60:]

    print("Total Positive : " , len(positive_df))
    print("Positive Train : " , len(positive_train_df))
    print("Positive Val : " , len(positive_val_df))
    print("Total Negative : " , len(negative_df))
    print("Negative Train : " , len(negative_train_df))
    print("Negative Val : " , len(negative_val_df))

    print(positive_train_df)

    #process(positive_train_df, 'train')
    #process(negative_train_df, 'train')
    #process(positive_val_df, 'validation')
    #process(negative_val_df, 'validation')




if __name__ == "__main__":
    main()
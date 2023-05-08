import json
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import drive

drive.mount('/content/drive')

class WikiartDataset(Dataset):  # WikiartDataset class inherits from the PyTorch Dataset class, which is a base class for all datasets in PyTorch
    def __init__(self, dataset_path):

        self.image_path = os.path.join(dataset_path, "image")  # self.image_path is one "instance variable" since self refers to an instance of WikiartDataset class
        self.music_path = os.path.join(dataset_path, "music")
        self.meta_path = os.path.join(dataset_path, "meta")
        self.decades = [str(i) for i in range(1480, 2010, 10)]
        self.data = {'image': {}, 'music': {}, 'meta': {}}

        for path, type in zip([self.image_path, self.music_path], ['image', 'music']): # [(self.image_path, 'image'), (self.music_path, 'music')]
            for decade in self.decades:
                decade_path = os.path.join(path, decade)
                self.data[type][decade] = os.listdir(decade_path) # returns a list of all files in this decade directory, which will be either a list of string image filenames or a list of string music filenames
        
        for file_name in os.listdir(self.meta_path):
            file_path = os.path.join(self.meta_path, file_name)
            with open(file_path, 'r') as f:
                metadata = json.load(f)
            self.data['meta'][file_name] = metadata

dataset_path = "/content/drive/MyDrive/WikiArt-IMSLP Dataset"
dataset = WikiartDataset(dataset_path)





"""
    def __getitem__(self, idx):
        # Load image data
        img_file = os.path.join(self.image_folder, f"{idx}.jpg")
        img = Image.open(img_file)

        # Load music data
        music_folder = os.path.join(self.music_folder, "1850")
        music_file = os.path.join(music_folder, f"{idx}.mp3")
        y, sr = librosa.load(music_file)

        # Return the image and music data as a dictionary
        sample = {
            "image": img,
            "music": y,
            "sr": sr
        }
        return sample

"""

from _train import *
from _test import *
from google.colab import drive

drive.mount('/content/drive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

process = input("What would you like to do? (train/test): ")

if process == "train":
    train(device)
elif process == "test":
    test(device)
else:
    print("Invalid choice. Please select 'train' or 'test'.")

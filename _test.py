import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from _models import *
from _utils import *

def test_generator(device, data, enc_dim=10, batch_size=1, noise_dim=100):
    counter, loss = 0, 0
    conditional_noise_dim = noise_dim + enc_dim
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
    images = {"renaissance": [], "baroque": [], "classical": [], "romantic": [], "modern": []}

    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load("/content/drive/MyDrive/MODELS/encoder.pth"))
    encoder.eval()

    generator = Generator(in_dim=conditional_noise_dim).to(device)
    generator.load_state_dict(torch.load("/content/drive/MyDrive/MODELS/generator.pth", map_location=torch.device('cpu')))
    generator.eval()

    #classifier = Classifier().to(device)
    #classifier.load_state_dict(torch.load("/content/drive/MyDrive/MODELS/classifier.pth"))
    #classifier.eval()

    pbar = tqdm(data_loader, desc="Sample {}".format(counter), total=len(data_loader))
    for _, dl in enumerate(pbar):
        counter += 1
        melspecs, eras, _ = dl
        melspecs = torch.transpose(torch.stack(melspecs, dim=0), 0, 1)
        noise = torch.randn(batch_size, noise_dim).to(device)
        actual_eras = torch.tensor([to_numerical(era) for era in eras])
        encoding = encoder(melspecs.to(device))
        input = torch.cat((noise, encoding), dim=1).to(device)

        SI = generator(input).to(device)

        #CSI = classifier(SI).to(device)
        #loss += 100 * nll_loss(CSI, actual_eras.to(device))
        pbar.set_postfix({'Classifier Loss': '{0:.3f}'.format(loss / (counter + 1))})

        [images[era].append(si.detach().cpu().numpy()) for si, era in zip(SI, eras)]
    save_collage(images, size=(6,6))

def test(device):
    dataset_path = "/content/drive/MyDrive/DATASETS/musart-dataset/testing"
    art_eras = ["renaissance", "baroque", "classical", "romantic", "modern"]
    prepared = input("<testing_data.npy> already exists in ~/content/drive/MyDrive/DATA: ")
    
    if prepared == "n":
        data = prepare_data(dataset_path, art_eras, sr=22050, n_fft=1024, hop_length=256, n_mels=128, display=False)
        np.save('/content/drive/MyDrive/DATA/testing_data.npy', np.array(data))
    elif prepared == "y":
        data = np.load('/content/drive/MyDrive/DATA/testing_data.npy', allow_pickle=True)
        data = data.tolist()
    else:
        print("Invalid choice. Please select 'y' or 'n'.")

    test_generator(device, data, enc_dim=10, batch_size=1, noise_dim=100)

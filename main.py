import torch
import os
import random
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from google.colab import drive
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import librosa
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torch.nn import LogSoftmax
#from models import *

# UTILS ###############################################################################################################################

def show_spectrogram(name, ms, sr, hop_length):
    ms = librosa.power_to_db(ms, ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(ms, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel-frequency spectrogram - {name}")
    plt.xlabel('Time')
    plt.ylabel('Hz')
    plt.show()

def get_mel_spectrogram(audio_path, sr, n_fft, hop_length, n_mels, display):
    spectrograms = []
    waveform, _ = librosa.load(audio_path, sr=sr)
    audio_name = os.path.basename(audio_path)
    for i in range(3):
        segment = waveform[int(i * 2.97 * sr):int((i + 1) * 2.97 * sr)]
        spectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        show_spectrogram(f"{audio_name} (Segment {i+1})", spectrogram, sr, hop_length) if display else None
        spectrograms.append(spectrogram)
    return spectrograms

def prepare_data(dataset_path, eras, sr, n_fft, hop_length, n_mels, display):
    data, transform = [], transforms.ToTensor()
    eras_paths = [os.path.join(dataset_path, era) for era in eras]
    for era, path in zip(eras, eras_paths):
        audios = sorted([file for file in os.listdir(path) if file.endswith('.mp3')], key=lambda x: int(x[1:].split('.')[0]))
        for audio in audios:
            audio_path = os.path.join(path, audio)
            ms = get_mel_spectrogram(audio_path, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, display=display)
            id = int(audio.split(".")[0][1:])
            pair_names = ["i" + str(id + len(audios) * i) + ".jpg" for i in range(3)]
            pairs = [transform(Image.open(os.path.join(path, pair))) for pair in pair_names]
            data.extend([(ms, era, pair) for pair in pairs])
    return data

def triplet_margin_loss(anchor, positive, negative):
    TML = nn.TripletMarginLoss(margin=1.0, p=2)
    loss = TML(anchor, positive, negative)
    return loss

def feature_matching_loss(pair_features, style_features):
    criterion = nn.MSELoss()
    return criterion(pair_features, style_features)

def nll_loss(output, target):
    nll = nn.NLLLoss()
    log_softmax = LogSoftmax(dim=1)
    loss = nll(log_softmax(output), target)
    return loss

def gram_matrix(input):
    batch_size, num_channels, height, width = input.size()
    features = input.view(batch_size * num_channels, height * width)
    gram = torch.mm(features, features.t())
    norm_gram = gram.div(batch_size * num_channels * height * width)
    return norm_gram

def style_loss(model, style_image, pair_image):
    style_features, pair_features = model(style_image), model(pair_image)
    loss_module = nn.MSELoss().to(device)
    style_gram = gram_matrix(style_features)
    pair_gram = gram_matrix(pair_features)
    loss = loss_module(style_gram, pair_gram)
    return loss

def show_image(image):
    image = image.detach().cpu().numpy()
    image = image.transpose(1, 2, 0)  # transposes the dimensions to match the image format (H, W, C)
    plt.imshow(image) # scales the values in the tensor to the appropriate color range and displays it as an image
    plt.axis('off')
    plt.show()

def to_numerical(categorical):
    if categorical == "renaissance":
        return 0
    elif categorical == "baroque":
        return 1
    elif categorical == "classical":
        return 2
    elif categorical == "romantic":
        return 3
    elif categorical == "modern":
        return 4

# MAIN FUNCTION #######################################################################################################################

drive.mount('/content/drive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = "/content/drive/MyDrive/DATASETS/musart-dataset"
eras = ["renaissance", "baroque", "classical", "romantic", "modern"]
data = prepare_data(dataset_path, eras, sr=22050, n_fft=1024, hop_length=256, n_mels=128, display=False) # a list of 900*5=4500 elements where each element is a tuple (specs, era, pair) & data[i][0] = i-th specs array is a list of 3 spectrograms

BATCH_SIZE = 15 # number of samples in each batch
data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=15) # len(data_loader) = num_batches = 900*5 // 15 = 300
batches = list(data_loader) # a list of 300 lists\batches

# 1) TRAIN THE ENCODER ################################################################################################################

learning_rate = 0.0001
num_epochs = 60

encoder = Encoder().to(device)
encoder.load_state_dict(torch.load("/content/drive/MyDrive/MODELS/encoder.pth", map_location=torch.device('cpu')))
enc_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=0.0001)
encoder.train()

for epoch in range(num_epochs):
    enc_epoch_loss, steps = 0, 0
    pbar = tqdm(data_loader, desc="Training", total=len(data_loader))
    for i, dl in enumerate(pbar): # dl[0]: the 3 spectrogram-batches of 15 specs each, dl[1]: the 15 era-labels, dl[2]: the 15 pair-images
        melspecs, eras, pairs = dl
        melspecs = torch.transpose(torch.stack(melspecs, dim=0), 0, 1) # torch.Size([15, 3, 128, 256])
        
        pos_melspecs, neg_melspecs = [], []
        for sample in range(BATCH_SIZE):
            pos, neg = False, False

            while not pos or not neg:
                rbID = random.randint(0, len(data_loader) - 1)
                rand_melspecs, rand_eras = torch.transpose(torch.stack(batches[rbID][0], dim=0), 0, 1), batches[rbID][1] # torch.Size([15, 3, 128, 256]) & tuple of strings
                rsID = random.randint(0, BATCH_SIZE - 1)
                rand_melspec, rand_era = rand_melspecs[rsID], rand_eras[rsID] # torch.Size([3, 128, 256]) & string

                if not pos and rand_era == eras[sample] and not torch.all(torch.eq(rand_melspec, melspecs[rsID])):
                    pos_melspecs.append(rand_melspec)
                    pos = True
                elif not neg and rand_era != eras[sample]:
                    neg_melspecs.append(rand_melspec)
                    neg = True

        pos_melspecs, neg_melspecs = torch.stack(pos_melspecs, dim=0), torch.stack(neg_melspecs, dim=0)
        EA, EPA, ENA = encoder(melspecs.to(device)), encoder(pos_melspecs.to(device)), encoder(neg_melspecs.to(device))

        enc_optimizer.zero_grad()
        enc_loss = triplet_margin_loss(EA, EPA, ENA)
        enc_loss.backward()
        enc_optimizer.step()

        enc_epoch_loss += enc_loss.item()
        steps += 1
        pbar.set_postfix({'Classifier Loss': '{0:.3f}'.format(enc_epoch_loss/steps)})

        torch.save(encoder.state_dict(), "/content/drive/MyDrive/MODELS/encoder.pth")

# 2) TRAIN THE CLASSIFIER #################################################################################################################

learning_rate = 0.00005
num_epochs = 2

classifier = Classifier().to(device)
classifier.load_state_dict(torch.load("/content/drive/MyDrive/MODELS/classifier.pth", map_location=torch.device('cpu')))
clf_optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=0.0001)
classifier.train()

for epoch in range(num_epochs):
    clf_epoch_loss, steps = 0, 0
    pbar = tqdm(data_loader, desc="Training", total=len(data_loader))
    for i, dl in enumerate(pbar): # dl[0]: the 3 spectrogram-batches of 15 specs each, dl[1]: the 15 era-labels, dl[2]: the 15 pair-images
        melspecs, eras, pairs = dl
        PI = torch.stack([pair for pair in pairs])

        for img in PI:
            print(img)
            show_image(img)

        actual_eras, CPI = torch.tensor([to_numerical(era) for era in eras]).to(device), classifier(PI.to(device))
        clf_optimizer.zero_grad()
        clf_loss = nll_loss(CPI, actual_eras)
        clf_loss.backward()
        clf_optimizer.step()

        clf_epoch_loss += clf_loss.item()
        steps += 1
        pbar.set_postfix({'Classifier Loss': '{0:.3f}'.format(clf_epoch_loss/steps)})

        torch.save(classifier.state_dict(), "/content/drive/MyDrive/MODELS/classifier.pth")

# TODO: 3) TRAIN THE GAN #####################################################################################################################

learning_rate = 0.0005
num_epochs = 15
vgg = VGG19(pretrained=True, require_grad=False).to(device)

generator = Generator().to(device)
generator.load_state_dict(torch.load("/content/drive/MyDrive/MODELS/generator.pth", map_location=torch.device('cpu')))
gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
generator.train()

discriminator = Discriminator().to(device)
discriminator.load_state_dict(torch.load("/content/drive/MyDrive/MODELS/discriminator.pth", map_location=torch.device('cpu')))
dis_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
discriminator.train()

for epoch in range(num_epochs):
    gen_epoch_loss, dis_epoch_loss, steps = 0, 0, 0
    pbar = tqdm(data_loader, desc="Training", total=len(data_loader))
    for i, dl in enumerate(pbar):
        melspecs, eras, pairs = dl
        melspecs = torch.transpose(torch.stack(melspecs, dim=0), 0, 1) # torch.Size([15, 3, 128, 256])
        EA = torch.cat((encoder(melspecs.to(device)), torch.randn(BATCH_SIZE, 256, 8, 8).to(device)), dim=1)
        SI = generator(EA.to(device))
        PI = torch.stack([pair for pair in pairs])
        actual_eras, CSI = torch.tensor([to_numerical(era) for era in eras]).to(device), classifier(SI.to(device))
        DPI = discriminator(PI.to(device))
        DSI = discriminator(SI.to(device))

        gen_optimizer.zero_grad()
        gen_loss = 10*feature_matching_loss(DPI, DSI) + 10*nll_loss(CSI, actual_eras) + style_loss(vgg, SI.to(device), PI.to(device))
        gen_loss.backward()
        gen_optimizer.step()
        gen_epoch_loss += gen_loss.item()
        torch.save(generator.state_dict(), "/content/drive/MyDrive/MODELS/generator.pth")

        dis_optimizer.zero_grad()
        DPI, DSI = DPI.detach(), DSI.detach()
        DPI.requires_grad_(); DSI.requires_grad_()
        dis_loss = - feature_matching_loss(DPI, DSI)
        dis_loss.backward()
        dis_optimizer.step()
        dis_epoch_loss += dis_loss.item()
        torch.save(discriminator.state_dict(), "/content/drive/MyDrive/MODELS/discriminator.pth")

        steps += 1
        pbar.set_postfix({'Generator Loss': '{0:.3f}'.format(gen_epoch_loss / steps),
                          'Discriminator Loss': '{0:.3f}'.format(dis_epoch_loss / steps)})

        """for image in SI:
            show_image(image)"""

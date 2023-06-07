import torch
import os
import random
import torch.nn as nn
import torch.optim as optim
from google.colab import drive
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import librosa
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torch.nn import LogSoftmax

from models import Encoder, Classifier, Generator, Discriminator

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
        spectrograms.append(spectrogram.numpy())
    return spectrograms

def prepare_data(dataset_path, eras, sr, n_fft, hop_length, n_mels, display):
    data = []
    eras_paths = [os.path.join(dataset_path, era) for era in eras]
    for era, path in zip(eras, eras_paths):
        audios = sorted([file for file in os.listdir(path) if file.endswith('.mp3')], key=lambda x: int(x[1:].split('.')[0]))
        for audio in audios:
            audio_path = os.path.join(path, audio)
            ms = get_mel_spectrogram(audio_path, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, display=display)
            id = int(audio.split(".")[0][1:])
            pairs = ["i" + str(id + len(audios) * i) + ".jpg" for i in range(3)]
            data.extend([(ms, era, pair) for pair in pairs])
    return data

def triplet_margin_loss(anchor, positive, negative):
    TML = nn.TripletMarginLoss(margin=1.0, p=2)
    loss = TML(anchor, positive, negative)
    return loss

#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#**#

def show_image(image):
    image = image.squeeze(0).detach().cpu().numpy()
    image = image.transpose(1, 2, 0)  # Transpose the dimensions to match the image format (H, W, C)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def display_layers(vgg, type):
    for idx, module in enumerate(vgg):
        if isinstance(module, type):
            print(f'Layer index: {idx}, Layer name: {module}')

def get_features(image, model, layers):
    features = {}
    for name, layer in model._modules.items(): # name: "0", "5", etc.
        image = layer(image)
        if name in layers:
            features[layers[name]] = image
    return features

def gram_matrix(layer_features):
    _, c, h, w = layer_features.size()
    layer_features = layer_features.view(c, h * w)
    gram = torch.mm(layer_features, layer_features.t())
    return gram

def style_loss(model, style_image, pair_image):
    style_loss = 0
    loss_module = nn.MSELoss().to(device)
    layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '28': 'conv5_1'}
    style_features = get_features(style_image, model, layers)
    pair_features = get_features(pair_image, model, layers)
    for layer in pair_features:
        style_gram = gram_matrix(pair_features[layer])
        pair_gram = gram_matrix(style_features[layer])
        layer_loss = loss_module(style_gram, pair_gram)
        style_loss += layer_loss
    return style_loss / len(pair_features)

def feature_matching_loss(pair_features, style_features):
    criterion = nn.MSELoss()
    return criterion(pair_features, style_features)

def nll_loss(output, target):
    nll = nn.NLLLoss()
    log_softmax = LogSoftmax(dim=1)
    loss = nll(log_softmax(output), target)
    return loss

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

dataset_path = "/content/drive/MyDrive/DATASETS/new_tester"
eras = ["renaissance", "baroque", "classical", "romantic", "modern"]
data = prepare_data(dataset_path, eras, sr=22050, n_fft=1024, hop_length=256, n_mels=128, display=False) # a list of 900*5=4500 elements where each element is a tuple (specs, era, pair) & data[i][0] = i-th specs array is a list of 3 spectrograms

BATCHES = 15
transform = transforms.ToTensor()
data_loader = DataLoader(data, batch_size=BATCHES, shuffle=True) # len(data_loader) = 900*5 // 15 = 300
batches = list(data_loader) # a list of 300 lists\batches

# 1) TRAIN THE ENCODER ################################################################################################################

learning_rate = 0.0001
num_epochs = 2

encoder = Encoder().to(device)
enc_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=0.0001)

for epoch in range(num_epochs):
    for dl in data_loader: # dl[0]: the 3 spectrogram-batches of 15 specs each, dl[1]: the 15 era-labels, dl[2]: the 15 pair-images
        melspecs, eras, pairs = dl
        melspecs = torch.transpose(torch.stack(melspecs, dim=0), 0, 1) # torch.Size([15, 3, 128, 256])
        
        pos_melspecs, neg_melspecs = [], []
        for sample in range(BATCHES):
            pos, neg = False, False

            while not pos or not neg:
                rbID = random.randint(0, len(data_loader) - 1)
                rand_melspecs, rand_eras = torch.transpose(torch.stack(batches[rbID][0], dim=0), 0, 1), batches[rbID][1] # torch.Size([15, 3, 128, 256]) & tuple of strings
                rsID = random.randint(0, BATCHES - 1)
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
        loss = triplet_margin_loss(EA, EPA, ENA)
        loss.backward()
        enc_optimizer.step()

torch.save(encoder.state_dict(), "/content/drive/MyDrive/MODELS/encoder.pth")

"""
for epoch in range(num_epochs):
    for dl in data_loader:
        melspecs, eras, pairs = dl
        melspecs = torch.transpose(torch.stack(melspecs, dim=0), 0, 1)
        encoded_audio = encoder(melspecs.to(device))
        random_index = random.randint(0, BATCHES - 1)
        encoded_audio_slice = encoded_audio[random_index, 0].detach().cpu().numpy()
        show_spectrogram(str(random_index), encoded_audio_slice, sr=22050, hop_length=256)
"""

# 2) TRAIN THE CLASSIFIER #################################################################################################################

learning_rate = 0.00005
num_epochs = 2

classifier = Classifier().to(device)
clf_optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=0.0001)

for epoch in range(num_epochs):
    for dl in data_loader: # dl[0]: the 3 spectrogram-batches of 15 specs each, dl[1]: the 15 era-labels, dl[2]: the 15 pair-images
        melspecs, eras, pairs = dl
        images = [Image.open(os.path.join(dataset_path, era, pair)) for era, pair in zip(eras, pairs)]
        PI = torch.stack([transform(img) for img in images])
        CSI, CPI = torch.tensor([to_numerical(era) for era in eras]).to(device), classifier(PI.to(device))
        clf_optimizer.zero_grad()
        clf_loss = nll_loss(CPI, CSI)
        clf_loss.backward()
        clf_optimizer.step()

torch.save(classifier.state_dict(), "/content/drive/MyDrive/MODELS/classifier.pth")

"""
random_image = torch.randn(1, 3, 64, 64).to(device)
print(classifier(random_image))

torch.set_printoptions(sci_mode=False)
random_image = transform(Image.open("/content/i54.jpg")).unsqueeze(0).to(device)
print(classifier(random_image))
"""

# TODO: 3) TRAIN THE GAN #####################################################################################################################

learning_rate = 0.0001
num_epochs = 2

generator = Generator().to(device)
discriminator = Discriminator().to(device)

gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
dis_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

vgg = models.vgg19(pretrained=True).features.to(device).eval()
# display_layers(vgg, nn.Conv2d)

for epoch in range(num_epochs):
    for ms, era, pair in data:
        ms = torch.tensor(ms).unsqueeze(0).expand(batch_size, in_channels, -1, -1).to(device)
        ms = torch.cat((encoder(ms), torch.randn(1, 256, 8, 8).to(device)), dim=1)
        PI = transform(Image.open(os.path.join(dataset_path, era, pair))).unsqueeze(0).to(device)
        SI = (generator(ms) + 1) / 2
        PF = discriminator(PI)
        SF = discriminator(SI)

        gen_optimizer.zero_grad()
        dis_optimizer.zero_grad()
        
        gen_loss = feature_matching_loss(PF,SF) + \
        nll_loss(classifier(SI), torch.tensor([to_numerical(era)]).to(device)) + \
        style_loss(vgg,SI,PI)

        show_image(SI)

        gen_loss.backward()
        gen_optimizer.step()

        PF, SF = PF.detach(), SF.detach()
        PF.requires_grad_(); SF.requires_grad_()

        dis_loss = - feature_matching_loss(PF,SF)
        dis_loss.backward()
        dis_optimizer.step()

show_image(SI)
show_image(PI)

print("RANDOMMMM")
random_input = (generator(torch.randn(1, 257, 8, 8).to(device)) + 1) / 2
show_image(random_input)

torch.save(generator.state_dict(), "/content/drive/MyDrive/MODELS/generator.pth")
torch.save(discriminator.state_dict(), "/content/drive/MyDrive/MODELS/discriminator.pth")

"""
third_ms = torch.tensor(data[3][0]).unsqueeze(0).expand(batch_size, in_channels, -1, -1).to(device)
encoded_audio = encoder(third_ms)
ms = torch.cat((encoded_audio, torch.randn(1, 256, 8, 8).to(device)), dim=1)

image = generator(ms)
show_image(image)
"""

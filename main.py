import torch
import os
import random
import torch.nn as nn
import torch.optim as optim
from google.colab import drive
import matplotlib.pyplot as plt
import librosa
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

from models import Encoder, Generator

# UTILS ###############################################################################################################################

def show_spectrogram(name, ms, sr=22050, hl=256):
    ms = librosa.power_to_db(ms, ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(ms, sr=sr, hop_length=hl, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel-frequency spectrogram - {name}")
    plt.xlabel('Time')
    plt.ylabel('Hz')
    plt.show()

def compute_mel_spectrogram(audio_path, display):
    waveform, sr = librosa.load(audio_path, sr=22050)
    audio_name = os.path.basename(audio_path)
    spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    show_spectrogram(audio_name, spectrogram, sr, 256) if display else None
    return spectrogram

def triplet_margin_loss(anchor, positive, negative):
    TML = nn.TripletMarginLoss(margin=1.0, p=2)
    loss = TML(anchor, positive, negative)
    return loss

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

def content_loss(model, style_image, pair_image):
    content_loss = 0
    loss_module = nn.MSELoss().to(device)
    layers = {'21': 'conv4_2'}
    content_features = get_features(style_image, model, layers)
    pair_features = get_features(pair_image, model, layers)
    for layer in pair_features:
        content_loss += loss_module(pair_features[layer], content_features[layer])
    return content_loss / len(pair_features)

def mse_loss(style_image, pair_image):
    loss_module = nn.MSELoss()
    loss = loss_module(style_image, pair_image)
    return loss

# MAIN FUNCTION #######################################################################################################################

drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_path = "/content/drive/MyDrive/DATASETS/new_tester"
eras = ["renaissance", "baroque", "classical", "romantic", "modern"]
eras_paths = [os.path.join(dataset_path, folder) for folder in eras]

mel_spectrograms = []
for era, path in zip(eras, eras_paths):
    for file in os.listdir(path):
        if file.endswith('.mp3'):
            audio_path = os.path.join(path, file)
            ms = compute_mel_spectrogram(audio_path, False)
            pair = "i" + file.split(".")[0][1:] + ".jpg"
            mel_spectrograms.append((ms, era, pair))

# 1) TRAIN THE ENCODER ################################################################################################################

learning_rate = 0.0001
num_epochs = 2
batch_size = 1
in_channels = 1

encoder = Encoder()
encoder = encoder.to(device)

optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for ms, era, pair in mel_spectrograms:
        ms = torch.tensor(ms).unsqueeze(0).expand(batch_size, in_channels, -1, -1).to(device)
        optimizer.zero_grad()
        pos_ms = random.choice([torch.tensor(m).unsqueeze(0).expand(batch_size, in_channels, -1, -1).to(device) for m, e, p in mel_spectrograms if e == era and m != ms])
        neg_ms = random.choice([torch.tensor(m).unsqueeze(0).expand(batch_size, in_channels, -1, -1).to(device) for m, e, p in mel_spectrograms if e != era])
        loss = triplet_margin_loss(encoder(ms), encoder(pos_ms), encoder(neg_ms)) # forward pass & loss calculation
        loss.backward()
        optimizer.step()
        #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

torch.save(encoder.state_dict(), "/content/drive/MyDrive/MODELS/encoder.pth")

# 2) CHECK THE ENCODER'S OUTPUT #########################################################################################################

encoder = Encoder()
encoder.load_state_dict(torch.load('/content/drive/MyDrive/MODELS/encoder.pth'))
encoder = encoder.to(device)

third_ms = torch.tensor(mel_spectrograms[3][0]).unsqueeze(0).expand(batch_size, in_channels, -1, -1).to(device) # 3rd element of the tuple at index 0
encoded_audio = encoder(third_ms)
encoded_audio = encoded_audio[0, 0].cpu().detach().numpy()
show_spectrogram("third", encoded_audio)

# 3) CHECK THE GENERATOR'S OUTPUT #########################################################################################################

learning_rate = 0.0001
num_epochs = 2
batch_size = 1
in_channels = 1

generator = Generator().to(device)

third_ms = torch.tensor(mel_spectrograms[3][0]).unsqueeze(0).expand(batch_size, in_channels, -1, -1).to(device)
encoded_audio = encoder(third_ms)
ms = torch.cat((encoded_audio, torch.randn(1, 256, 8, 8).to(device)), dim=1)

image = generator(ms)

show_image(image)

# 4) TRAIN THE GAN #####################################################################################################################

learning_rate = 0.0001
num_epochs = 2
batch_size = 1
in_channels = 1

generator = Generator().to(device)
transform = transforms.ToTensor()

optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

vgg = models.vgg19(pretrained=True).features.to(device).eval()
# display_layers(vgg, nn.Conv2d)

for epoch in range(num_epochs):
    for ms, era, pair in mel_spectrograms:
        ms = torch.tensor(ms).unsqueeze(0).expand(batch_size, in_channels, -1, -1).to(device)
        SA = torch.cat((encoder(ms), torch.randn(1, 256, 8, 8).to(device)), dim=1)
        PI = transform(Image.open(os.path.join(dataset_path, era, pair))).unsqueeze(0).to(device)
        SI = (generator(SA) + 1) / 2

        #show_image(SI)
        #show_image(PI)

        optimizer.zero_grad()
        
        loss = mse_loss(SI,PI)
        print(loss)

        #loss = 0*style_loss(vgg, SI, PI) + 0*content_loss(vgg, SI, PI) + 0*mse_loss(SI, PI) + vgg_loss(vgg, SI, PI)
        loss.backward()
        optimizer.step()

show_image(SI)
show_image(PI)

random_input = (generator(torch.randn(1, 257, 8, 8).to(device)) + 1) / 2
show_image(random_input)

torch.save(generator.state_dict(), "/content/drive/MyDrive/MODELS/generator.pth")

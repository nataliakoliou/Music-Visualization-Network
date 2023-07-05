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
import torch.nn.functional as F
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
            pairs = [Image.open(os.path.join(path, pair)) for pair in pair_names]
            data.extend([(ms, era, transform(pair)) for pair in pairs])
            mirrored = [pair.transpose(Image.FLIP_LEFT_RIGHT) for pair in pairs]
            data.extend([(ms, era, transform(mir)) for mir in mirrored])
    return data

def triplet_margin_loss(anchor, positive, negative):
    TML = nn.TripletMarginLoss(margin=1.0, p=5)
    loss = TML(anchor, positive, negative)
    return loss

def nll_loss(output, target):
    nll = nn.NLLLoss()
    log_softmax = LogSoftmax(dim=1)
    loss = nll(log_softmax(output), target)
    return loss

def stGen_loss(fake_logits, fake_output):
    real_labels = torch.ones_like(fake_output)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(fake_logits, real_labels)
    return loss

def stDis_loss(fake_logits, fake_output, real_logits, real_output, smooth=0.2):
    fake_labels = torch.zeros_like(fake_output) + smooth
    real_labels = torch.ones_like(real_output) * (1 - smooth)
    criterion = nn.BCEWithLogitsLoss()
    fake_loss = criterion(fake_logits, fake_labels)
    real_loss = criterion(real_logits, real_labels)
    loss = (fake_loss + real_loss)/2
    return loss

def mse_loss(predicted, target):
    criterion = nn.MSELoss()
    loss = criterion(predicted, target)
    return loss

def l1_loss(predicted, target):
    criterion = nn.L1Loss()
    loss = criterion(predicted, target)
    return loss

def transpose_image(image, range_min=0, range_max=1):
    min_value, max_value = image.min(), image.max()
    scaled_image = (image - min_value) / (max_value - min_value)
    transposed_image = (scaled_image * (range_max - range_min)) + range_min
    return transposed_image

def gram(x):
    c, _, _ = x.size()
    x = x.view(c, -1)
    return torch.mm(x, x.t())

def style_loss(style_features, pair_features):
    loss, total_samples = 0, len(pair_features)
    for sample in range(len(style_features)):
        style_gram = gram(style_features[sample])
        pair_gram = gram(pair_features[sample])
        loss += F.mse_loss(style_gram, pair_gram)
    return loss / total_samples

def xavier_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)

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
art_eras = ["renaissance", "baroque", "classical", "romantic", "modern"]
data = prepare_data(dataset_path, art_eras, sr=22050, n_fft=1024, hop_length=256, n_mels=128, display=False) # a list of 900*5=4500 elements where each element is a tuple (specs, era, pair) & data[i][0] = i-th specs array is a list of 3 spectrograms

BATCH_SIZE = 1 # number of samples in each batch
data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=15) # len(data_loader) = num_batches = 900*5 // 15 = 300
batches = list(data_loader) # a list of 300 lists\batches

# 1) TRAIN THE ENCODER ################################################################################################################

num_epochs = 60

encoder = Encoder().to(device)
#encoder.load_state_dict(torch.load("/content/drive/MyDrive/MODELS/encoder.pth", map_location=torch.device('cpu')))
enc_optimizer = optim.Adam(encoder.parameters(), lr=0.0001, betas=(0.5, 0.999))
encoder.train()

for epoch in range(num_epochs):
    enc_epoch_loss, steps = 0, 0
    pbar = tqdm(data_loader, desc="Epoch {}".format(epoch), total=len(data_loader))
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
        enc_loss = triplet_margin_loss(EA.view(EA.size(0), -1), EPA.view(EPA.size(0), -1), ENA.view(ENA.size(0), -1))
        enc_loss.backward()
        enc_optimizer.step()

        enc_epoch_loss += enc_loss.item()
        steps += 1
        pbar.set_postfix({'Encoder Loss': '{0:.3f}'.format(enc_epoch_loss/steps)})

        if epoch % 10 == 0 and i == 0:
            encoded_mel = EA[0].squeeze().cpu().detach().numpy()  # Assuming you want to visualize the first example in the batch
            plt.imshow(encoded_mel, cmap='hot')
            plt.colorbar()
            plt.show()

    torch.save(encoder.state_dict(), "/content/drive/MyDrive/MODELS/encoder.pth")

# TODO: 2.1) TRAIN THE CUSTOM CLASSIFIER #################################################################################################################

num_epochs = 100

classifier = Classifier().to(device)
classifier.load_state_dict(torch.load("/content/drive/MyDrive/MODELS/classifier.pth", map_location=torch.device('cpu')))
clf_optimizer = optim.Adam(classifier.parameters(), lr=0.0001, betas=(0.5, 0.999))
classifier.train()

for epoch in range(num_epochs):
    clf_epoch_loss, steps = 0, 0
    correct_predictions, total_predictions = 0, 0

    pbar = tqdm(data_loader, desc="Epoch {}".format(epoch), total=len(data_loader))
    for i, dl in enumerate(pbar): # dl[0]: the 3 spectrogram-batches of 15 specs each, dl[1]: the 15 era-labels, dl[2]: the 15 pair-images
        melspecs, eras, pairs = dl
        PI = torch.stack([pair for pair in pairs])

        """for img in PI:
            print(img)
            show_image(img)"""

        actual_eras, CPI = torch.tensor([to_numerical(era) for era in eras]).to(device), classifier(PI.to(device))
        
        clf_optimizer.zero_grad()
        clf_loss = nll_loss(CPI, actual_eras)
        clf_loss.backward()
        clf_optimizer.step()

        predicted_eras = torch.argmax(CPI, dim=1)
        correct_predictions += torch.sum(predicted_eras == actual_eras).item()
        total_predictions += actual_eras.size(0)
        accuracy = correct_predictions / total_predictions

        clf_epoch_loss += clf_loss.item()
        steps += 1
        pbar.set_postfix({'Classifier Loss': '{0:.3f}'.format(clf_epoch_loss/steps), 'Accuracy': '{0:.2%}'.format(accuracy)})

    torch.save(classifier.state_dict(), "/content/drive/MyDrive/MODELS/classifier.pth")

# TODO: 2.2) EVALUATE CUSTOM CLASSIFIER #################################################################################################################

learning_rate = 0.0001
num_epochs = 100

classifier = Classifier().to(device)
classifier.load_state_dict(torch.load("/content/drive/MyDrive/MODELS/classifier.pth", map_location=torch.device('cpu')))
classifier.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # remove that if it's already (64,64)
    transforms.ToTensor()
])

random_image = transform(Image.open("/content/drive/MyDrive/DATASETS/musart-dataset/renaissance/i444.jpg")).unsqueeze(0).to(device)

with torch.no_grad():
    predicted_era = classifier(random_image)
    predicted_era = torch.argmax(predicted_era, dim=1).item()

print("Predicted Era:", predicted_era)

# TODO: LOAD THE ENCODER/CLASSIFIER MODELS

encoder = Encoder().to(device)
encoder_state = torch.load("/content/drive/MyDrive/MODELS/encoder.pth")
encoder.load_state_dict(encoder_state)
encoder.eval()

classifier = Classifier().to(device)
classifier_state = torch.load("/content/drive/MyDrive/MODELS/classifier.pth")
classifier.load_state_dict(classifier_state)
classifier.eval()

# TODO: 3) TRAIN THE GAN #####################################################################################################################

num_epochs = 100
gen_iters = 1
dis_iters = 1

noise_dim = 100
num_classes = len(art_eras)
conditional_noise_dim = noise_dim + num_classes

w1 = 10
w2 = 2
w3 = 100

generator = Generator(noise_dim=conditional_noise_dim).to(device)
#generator.load_state_dict(torch.load("/content/drive/MyDrive/MODELS/generator.pth", map_location=torch.device('cpu')))
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
generator.apply(xavier_weights)
generator.train()

discriminator = Discriminator().to(device)
#discriminator.load_state_dict(torch.load("/content/drive/MyDrive/MODELS/discriminator.pth", map_location=torch.device('cpu')))
dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.00001, betas=(0.5, 0.999))
discriminator.apply(xavier_weights)
discriminator.train()

#vgg19 = VGG19().to(device)
#vgg19.eval()

for epoch in range(num_epochs):
    gen_epoch_loss, dis_epoch_loss, steps = 0, 0, 0
    pbar = tqdm(data_loader, desc="Epoch {}".format(epoch), total=len(data_loader))
    for i, dl in enumerate(pbar):
        melspecs, eras, pairs = dl
        #melspecs = torch.transpose(torch.stack(melspecs, dim=0), 0, 1)
        noise = torch.randn(BATCH_SIZE, noise_dim).to(device)

        #actual_eras = torch.tensor([to_numerical(era) for era in eras])
        #encoding = torch.eye(num_classes)[actual_eras].to(device)

        encoding = encoder(melspecs.to(device))
        input = torch.cat((noise, encoding), dim=1).to(device)

        SI = generator(input).to(device) # (15,3,64,64)
        PI = torch.stack([pair for pair in pairs]).requires_grad_(True).to(device)
        PI = transpose_image(PI, range_min=-1, range_max=1)

        #CSI = classifier(SI)
        #VSI = vgg19(SI)
        #VPI = vgg19(PI)

        for _ in range(dis_iters):
            dis_optimizer.zero_grad()
            logSI, sigSI = discriminator(SI.detach())
            logPI, sigPI = discriminator(PI)
            logPI.requires_grad_(); sigPI.requires_grad_()
            dis_loss = w1 * stDis_loss(logSI, sigSI, logPI, sigPI)
            dis_loss.backward(retain_graph=True)
            dis_optimizer.step()
            dis_epoch_loss += dis_loss.item()

        for _ in range(gen_iters):
            gen_optimizer.zero_grad()
            logSI, sigSI = discriminator(SI)
            gen_loss = w1 * stGen_loss(logSI, sigSI) #+ w2 * nll_loss(CSI, actual_eras.to(device)) + w3 * l1_loss(SI, PI)
            gen_loss.backward(retain_graph=True)
            gen_optimizer.step()
            gen_epoch_loss += gen_loss.item()

        steps += 1
        pbar.set_postfix({'Generator Loss': '{0:.3f}'.format(gen_epoch_loss / (steps * gen_iters)),
                          'Discriminator Loss': '{0:.3f}'.format(dis_epoch_loss / (steps * dis_iters))})

    if epoch % 1 == 0:
        SI = transpose_image(SI, range_min=0, range_max=1)
        show_image(SI[0])

    torch.save(generator.state_dict(), "/content/drive/MyDrive/MODELS/generator.pth")
    torch.save(discriminator.state_dict(), "/content/drive/MyDrive/MODELS/discriminator.pth")

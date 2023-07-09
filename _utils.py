import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import librosa
from torch.nn import LogSoftmax
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.image as mpimg

def show_spectrogram(name, ms, sr, hop_length):
    ms = librosa.power_to_db(ms, ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(ms, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel-frequency spectrogram - {name}")
    plt.xlabel('Time')
    plt.ylabel('Hz')
    plt.show()

def save_collage(images, size=(6,6)):
    for era, imgs in images.items():
        collage_rows = []
        x, y = size
        for row in range(y):
            collage_row = None
            for col in range(x):
                image = transpose_image(imgs[row*x + col], range_min=0, range_max=1)
                image = image.transpose(1, 2, 0)
                if collage_row is None:
                    collage_row = image
                else:
                    collage_row = np.concatenate((collage_row, image), axis=1)
            collage_rows.append(collage_row)

        collage = np.concatenate(collage_rows, axis=0)
        output_dir = '/content/drive/MyDrive/DATA/collages'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{era}.png")
        mpimg.imsave(output_path, collage)

def save_images(style_image, pair_image, epoch):
    style_image = style_image.detach().cpu().numpy()
    pair_image = pair_image.detach().cpu().numpy()
    image = np.concatenate((style_image.transpose(1, 2, 0), pair_image.transpose(1, 2, 0)), axis=1)
    output_dir = '/content/drive/MyDrive/DATA/gan_logs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"gan_{epoch}.png")
    mpimg.imsave(output_path, image)

def save_encodings(eras_dict, enc_dim, epoch, count):
    plt.figure()
    for era, encoding in eras_dict.items():
        plt.plot(range(0, enc_dim), encoding, label=era.capitalize())
        eras_dict[era] = None
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'Audio Encodings - Epoch {epoch}')
    plt.legend()
    output_dir = '/content/drive/MyDrive/DATA/enc-logs'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"enc_{epoch}_{count}.png")
    plt.savefig(output_path)
    plt.close()
    return eras_dict

def monitor_encoder(losses, num_epochs):
    epochs = range(0, num_epochs)
    plt.plot(epochs, losses, label='Encoder')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    output_dir = '/content/drive/MyDrive/DATA/losses'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'enc_loss.png')
    plt.savefig(output_path)
    plt.close()

def monitor_gan(losses, num_epochs):
    epochs = range(0, num_epochs)
    plt.plot(epochs, losses['gen'], label='Generator')
    plt.plot(epochs, losses['dis'], label='Discriminator')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    output_dir = '/content/drive/MyDrive/DATA/losses'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'gan_loss.png')
    plt.savefig(output_path)
    plt.close()

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
    TML = nn.TripletMarginLoss(margin=3.0, p=5)
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

def transpose_image(image, range_min=0, range_max=1):
    min_value, max_value = image.min(), image.max()
    scaled_image = (image - min_value) / (max_value - min_value)
    transposed_image = (scaled_image * (range_max - range_min)) + range_min
    return transposed_image

def xavier_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)

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

def batch_generator(data_loader):
    for batch in data_loader:
        yield batch
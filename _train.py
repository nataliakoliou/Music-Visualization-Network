import torch
import random
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from _models import *
from _utils import *

def enc_logs(batch_size, enc_dim, eras, eras_dict, encodings, epoch, count):
    random_sample = random.randint(0, batch_size - 1)
    for era in eras_dict.keys():
        if eras[random_sample] == era:
            eras_dict[era] = encodings[random_sample].squeeze().cpu().detach().numpy()   
    if np.all([v is not None for v in eras_dict.values()]):
        eras_dict = save_encodings(eras_dict, enc_dim, epoch, count)
        count += 1
    return eras_dict, count

def gan_logs(batch_size, epoch, style_images, pair_images):
    style_images = transpose_image(style_images, range_min=0, range_max=1)
    pair_images = transpose_image(pair_images, range_min=0, range_max=1)
    random_sample = random.randint(0, batch_size - 1)
    save_images(style_images[random_sample], pair_images[random_sample], epoch)

def train_encoder(device, data, enc_dim=10, batch_size=1, num_epochs=15, visualize=True):
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=2)
    losses, batches = [], [batch for batch in batch_generator(data_loader)]

    encoder = Encoder().to(device)
    enc_optimizer = optim.Adam(encoder.parameters(), lr=0.00005, betas=(0.5, 0.999))
    encoder.train()

    for epoch in range(num_epochs):
        enc_epoch_loss, steps, count = 0, 0, 0
        eras_dict = {"renaissance": None, "baroque": None, "classical": None, "romantic": None, "modern": None}
        pbar = tqdm(data_loader, desc="Epoch {}".format(epoch), total=len(data_loader))
        for _, dl in enumerate(pbar):
            melspecs, eras, _ = dl
            melspecs = torch.transpose(torch.stack(melspecs, dim=0), 0, 1)
            pos_melspecs, neg_melspecs = [], []

            for sample in range(batch_size):
                pos, neg = False, False

                while not pos or not neg:
                    rbID = random.randint(0, len(data_loader) - 1)
                    rand_melspecs, rand_eras = torch.transpose(torch.stack(batches[rbID][0], dim=0), 0, 1), batches[rbID][1]
                    rsID = random.randint(0, batch_size - 1)
                    rand_melspec, rand_era = rand_melspecs[rsID], rand_eras[rsID]

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
            eras_dict, count = enc_logs(batch_size, enc_dim, eras, eras_dict, EA, epoch, count) if visualize and count < 5 else (eras_dict, count)

        losses.append(enc_epoch_loss / steps)
        torch.save(encoder.state_dict(), "/content/drive/MyDrive/MODELS/encoder.pth")

    monitor_encoder(losses, num_epochs)

def train_gan(device, data, enc_dim=10, batch_size=15, num_epochs=100, gen_iters=1, dis_iters=1, noise_dim=100, visualize=True):
    gan_losses = {'gen': [], 'dis': []}
    conditional_noise_dim = noise_dim + enc_dim
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=2)

    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load("/content/drive/MyDrive/MODELS/encoder.pth"))
    encoder.eval()

    generator = Generator(in_dim=conditional_noise_dim).to(device)
    gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    generator.apply(xavier_weights)
    generator.train()

    discriminator = Discriminator().to(device)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))
    discriminator.apply(xavier_weights)
    discriminator.train()

    for epoch in range(num_epochs):
        gen_epoch_loss, dis_epoch_loss, steps = 0, 0, 0
        random_batch = random.randint(0, len(data_loader) - 1)
        pbar = tqdm(data_loader, desc="Epoch {}".format(epoch), total=len(data_loader))
        for i, dl in enumerate(pbar):
            melspecs, _, pairs = dl
            melspecs = torch.transpose(torch.stack(melspecs, dim=0), 0, 1)
            noise = torch.randn(batch_size, noise_dim).to(device)

            encoding = encoder(melspecs.to(device))
            input = torch.cat((noise, encoding), dim=1).to(device)

            SI = generator(input).to(device)
            PI = torch.stack([pair for pair in pairs]).requires_grad_(True).to(device)
            PI = transpose_image(PI, range_min=-1, range_max=1)

            for _ in range(dis_iters):
                dis_optimizer.zero_grad()
                logSI, sigSI = discriminator(SI.detach())
                logPI, sigPI = discriminator(PI)
                logPI.requires_grad_(); sigPI.requires_grad_()
                dis_loss = stDis_loss(logSI, sigSI, logPI, sigPI)
                dis_loss.backward(retain_graph=False)
                dis_optimizer.step()
                dis_epoch_loss += dis_loss.item()

            for _ in range(gen_iters):
                gen_optimizer.zero_grad()
                logSI, sigSI = discriminator(SI)
                gen_loss = stGen_loss(logSI, sigSI)
                gen_loss.backward(retain_graph=False)
                gen_optimizer.step()
                gen_epoch_loss += gen_loss.item()

            steps += 1
            pbar.set_postfix({'Generator Loss': '{0:.3f}'.format(gen_epoch_loss / (steps * gen_iters)),
                            'Discriminator Loss': '{0:.3f}'.format(dis_epoch_loss / (steps * dis_iters))})

            gan_logs(batch_size, epoch, SI, PI) if visualize and random_batch == i else None

        gan_losses['gen'].append(gen_epoch_loss / (steps * gen_iters))
        gan_losses['dis'].append(dis_epoch_loss / (steps * dis_iters))
        monitor_gan(gan_losses, epoch+1)

        torch.save(generator.state_dict(), "/content/drive/MyDrive/MODELS/generator.pth")
        torch.save(discriminator.state_dict(), "/content/drive/MyDrive/MODELS/discriminator.pth")

def train(device):
    dataset_path = "/content/drive/MyDrive/DATASETS/musart-dataset/training"
    art_eras = ["renaissance", "baroque", "classical", "romantic", "modern"]
    prepared = input("<training_data.npy> already exists in ~/content/drive/MyDrive/DATA (y/n): ")

    if prepared == "n":
        data = prepare_data(dataset_path, art_eras, sr=22050, n_fft=1024, hop_length=256, n_mels=128, display=False)
        np.save('/content/drive/MyDrive/DATA/training_data.npy', np.array(data))
    elif prepared == "y":
        data = np.load('/content/drive/MyDrive/DATA/training_data.npy', allow_pickle=True)
        data = data.tolist()
    else:
        print("Invalid choice. Please select 'y' or 'n'.")

    #train_encoder(device, data, enc_dim=10, batch_size=1, num_epochs=15, visualize=True)
    train_gan(device, data, enc_dim=10, batch_size=15, num_epochs=300, gen_iters=1, dis_iters=1, noise_dim=100, visualize=True)
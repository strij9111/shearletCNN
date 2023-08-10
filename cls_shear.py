from scipy.io import wavfile
from torch.utils.mobile_optimizer import optimize_for_mobile
from pytorch_metric_learning import losses
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import torchaudio.functional as TAF
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
from typing import List, Tuple
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import functools
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import os
import sys
from audiomentations import Compose, SevenBandParametricEQ, RoomSimulator, AirAbsorption, TanhDistortion, TimeStretch, PitchShift, AddGaussianNoise, Gain, Shift, BandStopFilter, AddBackgroundNoise, PolarityInversion
import pyshearlab

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

root_dir = "e:\\dvc"
batch_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

C1_kernel_size = (8,20) # Customary to use odd and square kernel/filter size fxf 
num_filters_conv1 = 256
num_filters_conv2 = 512
C2_kernel_size = (4,10) # Customary to use odd and square kernel/filter size fxf 
mp2d_size = 4      # MaxPooling2d window size (= stride)
#fc1_in_size = 32*12*16
fc1_out_size = 256
fc2_out_size = 100
avg_pool_size = 9


class M5(nn.Module):
    def __init__(self):
        super(M5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=33, out_channels=num_filters_conv1, kernel_size=C1_kernel_size)
        self.bn1   = nn.BatchNorm2d(num_filters_conv1)
        self.conv2 = nn.Conv2d(num_filters_conv1, num_filters_conv2, C2_kernel_size)
        self.bn2   = nn.BatchNorm2d(num_filters_conv2)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((avg_pool_size, avg_pool_size))
        self.fc1 = nn.Linear(num_filters_conv2*avg_pool_size*avg_pool_size, fc1_out_size)
        self.bn3   = nn.BatchNorm1d(fc1_out_size)
        self.fc2 = nn.Linear(fc1_out_size, fc2_out_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # переупорядочивание размерностей входного тензора
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), mp2d_size))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), mp2d_size))
        x = self.adaptive_avg_pool(x)
        x = x.reshape(-1, num_filters_conv2*avg_pool_size*avg_pool_size)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

        
        
class CommandDataset(Dataset):

    def __init__(self, meta, root_dir, sample_rate, labelmap, augment=True):
        self.meta = meta
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.labelmap = labelmap
        self.augment = augment
        self.sigma = 30
        self.scales = 3
        self.thresholdingFactor = 1
        
        n_mels = 251
        hop_length = 64
        sample_rate = 16000 
        
        self.shearletSystem = pyshearlab.SLgetShearletSystem2D(0, 251, 251, self.scales)
        self.resampler = torchaudio.transforms.Resample(orig_freq=32000, new_freq=16000)
        
        if self.augment:
            self.augmentations = Compose([
#                TimeStretch(min_rate=0.8, max_rate=1.2, p=0.1),
#                PitchShift(min_semitones=-6, max_semitones=6, p=0.1),
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.030, p=0.1),
                Gain(min_gain_in_db=-3, max_gain_in_db=3, p=0.1),
#                BandStopFilter(min_bandwidth_fraction=0.01, max_bandwidth_fraction=0.25, p=0.1),
#                PolarityInversion(p=0.1),
#                Shift(min_fraction=-0.1, max_fraction=0.1, p=0.1),
                AirAbsorption(p=0.4),
                TanhDistortion(p=0.1),
#                SevenBandParametricEQ(p=0.3)
            ])
        
        self.transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=2048, hop_length=hop_length, n_mels=n_mels),
#            torchaudio.transforms.TimeMasking(time_mask_param=int(0.2 * 16000/160)),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        )

#        self.transform_spec = torchaudio.transforms.Spectrogram(n_fft=320, hop_length=160, power=None)
#        self.transform_inverse = torchaudio.transforms.InverseSpectrogram(n_fft=320, hop_length=160)
    
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.meta['path'].iloc[idx]
        signal, sample_rate = torchaudio.load(file_name)
        signal = self.resampler(signal)
        
        if signal.shape[1] < 16000:
            padding_size = 16000 - signal.shape[1]
            signal = F.pad(signal, (0, padding_size))

        if signal.shape[1] > 16000:
            start = torch.randint(0, signal.shape[1] - 16000, (1,)).item()
            signal = signal[:, start: start + 16000]
            

#        signal = self.transform_spec(signal)
#        signal = self.transform_inverse(signal)
        
        if self.augment:
            signal = self.augmentations(samples=signal.numpy(), sample_rate=16000)
            signal = torch.from_numpy(signal)
            
            
        spec = self.transform(signal)
        spec = spec.detach().cpu().numpy().astype('float32')
        coeffs = pyshearlab.SLsheardec2D(spec, self.shearletSystem)

        # thresholding
        oldCoeffs = coeffs.copy()
        weights = np.ones(coeffs.shape)

        for j in range(len(self.shearletSystem["RMS"])):
            weights[:,:,j] = self.shearletSystem["RMS"][j]*np.ones((251, 251))
            
        coeffs = np.real(coeffs)
        zero_indices = np.abs(coeffs) / (self.thresholdingFactor * weights * self.sigma) < 1
        coeffs[zero_indices] = 0
        
        label = self.meta['label'].iloc[idx]

        return torch.from_numpy(coeffs), self.labelmap[label]


labels = {}
for i in range(100):
    key = f's{i+1}'
    labels[key] = i


data = pd.DataFrame([
    {'label': i[0].split("\\")[-1], 'path': i[0] + "\\" + j}
    for i in os.walk(root_dir)
    for j in i[2]
])

# print(data.label.value_counts())
train, val, _, _ = train_test_split(data, data['label'], test_size=0.2)

train_dataset = CommandDataset(
    meta=train, root_dir=root_dir, sample_rate=16000, labelmap=labels, augment=False)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

val_dataset = CommandDataset(
    meta=val, root_dir=root_dir, sample_rate=16000, labelmap=labels, augment=False)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

model = M5()
model.to(device)

EPOCHS = 50
lr = 0.001
best_val_loss = float('inf')
epochs_without_improvement = 0

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

if __name__ == '__main__':
    for epoch in range(EPOCHS):

        model.train()

        train_loss = []
        for batch, targets in tqdm(train_dataloader, desc=f"Epoch: {epoch}"):
            optimizer.zero_grad()
            batch = batch.to(device)
            targets = targets.to(device)
            predictions = model(batch)

            loss = F.nll_loss(predictions, targets)
            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

        print('Training loss:', np.mean(train_loss))

        model.eval()

        val_loss = []
        correct = 0
        all_preds = []
        all_targets = []

        for batch, targets in tqdm(val_dataloader, desc=f"Epoch: {epoch}"):

            with torch.no_grad():

                batch = batch.to(device)
                targets = targets.to(device)
                input = batch
                predictions = model(batch)

                loss = F.nll_loss(predictions, targets)

                pred = get_likely_index(predictions).to(device)
                correct += number_of_correct(pred, targets)

                val_loss.append(loss.item())

                # Сохраняем предсказания и метки
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Подсчет F-меры
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 score: {f1:.2f}')
        
        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            epochs_without_improvement = 0
#            torch.save(model.state_dict(), 'best_model.pth')
            traced_model = torch.jit.trace(model, input)
            traced_model.save('model_traced.pt')

#            model_dynamic_quantized = torch.quantization.quantize_dynamic(
#                traced_model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
#            traced_quantized_model = torch.jit.trace(
#                model_dynamic_quantized, input, strict=False)
#            optimized_traced_quantized_model = optimize_for_mobile(
#                traced_quantized_model)
#            optimized_traced_quantized_model.save('model_traced.pt')
#            optimized_traced_quantized_model._save_for_lite_interpreter("best_model.ptl")
        else:
            epochs_without_improvement += 1

        # Ранняя остановка, если количество эпох без улучшений превысило patience
        if epochs_without_improvement >= 20:
            print(f'Early stopping at epoch {epoch + 1}')
#            break

        print(
            f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(val_dataloader.dataset)} ({100. * correct / len(val_dataloader.dataset):.0f}%)\n")
        print('Val loss:', np.mean(val_loss))

        scheduler.step()

    torch.save(model.state_dict(), 'model.pth')

import math, random
import torch
import torchaudio
from torchaudio import transforms
import librosa
import librosa.display
from IPython.display import Audio
import matplotlib.pyplot as plt
import numpy as np

def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)

def rechannel(aud, new_channel):
    sig, sr = aud

    if sig.shape[0] == new_channel:
        # Nothing to do
        return aud

    if new_channel == 1:
        # Convert from stereo to mono by selecting only the first channel
        resig = sig[:1, :]
    else:
        # Convert from mono to stereo by duplicating the first channel
        resig = torch.cat([sig, sig])

    return (resig, sr)

def resample(aud, newsr):
    sig, sr = aud

    if sr == newsr:
        # Nothing to do
        return aud

    num_channels = sig.shape[0]
    # Resample first channel
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])

    if num_channels > 1:
        # Resample the second channel and merge both channels
        retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
        resig = torch.cat([resig, retwo])

    return (resig, newsr)

def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr // 1000 * max_ms

    if sig_len > max_len:
        # Truncate the signal to the given length
        sig = sig[:, :max_len]
    elif sig_len < max_len:
        # Length of padding to add at the beginning and end of the signal
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len

        # Pad with 0s
        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))

        sig = torch.cat((pad_begin, sig, pad_end), 1)

    return (sig, sr)

def time_shift(aud, shift_limit):
    sig, sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)

def spectro_gram(aud, n_mels=128, n_fft=400, hop_len=None):
    sig, sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)

def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
        aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
        aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec

import librosa

def display_mfcc(spec, aud):
    sig, sr = aud

    # Tính toán Mel spectrogram từ spec
  

    # Tính toán MFCC từ Mel spectrogram
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(spec), n_mfcc=13)

    plt.figure(figsize=(12, 8))

    # Thay đổi mã màu ở đây (cmap='viridis')

    # Display MFCC for Channel 1
    plt.subplot(2, 1, 1)
    mfcc_ch1 = np.squeeze(mfcc[0])
    plt.imshow(mfcc_ch1, cmap='viridis', aspect='auto', origin='lower')
    plt.title('MFCC - Channel 1')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')

    # Display MFCC for Channel 2
    plt.subplot(2, 1, 2)
    mfcc_ch2 = np.squeeze(mfcc[1])
    plt.imshow(mfcc_ch2, cmap='viridis', aspect='auto', origin='lower')
    plt.title('MFCC - Channel 2')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')

    plt.tight_layout()
    plt.show()



def main():
    audio_file = "Hello.wav"
    aud = open(audio_file)

    # Display original audio
    print("Original Audio:")

    # Perform transformations and display processed audio
    aud = rechannel(aud, 2)  # Convert to 2 channel
    aud = resample(aud, 88200)  # Resample to 88200 Hz
    aud = pad_trunc(aud, 2000)  # Pad or truncate to a length of 5000 ms
    a=2.0/100
    aud = time_shift(aud,a)
    print("Processed Audio:")

    # Create and display Spectrogram
    spec = spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)
    print("Spectrogram:")

    # Apply augmentations and display augmented Spectrogram
    # (Work in progress)

    # Display MFCC from Mel Spectrogram
    display_mfcc(spec, aud)

if __name__ == "__main__":
    main()

import os
import random
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm
from sklearn.model_selection import train_test_split

#config
INPUT_DIR = r"C:\Users\USER\Desktop\wavefake\LJSpeech-1.1\LJSpeech-1.1\wavs"   #path to original LJSpeech wav files
OUTPUT_DIR = r"D:\Datasets\processed_ljspeech"
SAMPLE_RATE = 22050
N_MELS = 80
FRAME_SIZE = 2048
HOP_LENGTH = 512
SEED = 42

#utility functions
def normalize_audio(y):
    """Normalize amplitude to -1 to 1"""
    return y / np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else y

def trim_silence(y):
    """Trim leading/trailing silence"""
    yt, _ = librosa.effects.trim(y, top_db=20)
    return yt

def save_mel_png(mel, save_path):
    """Save mel-spectrogram as PNG"""
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(mel, aspect='auto', origin='lower')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_file(file_path, output_audio_path, output_mel_path, output_png_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    y = trim_silence(y)
    y = normalize_audio(y)

  
    sf.write(output_audio_path, y, sr)

    #create Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=FRAME_SIZE,
                                         hop_length=HOP_LENGTH, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    #normalize spectrogram (mean=0, std=1)
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    np.save(output_mel_path, mel_db)
    save_mel_png(mel_db, output_png_path)

#splitting dataset
def split_dataset(file_list):
    train, test_val = train_test_split(file_list, test_size=0.2, random_state=SEED)
    val, test = train_test_split(test_val, test_size=0.5, random_state=SEED)
    return train, val, test

#exec
def main():
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".wav")]
    train, val, test = split_dataset(all_files)

    sets = {'train': train, 'val': val, 'test': test}
    for set_name, files in sets.items():
        for subfolder in ['audio', 'mel', 'images']:
            os.makedirs(os.path.join(OUTPUT_DIR, set_name, subfolder), exist_ok=True)

        print(f"Processing {set_name} set ({len(files)} files)...")
        for file in tqdm(files):
            src_path = os.path.join(INPUT_DIR, file)
            base_name = os.path.splitext(file)[0]

            out_audio = os.path.join(OUTPUT_DIR, set_name, 'audio', f"{base_name}.wav")
            out_mel = os.path.join(OUTPUT_DIR, set_name, 'mel', f"{base_name}.npy")
            out_png = os.path.join(OUTPUT_DIR, set_name, 'images', f"{base_name}.png")

            process_file(src_path, out_audio, out_mel, out_png)

    print("preprocessing is donwe.")

if __name__ == "__main__":
    main()

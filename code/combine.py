import os
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import csv

#congif
LJSPEECH_PATH = r"D:\Datasets\processed_ljspeech"
WAVEFAKE_PATH = r"D:\processed_wavefake"
MERGED_PATH = r"C:\dataset"  # SSD recommended
RANDOM_SEED = 42
NUM_WORKERS = 8  # for parallel copying

#func
def copy_file_pair(src_audio, src_mel, dst_audio, dst_mel):
    shutil.copy2(src_audio, dst_audio)
    shutil.copy2(src_mel, dst_mel)

def process_split(split_name):
    lj_audio_dir = os.path.join(LJSPEECH_PATH, split_name, 'audio')
    wf_audio_dir = os.path.join(WAVEFAKE_PATH, split_name, 'audio')

    lj_files = sorted([os.path.join(lj_audio_dir, f) for f in os.listdir(lj_audio_dir) if f.endswith('.wav')])
    wf_files = sorted([os.path.join(wf_audio_dir, f) for f in os.listdir(wf_audio_dir) if f.endswith('.wav')])

    #balance fake samples per split
    if len(wf_files) > len(lj_files):
        np.random.seed(RANDOM_SEED)
        wf_files = list(np.random.choice(wf_files, len(lj_files), replace=False))

    #create destination dirs
    dst_audio_real = os.path.join(MERGED_PATH, split_name, 'audio', 'real')
    dst_audio_fake = os.path.join(MERGED_PATH, split_name, 'audio', 'fake')
    dst_mel_real = os.path.join(MERGED_PATH, split_name, 'mel', 'real')
    dst_mel_fake = os.path.join(MERGED_PATH, split_name, 'mel', 'fake')

    for folder in [dst_audio_real, dst_audio_fake, dst_mel_real, dst_mel_fake]:
        os.makedirs(folder, exist_ok=True)

    #copy real files
    real_labels = []
    real_pairs = []
    for f in lj_files:
        fname = os.path.basename(f)
        mel_file = os.path.join(LJSPEECH_PATH, split_name, 'mel', fname.replace('.wav', '.npy'))
        dst_audio = os.path.join(dst_audio_real, fname)
        dst_mel = os.path.join(dst_mel_real, fname.replace('.wav', '.npy'))
        real_pairs.append((f, mel_file, dst_audio, dst_mel))
        real_labels.append((os.path.relpath(dst_audio, MERGED_PATH), 0))

    #copy fake files
    fake_labels = []
    fake_pairs = []
    for f in wf_files:
        fname = os.path.basename(f)
        mel_file = os.path.join(WAVEFAKE_PATH, split_name, 'mel', fname.replace('.wav', '.npy'))
        dst_audio = os.path.join(dst_audio_fake, fname)
        dst_mel = os.path.join(dst_mel_fake, fname.replace('.wav', '.npy'))
        fake_pairs.append((f, mel_file, dst_audio, dst_mel))
        fake_labels.append((os.path.relpath(dst_audio, MERGED_PATH), 1))

    #parallel copy
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(lambda p: copy_file_pair(*p), real_pairs),
                  total=len(real_pairs), desc=f"{split_name} real"))
        list(tqdm(executor.map(lambda p: copy_file_pair(*p), fake_pairs),
                  total=len(fake_pairs), desc=f"{split_name} fake"))

    #save labels CSV
    csv_path = os.path.join(MERGED_PATH, 'logs', f"{split_name}_labels.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'label'])
        writer.writerows(real_labels + fake_labels)

    print(f"{split_name}: merged {len(real_labels)} real + {len(fake_labels)} fake files")

#main prcedure
def merge_datasets():
    for split in ['train', 'val', 'test']:
        process_split(split)
    print("\nDataset merge complete!")

if __name__ == "__main__":
    merge_datasets()

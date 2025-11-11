import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

#config
INPUT_DIR = r"D:\AOT wall\wavefake\generated_audio"  #waveFake root
OUTPUT_ROOT = r"D:\processedWavefakeEnglish"       #root for processed folders

SAMPLE_RATE = 22050
N_MELS = 80
FRAME_SIZE = 2048
HOP_LENGTH = 512
NUM_WORKERS = max(1, cpu_count() - 1)  #using all CPU cores minus 1 for faster procfessing


def normalize_audio(y):
    return y / np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else y

def trim_silence(y):
    yt, _ = librosa.effects.trim(y, top_db=20)
    return yt

def save_mel_png(mel, save_path):
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(mel, aspect='auto', origin='lower')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_file(args):
    file_path, output_folder = args
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    out_audio = os.path.join(output_folder, 'audio', f"{base_name}.wav")
    out_mel = os.path.join(output_folder, 'mel', f"{base_name}.npy")
    out_png = os.path.join(output_folder, 'images', f"{base_name}.png")

    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        y = trim_silence(y)
        y = normalize_audio(y)

    
        sf.write(out_audio, y, SAMPLE_RATE)

       
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=FRAME_SIZE,
                                             hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

        np.save(out_mel, mel_db)
        save_mel_png(mel_db, out_png)
    except Exception as e:
        print(f"error processing {file_path}: {e}")


def main():
    folders = [f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))]
    folders = [f for f in folders if not f.lower().startswith("just")]

    print(f"english aaveFake folders to process: {folders}")

    for folder in folders:
        input_folder = os.path.join(INPUT_DIR, folder)
        output_folder = os.path.join(OUTPUT_ROOT, folder)

        #create subfolders
        for sub in ['audio', 'mel', 'images']:
            os.makedirs(os.path.join(output_folder, sub), exist_ok=True)

        #collect all wav files recursively
        wav_files = []
        for root, _, files in os.walk(input_folder):
            for f in files:
                if f.endswith(".wav"):
                    wav_files.append(os.path.join(root, f))

        print(f"processing folder '{folder}' with {len(wav_files)} files using {NUM_WORKERS} workers.")

        #prepare arguments for multiprocessing
        args_list = [(f, output_folder) for f in wav_files]

        #multiprocessing Pool
        with Pool(NUM_WORKERS) as pool:
            list(tqdm(pool.imap(process_file, args_list), total=len(args_list), desc=folder))

    print("all aaveFake english folders processed.")

if __name__ == "__main__":
    main()

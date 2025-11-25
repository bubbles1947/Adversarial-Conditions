import os
import librosa
import numpy as np
from tqdm import tqdm
import hashlib
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed


dataset_dir = r"C:\Users\USER\Desktop\498r\for-2sec\for-2seconds\testing"
output_dir = r"C:\Users\USER\Desktop\498r\for"
os.makedirs(output_dir, exist_ok=True)

sr = 22050
n_mels = 128
hop_length = 256
n_fft = 1024
target_shape = (224, 224)

hashes = set()

def process_file(file_path, class_output_dir):
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)

        h = hashlib.md5(y.tobytes()).hexdigest()
        if h in hashes:
            return None
        hashes.add(h)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                             hop_length=hop_length, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_resized = cv2.resize(mel_db, target_shape, interpolation=cv2.INTER_CUBIC)


        out_file = os.path.join(class_output_dir, os.path.splitext(os.path.basename(file_path))[0] + ".npy")
        np.save(out_file, mel_resized)
        return out_file
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

all_files = []
class_dirs = ['real', 'fake']
for class_name in class_dirs:
    class_input_dir = os.path.join(dataset_dir, class_name)
    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    for file in os.listdir(class_input_dir):
        if file.endswith(('.wav', '.flac', '.mp3')):
            all_files.append((os.path.join(class_input_dir, file), class_output_dir))

#threadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=8) as executor:  
    futures = [executor.submit(process_file, fpath, out_dir) for fpath, out_dir in all_files]
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
        pass  #just to show progress bar

print("all files processed.")

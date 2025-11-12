
import os
import numpy as np
import librosa
from tqdm import tqdm
import pyloudnorm as pyln
import soundfile as sf
from sklearn.model_selection import train_test_split

#config
CONFIG = {
    'wavefake_path': r"D:\AOT wall\wavefake\generated_audio",
    'output_path': r"D:\processed_wavefake",
    'sample_rate': 22050,
    'n_mels': 80,
    'n_fft': 2048,
    'hop_length': 512,
    'test_size': 0.1,
    'val_size': 0.1,
    'random_seed': 42,
    'lufs_target': -25.0
}

#functions
def preprocess_audio_file(audio_path, target_sr=22050, lufs_target=-25.0):
    try:
        #load audio
        audio, sr = librosa.load(audio_path, sr=target_sr)

        #trim silence
        intervals = librosa.effects.split(audio, top_db=40)
        audio_trimmed = np.concatenate([audio[start:end] for start, end in intervals])

        #LUFS normalization
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_trimmed)
        audio_norm = pyln.normalize.loudness(audio_trimmed, loudness, lufs_target)

        #prevent clipping
        audio_norm = np.clip(audio_norm, -1.0, 1.0)

        #mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_norm,
            sr=sr,
            n_mels=CONFIG['n_mels'],
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length']
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        #standardization
        mel_spec_std = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-9)

        return audio_norm, mel_spec_std

    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None, None

def save_outputs(audio, mel, out_audio_path, out_mel_path):
    #save normalized audio
    sf.write(out_audio_path, audio, CONFIG['sample_rate'], subtype='PCM_16')
    #save mel spectrogram
    np.save(out_mel_path, mel.astype(np.float32))

#main procesing
def preprocess_wavefake():
    base_out = CONFIG['output_path']
    splits = ['train', 'val', 'test']

    #create directories
    for split in splits:
        for folder in ['audio', 'mel']:
            os.makedirs(os.path.join(base_out, split, folder), exist_ok=True)
    os.makedirs(os.path.join(base_out, 'logs'), exist_ok=True)

    #gather all valid .wav files, skip unwanted folders
    all_files = []
    for root, dirs, files in os.walk(CONFIG['wavefake_path']):
        #skip folders starting with 'just' or 'just_full_band_melgan'
        if any(os.path.basename(root).lower().startswith(skip) for skip in ['just', 'just_full_band_melgan']):
            continue
        for f in files:
            if f.endswith('.wav'):
                all_files.append(os.path.join(root, f))

    #train/Val/Test split
    train_val, test = train_test_split(all_files, test_size=CONFIG['test_size'], random_state=CONFIG['random_seed'])
    train, val = train_test_split(train_val, test_size=CONFIG['val_size'], random_state=CONFIG['random_seed'])
    splits_dict = {'train': train, 'val': val, 'test': test}

    #process files
    for split_name, files in splits_dict.items():
        print(f"\nProcessing {split_name} set ({len(files)} files)...")
        for fpath in tqdm(files, desc=split_name):
            fname = os.path.splitext(os.path.basename(fpath))[0]
            audio_out = os.path.join(base_out, split_name, 'audio', f"{fname}.wav")
            mel_out = os.path.join(base_out, split_name, 'mel', f"{fname}.npy")

            audio_norm, mel_std = preprocess_audio_file(fpath, CONFIG['sample_rate'], CONFIG['lufs_target'])
            if audio_norm is not None:
                save_outputs(audio_norm, mel_std, audio_out, mel_out)

    print("\nwaveFake preprocessing complete")

if __name__ == "__main__":
    preprocess_wavefake()
